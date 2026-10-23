"""Original Piper downloader contracts against an owned HTTP peer."""

import base64
import contextlib
import hashlib
import http.server
import importlib.abc
import io
import json
import logging
import os
from pathlib import Path
import sys
import tempfile
import threading
import unittest
import zlib

from standalone import local_piper_download_voices as downloader


class NoNative(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.', 1)[0] in {
            'ctypes', 'subprocess', 'numpy', 'onnxruntime', 'piper', 'espeakbridge',
            'torch', 'pip', 'uv', 'pm',
        }:
            raise AssertionError('Native dependency or installer in Piper download: ' + fullname)


def _exercise(module, case, payloads):
    """Observe file effects and real urllib HTTP requests, without replacing urlopen."""
    records = []
    requests = []
    server_errors = []
    plan = list(case.get('responses', []))

    class Handler(http.server.BaseHTTPRequestHandler):
        protocol_version = 'HTTP/1.0'

        def log_message(self, *_):
            pass

        def do_GET(self):
            index = len(requests)
            requests.append({
                'method': self.command, 'path': self.path,
                'accept_encoding': self.headers.get('Accept-Encoding'),
                'connection': self.headers.get('Connection'),
            })
            if index >= len(plan):
                server_errors.append('unexpected request: ' + self.path)
                response = {'status': 599}
            else:
                response = plan[index]
                if response['path'] != self.path:
                    server_errors.append('expected ' + response['path'] + ', got ' + self.path)
            payload = payloads.get(response.get('payload'), b'')
            self.send_response(response.get('status', 200))
            self.send_header('Content-Length', str(response.get('declared_length', len(payload))))
            if 'redirect' in response:
                self.send_header('Location', self.server.peer_url + response['redirect'])
            self.end_headers()
            try:
                self.wfile.write(payload)
            except (BrokenPipeError, ConnectionResetError):
                # Opening the output file may fail after urlopen receives headers.
                pass

    with tempfile.TemporaryDirectory() as temporary:
        base = Path(temporary)
        target = base / 'voices'
        if case.get('create_directory', True):
            target.mkdir()
        for name, value in case.get('initial_files', {}).items():
            path = target / name
            if value == {'directory': True}:
                path.mkdir(parents=True)
            else:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(payloads[value])

        server = http.server.ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        server.daemon_threads = True
        server.peer_url = 'http://127.0.0.1:' + str(server.server_port)
        thread = threading.Thread(target=server.serve_forever, kwargs={'poll_interval': 0.01}, daemon=True)
        previous = module.URL_FORMAT, module.VOICES_JSON, sys.argv
        previous_no_proxy = os.environ.get('no_proxy')
        os.environ['no_proxy'] = '127.0.0.1'
        root_logger = logging.getLogger()
        previous_handlers, previous_level = list(root_logger.handlers), root_logger.level
        root_logger.handlers = []
        module.URL_FORMAT = server.peer_url + '/resources/{lang_family}/{lang_code}/{voice_name}/{voice_quality}/{lang_code}-{voice_name}-{voice_quality}{extension}?download=true'
        module.VOICES_JSON = server.peer_url + '/voices.json?download=true'
        thread.start()
        try:
            for action in case['actions']:
                output, errors = io.StringIO(), io.StringIO()
                failure = None
                with contextlib.redirect_stdout(output), contextlib.redirect_stderr(errors):
                    try:
                        if action['kind'] == 'download':
                            module.download_voice(action['voice'], target, action.get('force', False))
                        elif action['kind'] == 'list':
                            module.list_voices()
                        elif action['kind'] == 'main':
                            sys.argv = ['piper-download-voices'] + [
                                str(target) if value == '$directory' else value
                                for value in action['argv']
                            ]
                            module.main()
                        else:
                            raise AssertionError('Unknown test action: ' + action['kind'])
                    except (Exception, SystemExit) as error:
                        failure = {'type': type(error).__name__, 'message': str(error).replace(str(base), '$temporary').replace(server.peer_url, '$peer')}
                files = {}
                if target.exists():
                    for path in sorted(target.rglob('*')):
                        name = path.relative_to(target).as_posix()
                        files[name] = {'directory': True} if path.is_dir() else {
                            'bytes': path.stat().st_size,
                            'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                        }
                records.append({
                    'failure': failure, 'stdout': output.getvalue(),
                    'stderr': errors.getvalue().replace(str(base), '$temporary').replace(server.peer_url, '$peer').replace(module._LOGGER.name, '$module'),
                    'directory_exists': target.exists(), 'files': files,
                    'requests': list(requests),
                })
        finally:
            module.URL_FORMAT, module.VOICES_JSON, sys.argv = previous
            if previous_no_proxy is None:
                os.environ.pop('no_proxy', None)
            else:
                os.environ['no_proxy'] = previous_no_proxy
            for handler in root_logger.handlers:
                handler.close()
            root_logger.handlers, root_logger.level = previous_handlers, previous_level
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)
            if thread.is_alive():
                raise AssertionError('Owned HTTP peer did not stop')
        if server_errors:
            raise AssertionError(server_errors)
        if len(requests) != len(plan):
            raise AssertionError(('Unused original HTTP response', len(requests), len(plan)))
    return records


class PiperDownloaderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        wrapper = json.loads((Path(__file__).parent / 'fixtures/local_piper_downloader.json').read_text())
        raw = zlib.decompress(base64.b64decode(wrapper['data_zlib_base64']))
        assert hashlib.sha256(raw).hexdigest() == wrapper['data_sha256']
        cls.fixture = json.loads(raw)
        cls.payloads = {name: base64.b64decode(value) for name, value in cls.fixture['payloads'].items()}

    def setUp(self):
        guard = NoNative()
        sys.meta_path.insert(0, guard)
        self.addCleanup(sys.meta_path.remove, guard)

    def test_original_http_download_cache_empty_force_failure_and_partial_file_contracts(self):
        cases = [case for case in self.fixture['cases'] if case['group'] == 'download']
        self.assertEqual(len(cases), 24)
        for case in cases:
            with self.subTest(case=case['name']):
                self.assertEqual(_exercise(downloader, case, self.payloads), case['observed'])

    def test_original_listing_cli_aliases_argument_errors_and_directory_creation(self):
        cases = [case for case in self.fixture['cases'] if case['group'] == 'cli']
        self.assertEqual(len(cases), 13)
        for case in cases:
            with self.subTest(case=case['name']):
                self.assertEqual(_exercise(downloader, case, self.payloads), case['observed'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
