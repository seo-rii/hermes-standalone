"""Qualified storage contracts against actual native storage observations."""
from copy import deepcopy
import asyncio
import contextvars
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

from standalone import relay_atif_storage as storage
NATIVE = json.loads((Path(__file__).parent / 'fixtures/relay_atif_storage.json').read_text())
ENV = {'AWS_ACCESS_KEY_ID': 'fixture-env-key', 'AWS_SECRET_ACCESS_KEY': 'fixture-env-secret',
       'AWS_SESSION_TOKEN': 'fixture-env-token', 'AWS_REGION': 'eu-west-1', 'AWS_ALLOW_HTTP': 'true',
       'HERMES_ATIF_SECRET': 'fixture-explicit-secret', 'HERMES_ATIF_TOKEN': 'fixture-explicit-token',
       'NO_PROXY': '*'}


class Peer(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_POST(self):
        self.capture()

    def do_PUT(self):
        self.capture()

    def capture(self):
        body = self.rfile.read(int(self.headers['Content-Length']))
        self.server.records.append({'method': self.command, 'path': self.path,
            'headers': {name.lower(): value for name, value in self.headers.items()}, 'body': body.decode()})
        code = 400 if self.path.startswith('/failed') else 200
        attempts = self.server.attempts.get(self.path, 0)
        self.server.attempts[self.path] = attempts + 1
        if '/retry500-bucket/' in self.path and attempts < 3:
            code = 503
        if '/retry429-bucket/' in self.path and attempts < 2:
            code = 429
        self.send_response(code)
        self.send_header('Content-Length', '0')
        self.end_headers()


class NativeStorageTests(unittest.TestCase):
    def test_plugin_exports_completed_roots_and_flushes_open_roots_once_with_fatal_cleanup(self):
        from standalone import relay, relay_plugin
        with tempfile.TemporaryDirectory(prefix='hermes-atif-plugin-') as temporary:
            config = {'version': 1, 'components': [{'kind': 'observability', 'config': {
                'version': 4, 'atif': {'enabled': True, 'output_directory': temporary,
                                     'filename_template': '{metadata.team:-fallback}/{session_id}.json'}}}]}
            activation = asyncio.run(relay_plugin.initialize(config))
            def completed():
                handle = relay.scope.push('owned-atif-completed', relay.ScopeType.Agent, metadata={'team': 'owned/team'})
                relay.scope.pop(handle)
                return handle.uuid
            session = contextvars.Context().run(completed)
            relay.subscribers.flush()
            self.assertEqual(json.loads((Path(temporary) / 'owned-team' / (session + '.json')).read_text())['session_id'], session)
            opened = contextvars.Context().run(relay.scope.push, 'owned-atif-open', relay.ScopeType.Agent)
            relay.subscribers.flush()
            self.assertFalse((Path(temporary) / 'fallback' / (opened.uuid + '.json')).exists())
            asyncio.run(activation.close())
            self.assertFalse(activation.is_active)
            self.assertTrue((Path(temporary) / 'fallback' / (opened.uuid + '.json')).is_file())
            self.assertEqual(len(list(Path(temporary).rglob('*.json'))), 2)
            asyncio.run(activation.close())
            broken = asyncio.run(relay_plugin.initialize(config))
            contextvars.Context().run(relay.scope.push, 'owned-atif-bad-metadata', relay.ScopeType.Agent, metadata={'team': 17})
            relay.subscribers.flush()
            with self.assertRaises(RuntimeError):
                asyncio.run(broken.close())
            self.assertFalse(broken.is_active)
            with self.assertRaises(RuntimeError):
                asyncio.run(broken.close())
            fresh = asyncio.run(relay_plugin.initialize(config))
            asyncio.run(fresh.close())

    @unittest.skipUnless(os.name == 'posix', 'POSIX handle-relative filesystem boundary')
    def test_atomic_private_files_preserve_old_data_and_confine_existing_descendants(self):
        with tempfile.TemporaryDirectory(prefix='hermes-atif-files-') as temporary:
            root = Path(temporary) / 'root'
            root.mkdir()
            outside = Path(temporary) / 'outside'
            outside.mkdir()
            storage.atomic_private_write(root, 'new/child/file.json', b'first')
            target = root / 'new/child/file.json'
            self.assertEqual(target.read_bytes(), b'first')
            self.assertEqual(target.stat().st_mode & 0o777, 0o600)
            self.assertEqual(target.parent.stat().st_mode & 0o777, 0o700)
            os.link(target, outside / 'old-inode')
            storage.atomic_private_write(root, 'new/child/file.json', b'replacement')
            self.assertEqual((outside / 'old-inode').read_bytes(), b'first')
            self.assertEqual(target.read_bytes(), b'replacement')
            with patch.object(storage.os, 'replace', side_effect=OSError('injected rename failure')):
                with self.assertRaises(OSError):
                    storage.atomic_private_write(root, 'new/child/file.json', b'never-published')
            self.assertEqual(target.read_bytes(), b'replacement')
            self.assertFalse(list(target.parent.glob('.*.tmp')))
            (root / 'alias').symlink_to(outside, target_is_directory=True)
            with self.assertRaises(OSError):
                storage.atomic_private_write(root, 'alias/escape.json', b'escape')
            self.assertFalse((outside / 'escape.json').exists())
            protected = outside / 'protected'
            protected.write_bytes(b'unchanged')
            (root / 'leaf').symlink_to(protected)
            with self.assertRaises(OSError):
                storage.atomic_private_write(root, 'leaf', b'escape')
            self.assertEqual(protected.read_bytes(), b'unchanged')
            os.mkfifo(root / 'fifo')
            with self.assertRaises(OSError):
                storage.atomic_private_write(root, 'fifo', b'cannot-block')
            with self.assertRaises(ValueError):
                storage.atomic_private_write(root, '../outside/traversal', b'escape')
            # A configured root may itself be a trusted platform alias.
            configured = Path(temporary) / 'configured-alias'
            configured.symlink_to(root, target_is_directory=True)
            storage.atomic_private_write(configured, 'trusted/file', b'configured')
            self.assertEqual((root / 'trusted/file').read_bytes(), b'configured')

    def test_native_filename_validation_and_rendering(self):
        for case in NATIVE['validations']:
            with self.subTest(template=case['template']):
                try:
                    storage.validate_template(case['template'])
                    accepted = True
                except ValueError:
                    accepted = False
                self.assertEqual(accepted, case['accepted'])
        for case in NATIVE['renders']:
            with self.subTest(name=case['name'], template=case['template']):
                try:
                    rendered = storage.render_filename(case['template'], 'SESSION', case['metadata'])
                    paths = [rendered.replace('SESSION', '$SESSION')]
                except ValueError:
                    paths = []
                self.assertEqual(paths, case['files'])

    def test_native_sigv4_requests_including_credentials_and_escaped_keys(self):
        count = 0
        for case in NATIVE['storage_scenarios']:
            for record in case['records']:
                if record['method'] != 'PUT':
                    continue
                headers = {name.lower(): value for name, value in record['headers'].items()}
                authorization = headers['authorization']
                access_key = authorization.split('Credential=', 1)[1].split('/', 1)[0]
                secret = 'fixture-explicit-secret' if access_key == 'fixture-explicit-key' else 'fixture-env-secret'
                region = authorization.split('Credential=', 1)[1].split('/')[2]
                url = 'http://' + headers['host'] + record['path']
                actual = storage.sign_s3(url, record['body'].encode(), access_key, secret,
                    headers.get('x-amz-security-token'), region, headers['x-amz-date'])
                with self.subTest(case=case['name'], path=record['path']):
                    for name, value in actual.items():
                        self.assertEqual(value, headers[name], name)
                count += 1
        self.assertGreaterEqual(count, 12)

    def test_actual_owned_http_and_s3_replay_remote_only_fallback_and_rotation(self):
        server = ThreadingHTTPServer(('127.0.0.1', 0), Peer)
        server.records = []
        server.attempts = {}
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        origin = 'http://127.0.0.1:' + str(server.server_port)
        try:
            with tempfile.TemporaryDirectory(prefix='hermes-atif-candidate-') as temporary, patch.dict(os.environ, ENV, clear=True):
                os.environ['AWS_ENDPOINT_URL'] = origin
                os.environ['ATIF_OWNED_HEADER'] = 'env-one'
                token = Path(temporary) / 'header'
                token.write_text('file-one\n')
                for case in NATIVE['storage_scenarios']:
                    with self.subTest(case=case['name']):
                        target = Path(temporary) / case['name']
                        destinations = deepcopy(case['storage'])
                        for destination in destinations:
                            for key in ('endpoint', 'endpoint_url'):
                                if key in destination:
                                    destination[key] = destination[key].replace(NATIVE['origin'], origin)
                            if destination.get('header_file'):
                                destination['header_file'] = {'X-From-File': str(token)}
                        service = storage.AtifStorage({'storage': destinations, 'output_directory': str(target),
                            'filename_template': '{metadata.team}/{session_id}.json'})
                        start = len(server.records)
                        sessions = {}
                        for record in case['records']:
                            sessions.setdefault(record['payload']['session_id'], record)
                        for index, (session_id, record) in enumerate(sessions.items()):
                            if case['name'] == 'http-success':
                                os.environ['ATIF_OWNED_HEADER'] = 'env-mutated'
                                token.write_text('file-one\n' if index == 0 else 'file-two\n')
                            service.write(session_id, {'team': 'owned/team'}, record['body'].encode())
                        actual = server.records[start:]
                        self.assertEqual(len(actual), len(case['records']))
                        for observed, expected in zip(actual, case['records']):
                            self.assertEqual(observed['method'], expected['method'])
                            self.assertEqual(observed['path'], expected['path'])
                            self.assertEqual(observed['body'], expected['body'])
                            expected_headers = {name.lower(): value for name, value in expected['headers'].items()}
                            if observed['method'] == 'POST':
                                for name in ('content-type', 'x-nemo-relay-atif-filename', 'x-nemo-relay-atif-session-id',
                                             'x-literal', 'x-from-env', 'x-from-file'):
                                    self.assertEqual(observed['headers'].get(name), expected_headers.get(name), name)
                            else:
                                self.assertEqual(observed['headers']['x-amz-security-token'], expected_headers['x-amz-security-token'])
                                self.assertEqual(observed['headers']['authorization'].split(', SignedHeaders=')[0].split('/', 1)[0],
                                                 expected_headers['authorization'].split(', SignedHeaders=')[0].split('/', 1)[0])
                        self.assertEqual(len(list(target.rglob('*.json'))) if target.exists() else 0, case['local_files'])
                        if not case['local_files']:
                            self.assertFalse(target.exists(), 'Remote success must not touch the local destination')
                        for path in target.rglob('*.json'):
                            self.assertEqual(path.read_text(), sessions[path.stem]['body'])
                            if os.name == 'posix':
                                self.assertEqual(path.stat().st_mode & 0o777, 0o600)
                        diagnostics = service.runtime_diagnostics()
                        for field in ('code', 'component', 'count', 'field', 'session_id'):
                            self.assertEqual([row[field] for row in diagnostics], [row[field] for row in case['diagnostics']])
                        if case['close_error']:
                            with self.assertRaises(RuntimeError):
                                service.close()
                        else:
                            service.close()
        finally:
            server.shutdown(); server.server_close(); thread.join(3)


if __name__ == '__main__':
    unittest.main(verbosity=2)
