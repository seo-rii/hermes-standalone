"""ATIF filename and storage contracts from reviewed NeMo Relay 0.10.0.

Destinations own activation-time credentials. HTTP metadata headers refresh at
upload; remote successes suppress local output, and an all-remote failure
preserves the trajectory locally. Native failures remain visible at teardown.
"""
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import hmac
import os
from pathlib import Path
import random
import re
import stat
import threading
import time
from urllib import request as http
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlsplit, urlunsplit
import uuid


DEFAULT_TEMPLATE = 'nemo-relay-atif-{session_id}.json'
_PREFIX = '{metadata.'


def relative_parts(value):
    path = Path(value)
    # Rust retains a leading CurDir but normalizes interior CurDir components.
    leading = value.split(os.sep, 1)[0] == '.'
    if path.is_absolute() or path.drive or leading or not path.parts or '..' in path.parts:
        raise ValueError('ATIF filename must be a path-safe relative path')
    return path.parts


def _expression(value):
    selector, separator, fallback = value.partition(':-')
    if not re.fullmatch(r'[A-Za-z0-9_-]+(?:\.[A-Za-z0-9_-]+)*', selector):
        raise ValueError('Invalid ATIF metadata selector')
    return selector, fallback if separator else None


def _safe_metadata_path(value):
    return bool(value) and all(part not in {'', '.', '..'} and re.fullmatch(r'[A-Za-z0-9_.~-]+', part)
                               for part in value.split('/'))


def validate_template(template):
    if not isinstance(template, str) or '{session_id}' not in template:
        raise ValueError('ATIF filename_template must contain {session_id}')
    relative_parts(template.replace('{session_id}', 'session').replace(_PREFIX, 'metadata.'))
    cursor = 0
    while (start := template.find(_PREFIX, cursor)) >= 0:
        end = template.find('}', start + len(_PREFIX))
        if end < 0:
            raise ValueError('ATIF filename_template has an unclosed metadata placeholder')
        _, fallback = _expression(template[start + len(_PREFIX):end])
        if fallback is not None and not _safe_metadata_path(fallback):
            raise ValueError('ATIF metadata fallback must be a path-safe relative fragment')
        cursor = end + 1


def _sanitize(value):
    output, replacing, cursor = [], False, 0
    while cursor < len(value):
        character = value[cursor]
        cursor += 1
        if character == '.':
            start = cursor - 1
            while cursor < len(value) and value[cursor] == '.':
                cursor += 1
            safe = cursor - start == 1 and not (not output and cursor == len(value))
        else:
            safe = character.isascii() and (character.isalnum() or character in '-_~')
        if safe:
            output.append(character)
            replacing = False
        elif not replacing:
            output.append('-')
            replacing = True
    return ''.join(output)


def render_filename(template, session_id, metadata=None):
    validate_template(template)
    if not isinstance(session_id, str) or session_id in {'', '.', '..'} or not re.fullmatch(r'[A-Za-z0-9_.~-]+', session_id):
        raise ValueError('ATIF session_id must be a path-safe component')
    result, cursor = template.replace('{session_id}', session_id), 0
    while (start := result.find(_PREFIX, cursor)) >= 0:
        end = result.find('}', start + len(_PREFIX))
        selector, fallback = _expression(result[start + len(_PREFIX):end])
        value = metadata
        for segment in selector.split('.'):
            if value is None:
                break
            if not isinstance(value, dict):
                raise ValueError('ATIF metadata selector traversed a non-object value')
            value = value.get(segment)
        if value is None:
            value = fallback
        elif isinstance(value, str):
            value = _sanitize(value)
        else:
            raise ValueError('ATIF metadata selector resolved to a non-string value')
        if value is None or not _safe_metadata_path(value):
            raise ValueError('ATIF metadata must resolve to a path-safe string')
        result = result[:start] + value + result[end + 1:]
        cursor = start + len(value)
    relative_parts(result)
    return result


def _environment(variable):
    if not isinstance(variable, str) or not variable or variable.strip() != variable or '=' in variable or '\0' in variable:
        raise ValueError('ATIF secret field must name an environment variable')
    value = os.environ.get(variable)
    if not value:
        raise ValueError('ATIF configured environment variable is missing or empty')
    return value


def _http_options(config):
    from standalone.relay_plugin import _header_destination, _headers
    endpoint = config.get('endpoint')
    _header_destination(endpoint, config, signal='logs')
    if endpoint != endpoint.strip():
        raise ValueError('ATIF HTTP endpoint must be unpadded')
    timeout = config.get('timeout_millis', 3000)
    if type(timeout) is not int or not 0 < timeout < 2**64:
        raise ValueError('ATIF HTTP timeout_millis must be a positive integer')
    sources = [config.get(key) or {} for key in ('headers', 'header_env', 'header_file')]
    names = set()
    for source in sources:
        if not isinstance(source, dict):
            raise ValueError('ATIF HTTP header sources must be objects')
        for name in source:
            if not isinstance(name, str) or name.lower() in names:
                raise ValueError('ATIF HTTP headers must be unique across sources')
            names.add(name.lower())
    headers = dict(sources[0])
    headers.update({name: _environment(variable) for name, variable in sources[1].items()})
    files = {}
    for name, filename in sources[2].items():
        if not isinstance(filename, str) or not filename or not stat.S_ISREG(Path(filename).stat().st_mode):
            raise ValueError('ATIF header file must name a regular file')
        files[name] = str(Path(filename).absolute())
    _headers({'headers': {**headers, **{name: 'validated-file-name' for name in files}}})
    return {'endpoint': endpoint, 'headers': headers, 'header_file': files, 'timeout_millis': timeout}


def _s3_options(config):
    if not isinstance(config.get('bucket'), str) or not config['bucket'].strip():
        raise ValueError('ATIF S3 bucket must be nonempty')
    options = {'bucket': config['bucket']}
    for key, environment in (('access_key_id', 'AWS_ACCESS_KEY_ID'), ('region', 'AWS_REGION'),
                             ('endpoint_url', 'AWS_ENDPOINT_URL')):
        value = config.get(key)
        if value is None:
            value = os.environ.get(environment)
        if value is not None and not isinstance(value, str):
            raise ValueError('ATIF S3 connection fields must be strings')
        options[key] = value
    for key, environment in (('secret_access_key', 'AWS_SECRET_ACCESS_KEY'), ('session_token', 'AWS_SESSION_TOKEN')):
        variable = config.get(key + '_var')
        options[key] = _environment(variable) if variable is not None else os.environ.get(environment)
    allow_http = config.get('allow_http')
    if allow_http is None:
        raw = os.environ.get('AWS_ALLOW_HTTP', 'false').lower()
        if raw not in {'true', 'false', '1', '0'}:
            raise ValueError('Invalid AWS_ALLOW_HTTP value')
        allow_http = raw in {'true', '1'}
    if type(allow_http) is not bool:
        raise ValueError('ATIF allow_http must be a boolean')
    options['allow_http'] = allow_http
    if (options['access_key_id'] is None) != (options['secret_access_key'] is None):
        raise ValueError('ATIF static S3 credentials require both access key ID and secret')
    prefix = config.get('key_prefix') or ''
    if not isinstance(prefix, str):
        raise ValueError('ATIF S3 key_prefix must be a string')
    options['key_prefix'] = prefix.strip().rstrip('/') + '/' if prefix.strip().rstrip('/') else ''
    region = options['region'] or os.environ.get('AWS_DEFAULT_REGION') or 'us-east-1'
    endpoint = os.environ.get('AWS_ENDPOINT_URL_S3') or options['endpoint_url'] or 'https://s3.' + region + '.amazonaws.com'
    parsed = urlsplit(endpoint)
    if parsed.scheme not in {'http', 'https'} or not parsed.hostname or parsed.username is not None or parsed.password is not None:
        raise ValueError('ATIF S3 endpoint must be an HTTP(S) URL without user credentials')
    if parsed.scheme == 'http' and not allow_http:
        raise ValueError('ATIF S3 HTTP endpoint requires allow_http')
    parsed.port
    options['region'], options['endpoint_url'] = region, endpoint.rstrip('/')
    return options


def validate_config(config):
    validate_template(config.get('filename_template', DEFAULT_TEMPLATE))
    directory = config.get('output_directory')
    if directory is not None and (not isinstance(directory, str) or '\0' in directory or '..' in Path(directory).parts):
        raise ValueError('ATIF output_directory contains an invalid path')
    storage = config.get('storage', [])
    if not isinstance(storage, list):
        raise ValueError('ATIF storage must be an array')
    for destination in storage:
        if not isinstance(destination, dict) or destination.get('type') not in {'http', 's3'}:
            raise ValueError('ATIF storage must name an HTTP or S3 destination')
        if destination['type'] == 'http':
            _http_options(destination)
        else:
            _s3_options(destination)


def sign_s3(url, payload, access_key, secret, token, region, timestamp=None):
    """Return SigV4 headers; actual pinned native requests provide golden tests."""
    parsed = urlsplit(url)
    timestamp = timestamp or datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    digest = hashlib.sha256(payload).hexdigest()
    headers = {'host': parsed.netloc, 'x-amz-content-sha256': digest, 'x-amz-date': timestamp}
    if token is not None:
        headers['x-amz-security-token'] = token
    names = ';'.join(sorted(headers))
    canonical_headers = ''.join(name + ':' + ' '.join(headers[name].split()) + '\n' for name in sorted(headers))
    canonical = '\n'.join(('PUT', parsed.path or '/', parsed.query, canonical_headers, names, digest))
    scope = timestamp[:8] + '/' + region + '/s3/aws4_request'
    string_to_sign = 'AWS4-HMAC-SHA256\n' + timestamp + '\n' + scope + '\n' + hashlib.sha256(canonical.encode()).hexdigest()
    key = ('AWS4' + secret).encode()
    for part in (timestamp[:8], region, 's3', 'aws4_request'):
        key = hmac.digest(key, part.encode(), 'sha256')
    signature = hmac.new(key, string_to_sign.encode(), hashlib.sha256).hexdigest()
    headers['authorization'] = 'AWS4-HMAC-SHA256 Credential=' + access_key + '/' + scope + ', SignedHeaders=' + names + ', Signature=' + signature
    return headers


class _S3Destination:
    def __init__(self, config):
        self.options = _s3_options(config)
        self._session = None

    def _credentials(self):
        options = self.options
        if options['access_key_id'] is not None and options['secret_access_key'] is not None:
            return options['access_key_id'], options['secret_access_key'], options['session_token']
        # The embedded original SDK owns refreshable profile, web-identity,
        # container and instance credential discovery; no subprocess or install.
        if self._session is None:
            from botocore.session import get_session
            self._session = get_session()
        credentials = self._session.get_credentials()
        if credentials is None:
            raise OSError('ATIF S3 credentials are unavailable')
        frozen = credentials.get_frozen_credentials()
        return frozen.access_key, frozen.secret_key, frozen.token

    def put(self, filename, session_id, payload):
        from standalone.relay_plugin import _NoRedirect
        options = self.options
        # object_store::path::Path strips outer slashes and normalizes empty
        # separators before URL encoding; literal non-ASCII keys are UTF-8.
        # PathPart::from first escapes its INVALID ASCII set and all UTF-8
        # bytes. The S3 strict URL encoder then escapes the stored percent
        # signs a second time; spaces are escaped only by that second pass.
        safe = ''.join(chr(byte) for byte in range(32, 127) if chr(byte) not in '/\\{^}%`]">[~<#|*?')
        encoded_parts = []
        for part in (options['key_prefix'] + filename).split('/'):
            if part:
                encoded_parts.append('%2E' * len(part) if part in {'.', '..'} else ''.join(
                    chr(byte) if chr(byte) in safe else '%' + format(byte, '02X') for byte in part.encode()))
        key = '/'.join(encoded_parts)
        path = quote(options['bucket'], safe='') + '/' + quote(key, safe='/~')
        url = options['endpoint_url'] + '/' + path
        access_key, secret, token = self._credentials()
        request = http.Request(url, data=payload, headers=sign_s3(url, payload, access_key, secret, token, options['region']), method='PUT')
        opener = http.build_opener(_NoRedirect())
        start, backoff = time.monotonic(), .1
        rng = random.SystemRandom()
        for attempt in range(11):
            failure = 'ATIF S3 transport failed'
            try:
                with opener.open(request, timeout=30) as response:
                    response.read(65536)
                    return
            except HTTPError as error:
                code = error.code
                error.close()
                failure = 'ATIF S3 upload returned HTTP ' + str(code)
                if not (500 <= code < 600 or code in {408, 429}):
                    raise OSError(failure) from None
            except (URLError, TimeoutError, ConnectionError):
                pass
            if attempt == 10 or time.monotonic() - start > 180:
                raise OSError(failure) from None
            # Native Backoff returns the old interval, then stores a random
            # interval in [100ms, 2*old), capped at 15s. Signed credentials,
            # timestamp and payload remain identical over all retries.
            delay, backoff = backoff, min(15., rng.uniform(.1, backoff * 2))
            time.sleep(delay)


class _HttpDestination:
    def __init__(self, config):
        self.options = _http_options(config)

    def put(self, filename, session_id, payload):
        from standalone.relay_plugin import _headers, _NoRedirect
        options = self.options
        headers = _headers(options)
        # Native system headers override all configured sources after rotation.
        reserved = {'content-type', 'x-nemo-relay-atif-filename', 'x-nemo-relay-atif-session-id'}
        headers = {name: value for name, value in headers.items() if name.lower() not in reserved}
        headers.update({'Content-Type': 'application/json', 'X-Nemo-Relay-Atif-Filename': filename,
                        'X-Nemo-Relay-Atif-Session-Id': session_id})
        request = http.Request(options['endpoint'], data=payload, headers=headers, method='POST')
        opener = http.build_opener(_NoRedirect()) if options['headers'] or options['header_file'] else http.build_opener()
        try:
            with opener.open(request, timeout=options['timeout_millis'] / 1000) as response:
                if not 200 <= response.status < 300:
                    raise OSError('ATIF HTTP upload returned a non-success status')
                response.read(65536)
        except HTTPError as error:
            code = error.code
            error.close()
            raise OSError('ATIF HTTP upload returned HTTP ' + str(code)) from None
        except URLError:
            raise OSError('ATIF HTTP transport failed') from None


def _reject_target(parent, filename):
    try:
        metadata = os.stat(filename, dir_fd=parent, follow_symlinks=False)
    except FileNotFoundError:
        return
    if not stat.S_ISREG(metadata.st_mode):
        raise OSError('ATIF output target must be a regular file without symlinks')


def atomic_private_write(directory, filename, payload):
    from standalone.relay_files import _anchor
    parts = relative_parts(filename)
    anchor, missing = _anchor(directory)
    if os.name == 'nt':
        from standalone.relay_atif_windows import atomic_write
        return atomic_write(anchor, missing + list(parts[:-1]), parts[-1], payload)
    flags = os.O_RDONLY | os.O_DIRECTORY | getattr(os, 'O_CLOEXEC', 0)
    parent = os.open(anchor, flags)
    temporary = None
    try:
        for component in missing + list(parts[:-1]):
            try:
                os.mkdir(component, mode=0o700, dir_fd=parent)
            except FileExistsError:
                pass
            child = os.open(component, flags | os.O_NOFOLLOW, dir_fd=parent)
            os.close(parent)
            parent = child
        _reject_target(parent, parts[-1])
        for _ in range(16):
            name = '.' + parts[-1] + '.' + uuid.uuid4().hex + '.tmp'
            try:
                descriptor = os.open(name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | getattr(os, 'O_CLOEXEC', 0), 0o600, dir_fd=parent)
            except FileExistsError:
                continue
            temporary = name
            with os.fdopen(descriptor, 'wb') as output:
                output.write(payload)
                output.flush()
                os.fsync(output.fileno())
            _reject_target(parent, parts[-1])
            os.replace(name, parts[-1], src_dir_fd=parent, dst_dir_fd=parent)
            temporary = None
            return
        raise OSError('ATIF could not allocate a private temporary file')
    finally:
        if temporary is not None:
            os.unlink(temporary, dir_fd=parent)
        os.close(parent)


class AtifStorage:
    def __init__(self, config):
        validate_config(config)
        self.config = deepcopy(config)
        self.destinations = [(_HttpDestination if item['type'] == 'http' else _S3Destination)(item)
                             for item in config.get('storage', [])]
        self._diagnostics, self._lock = {}, threading.RLock()

    def _failure(self, code, field, session_id, message):
        key = (code, field)
        previous = self._diagnostics.get(key)
        self._diagnostics[key] = {'code': code, 'component': 'observability', 'field': field,
            'message': message, 'session_id': session_id, 'count': previous['count'] + 1 if previous else 1}

    def write(self, session_id, metadata, payload):
        filename = render_filename(self.config.get('filename_template', DEFAULT_TEMPLATE), session_id, metadata)
        with self._lock:
            successful = False
            for index, destination in enumerate(self.destinations):
                try:
                    destination.put(filename, session_id, payload)
                    successful = True
                except (OSError, ValueError):
                    self._failure('atif.remote_delivery_failed', 'storage[' + str(index) + ']', session_id,
                                  'ATIF remote trajectory upload failed')
            if not self.destinations or not successful:
                try:
                    atomic_private_write(self.config.get('output_directory') or '.', filename, payload)
                except OSError:
                    code = 'atif.local_fallback_failed' if self.destinations else 'atif.local_write_failed'
                    self._failure(code, 'output_directory', session_id, 'ATIF private trajectory write failed')

    def runtime_diagnostics(self):
        with self._lock:
            return deepcopy(list(self._diagnostics.values()))

    def close(self):
        with self._lock:
            if self._diagnostics:
                from standalone.relay_plugin import DeliveryFailure
                raise DeliveryFailure('ATIF runtime delivery failures: ' + ', '.join(
                    row['code'] + ' (' + str(row['count']) + ')' for row in self._diagnostics.values()))
