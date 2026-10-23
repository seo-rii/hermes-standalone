"""Owned ATOF WebSocket and long-lived HTTP NDJSON workers.

Uses the already embedded pure Python WebSocket/HTTP clients. A flush is the
original transport handoff barrier; it is not a remote application receipt.
Pinned native/source contracts are recorded in relay-atof-stream-provenance.json.
"""

import asyncio
from collections import deque
from copy import deepcopy
import logging
import os
from pathlib import Path
import stat
import threading
from urllib.parse import urlsplit, urlunsplit


logger = logging.getLogger(__name__)


def _activation_headers(config):
    from standalone.relay_plugin import _headers
    literal, environment, files = (config.get(name) or {} for name in ('headers','header_env','header_file'))
    if any(not isinstance(value,dict) for value in (literal,environment,files)):
        raise ValueError('ATOF header sources must be objects')
    names=set()
    for source in (literal,environment,files):
        for name in source:
            if not isinstance(name,str) or name.lower() in names:
                raise ValueError('ATOF header sources must have unique header names')
            names.add(name.lower())
    result=dict(literal)
    for name,variable in environment.items():
        if not isinstance(variable,str) or variable not in os.environ or not os.environ[variable].strip():
            raise ValueError('ATOF header environment variable is missing or blank')
        result[name]=os.environ[variable]
    for name,filename in files.items():
        if not isinstance(filename,str) or not filename or not stat.S_ISREG(Path(filename).stat().st_mode):
            raise ValueError('ATOF header file must name a regular file')
    # Validate file header names without reading their connection-time values.
    _headers({'headers':{**result,**{name:'validated-file-name' for name in files}}})
    return _headers({'headers':result})


def validate_stream(config):
    from standalone.relay_plugin import _header_destination
    transport = config.get('transport', 'http_post')
    if transport not in ('websocket', 'ndjson'):
        raise ValueError('Unknown ATOF streaming transport')
    timeout = config.get('timeout_millis', 3000)
    if type(timeout) is not int or not 0 < timeout < 2**64:
        raise ValueError('ATOF timeout_millis must be a positive integer')
    value = config.get('url')
    if not isinstance(value, str):
        raise ValueError('ATOF streaming endpoint must be a URL')
    try:
        endpoint = urlsplit(value)
        valid = endpoint.scheme in (('ws','wss') if transport == 'websocket' else ('http','https'))
        if not valid or not endpoint.hostname or endpoint.username is not None or endpoint.password is not None:
            raise ValueError('Invalid ATOF streaming endpoint')
        endpoint.port
    except ValueError:
        raise ValueError('Invalid ATOF streaming endpoint') from None
    if config.get('field_name_policy', 'preserve') not in ('preserve', 'replace_dots'):
        raise ValueError('Invalid ATOF field-name policy')
    projected = urlunsplit(({'ws':'http','wss':'https'}.get(endpoint.scheme, endpoint.scheme),
                           endpoint.netloc,endpoint.path,endpoint.query,endpoint.fragment))
    _header_destination(projected, config, signal='logs')
    try:
        _activation_headers(config)
    except OSError:
        raise ValueError('ATOF header file is unavailable') from None


class AtofStreamingSink:
    def __init__(self, config):
        validate_stream(config)
        self.config = deepcopy(config)
        self.timeout = self.config.get('timeout_millis', 3000)/1000
        # ATOF freezes literal/environment headers at activation. Files are
        # re-read for every WebSocket connection, or once for NDJSON startup.
        self._headers = _activation_headers(config)
        if self.config.get('header_file'):
            self.config['header_file']={name:str(Path(filename).absolute()) for name,filename in self.config['header_file'].items()}
        self._pending, self._diagnostics = deque(), {}
        self._state_lock, self._send_lock = threading.RLock(), threading.Lock()
        self._closed, self._socket = False, None
        self._ready = threading.Event()
        self._thread = threading.Thread(target=self._start, name='relay-atof-stream', daemon=True)
        self._thread.start()
        self._ready.wait()

    def _record(self, code):
        with self._state_lock:
            self._diagnostics[code] = min(2**64-1, self._diagnostics.get(code, 0)+1)
        logger.warning('ATOF streaming endpoint %s', code)

    def runtime_diagnostics(self):
        with self._state_lock:
            return [{'code':'atof.'+code, 'component':'observability', 'count':count,
                     'message':'ATOF '+self.config['transport']+' '+code}
                    for code,count in sorted(self._diagnostics.items())]

    def _start(self):
        asyncio.run(self._run())

    def _connection_headers(self):
        from standalone.relay_plugin import _headers
        return _headers({'headers':self._headers, 'header_file':self.config.get('header_file')})

    async def _connect_websocket(self):
        from websockets.asyncio.client import connect
        return await connect(self.config['url'], additional_headers=self._connection_headers(),
                             proxy=None, open_timeout=self.timeout, close_timeout=self.timeout,
                             ping_interval=None, compression=None)

    async def _drain_websocket(self):
        async def drain():
            while self._pending:
                try:
                    if self._socket is None:
                        self._socket = await self._connect_websocket()
                    await self._socket.send(self._pending[0].decode('utf-8'))
                    self._pending.popleft()
                except (OSError, ValueError, RuntimeError):
                    if self._socket is not None:
                        try:
                            await self._socket.close()
                        except Exception:
                            pass
                    self._socket = None
                    self._record('endpoint_reconnecting')
                    await asyncio.sleep(.05)
                except Exception:
                    # The pure WebSocket SDK's handshake/closed errors do
                    # not share OSError. Never include server text or headers.
                    self._socket = None
                    self._record('endpoint_reconnecting')
                    await asyncio.sleep(.05)
        try:
            await asyncio.wait_for(drain(), self.timeout)
        except TimeoutError:
            self._record('endpoint_flush_failed')

    async def _run(self):
        self._loop, self._queue = asyncio.get_running_loop(), asyncio.Queue()
        self._ready.set()
        session, upload, body_queue = None, None, None
        try:
            if self.config['transport'] == 'websocket':
                try:
                    self._socket = await self._connect_websocket()
                except Exception:
                    self._record('endpoint_reconnecting')
            else:
                import aiohttp
                body_queue = asyncio.Queue()
                async def body():
                    while True:
                        value, barrier = await body_queue.get()
                        if barrier is not None:
                            barrier.set()
                        elif value is None:
                            return
                        else:
                            yield value+b'\n'
                timeout = aiohttp.ClientTimeout(total=None, sock_connect=self.timeout)
                session = aiohttp.ClientSession(timeout=timeout, trust_env=False)
                async def request():
                    try:
                        headers=self._connection_headers()
                        headers['Content-Type']='application/x-ndjson'
                        async with session.post(self.config['url'],data=body(),headers=headers,
                                                allow_redirects=not bool(self._headers or self.config.get('header_file'))) as response:
                            if not 200 <= response.status < 300:
                                self._record('endpoint_delivery_failed')
                    except Exception:
                        self._record('endpoint_delivery_failed')
                    finally:
                        while not body_queue.empty():
                            _,barrier=body_queue.get_nowait()
                            if barrier is not None:
                                barrier.set()
                upload=asyncio.create_task(request())
            while True:
                command,payload,barrier = await self._queue.get()
                try:
                    if command == 'event':
                        if body_queue is None:
                            self._pending.append(payload)
                            await self._drain_websocket()
                        elif upload.done():
                            self._record('endpoint_delivery_failed')
                        else:
                            body_queue.put_nowait((payload,None))
                    elif command == 'flush':
                        if body_queue is None:
                            await self._drain_websocket()
                        elif not upload.done():
                            handoff=asyncio.Event()
                            body_queue.put_nowait((None,handoff))
                            waiting=asyncio.create_task(handoff.wait())
                            try:
                                done,_=await asyncio.wait((waiting,upload),timeout=self.timeout,
                                                          return_when=asyncio.FIRST_COMPLETED)
                                if not done:
                                    self._record('endpoint_flush_failed')
                            finally:
                                waiting.cancel()
                                await asyncio.gather(waiting,return_exceptions=True)
                    elif command == 'close':
                        if body_queue is None:
                            await self._drain_websocket()
                        else:
                            body_queue.put_nowait((None,None))
                            try:
                                await asyncio.wait_for(upload,self.timeout)
                            except TimeoutError:
                                self._record('endpoint_close_failed')
                        return
                except Exception:
                    self._record('endpoint_delivery_failed')
                finally:
                    if barrier is not None:
                        barrier.set()
        except BaseException:
            self._record('endpoint_failed')
            raise
        finally:
            if self._socket is not None:
                try:
                    await self._socket.close()
                except Exception:
                    pass
            if upload is not None and not upload.done():
                upload.cancel()
                await asyncio.gather(upload,return_exceptions=True)
            if session is not None:
                await session.close()
            self._ready.set()

    def _submit(self, command, payload=None, *, wait=False):
        with self._send_lock:
            if self._closed:
                return
            if not self._thread.is_alive():
                raise RuntimeError('ATOF streaming worker exited')
            barrier=threading.Event() if wait else None
            self._loop.call_soon_threadsafe(self._queue.put_nowait,(command,payload,barrier))
        if barrier is not None and not barrier.wait(self.timeout):
            self._record('endpoint_'+('close' if command=='close' else 'flush')+'_failed')

    def write(self, payload):
        self._submit('event',payload)

    def flush(self):
        self._submit('flush',wait=True)

    def close(self):
        self._submit('close',wait=True)
        with self._send_lock:
            self._closed=True
        self._thread.join(self.timeout+1)
        if self._thread.is_alive():
            self._record('endpoint_close_failed')
