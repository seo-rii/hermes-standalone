"""Owned RFC 6455/HTTP1 peer shared by ATOF fixture authoring and replay tests."""
import base64
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import socketserver
import ssl
import struct
import threading


class _Server(ThreadingHTTPServer):
    daemon_threads=True
    def server_bind(self):
        socketserver.TCPServer.server_bind(self)
        self.server_name,self.server_port='127.0.0.1',self.server_address[1]


class _Handler(BaseHTTPRequestHandler):
    protocol_version='HTTP/1.1'
    def log_message(self,*args):
        pass

    def _exact(self,count):
        value=self.rfile.read(count)
        if len(value)!=count:
            raise EOFError
        return value

    def _record(self,payload):
        self.server.owner.records.append(json.loads(payload))
        self.server.owner.changed.set()

    def do_GET(self):
        owner=self.server.owner
        owner.headers.append(dict(self.headers))
        owner.paths.append(self.path)
        if owner.refusals:
            owner.refusals-=1
            if owner.on_refusal is not None:
                owner.on_refusal(owner.refusals)
            self.send_response(503);self.send_header('Content-Length','0');self.end_headers()
            self.close_connection=True
            return
        key=self.headers['Sec-WebSocket-Key']
        accept=base64.b64encode(hashlib.sha1((key+'258EAFA5-E914-47DA-95CA-C5AB0DC85B11').encode()).digest()).decode()
        self.send_response(101);self.send_header('Upgrade','websocket');self.send_header('Connection','Upgrade')
        self.send_header('Sec-WebSocket-Accept',accept);self.end_headers();self.wfile.flush()
        owner.connected.set()
        fragment=bytearray()
        try:
            while True:
                first,second=self._exact(2)
                length=second&127
                if length==126:
                    length=struct.unpack('!H',self._exact(2))[0]
                elif length==127:
                    length=struct.unpack('!Q',self._exact(8))[0]
                if length>4*1024**2:
                    raise ValueError('Owned WebSocket frame bound')
                mask=self._exact(4) if second&128 else None
                payload=self._exact(length)
                if mask:
                    payload=bytes(value^mask[index&3] for index,value in enumerate(payload))
                opcode=first&15
                if opcode==8:
                    self.wfile.write(bytes([0x88,len(payload)])+payload);self.wfile.flush()
                    return
                if opcode==9:
                    self.wfile.write(bytes([0x8a,len(payload)])+payload);self.wfile.flush()
                elif opcode in (0,1):
                    fragment.extend(payload)
                    if first&128:
                        self._record(bytes(fragment));fragment.clear()
                else:
                    raise ValueError('Owned peer expected JSON text frames')
        except (EOFError,ConnectionError,OSError):
            pass
        finally:
            self.close_connection=True

    def do_POST(self):
        owner=self.server.owner
        owner.headers.append(dict(self.headers));owner.paths.append(self.path)
        owner.connected.set()
        pending=bytearray()
        try:
            if self.headers.get('Transfer-Encoding','').lower()=='chunked':
                while True:
                    header=self.rfile.readline()
                    if not header:
                        raise EOFError
                    length=int(header.split(b';',1)[0].strip(),16)
                    if not length:
                        while self.rfile.readline().strip():
                            pass
                        break
                    if length>4*1024**2:
                        raise ValueError('Owned HTTP chunk bound')
                    pending.extend(self._exact(length));assert self._exact(2)==b'\r\n'
                    while b'\n' in pending:
                        line,_,tail=pending.partition(b'\n');pending[:]=tail
                        if line:
                            self._record(line)
            else:
                pending.extend(self._exact(int(self.headers['Content-Length'])))
            if pending.strip():
                self._record(pending)
            self.send_response(owner.status);self.send_header('Content-Length','0');self.end_headers();self.wfile.flush()
        except (EOFError,ConnectionError,OSError):
            pass
        finally:
            owner.finished.set()
            self.close_connection=True


class AtofPeer:
    def __init__(self,*,certificate=None,key=None,refusals=0,status=200):
        self.records,self.headers,self.paths=[],[],[]
        self.changed,self.connected,self.finished=threading.Event(),threading.Event(),threading.Event()
        self.refusals,self.on_refusal,self.status=refusals,None,status
        self.server=_Server(('127.0.0.1',0),_Handler);self.server.owner=self
        self.tls=certificate is not None
        if self.tls:
            context=ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER);context.load_cert_chain(certificate,key)
            self.server.socket=context.wrap_socket(self.server.socket,server_side=True)
        self.port=self.server.server_port
        self.thread=threading.Thread(target=self.server.serve_forever,daemon=True)
        self.thread.start()

    def url(self,transport):
        scheme=('wss' if self.tls else 'ws') if transport=='websocket' else ('https' if self.tls else 'http')
        return scheme+'://127.0.0.1:'+str(self.port)+'/owned/stream?fixture=atof'

    def close(self):
        self.server.shutdown();self.server.server_close();self.thread.join(3)

    def __enter__(self):
        return self

    def __exit__(self,*exception):
        self.close()
