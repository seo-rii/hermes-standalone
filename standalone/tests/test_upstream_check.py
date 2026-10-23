"""The canonical-runner wrapper may only contact owned loopback peers."""

import subprocess
import sys
import unittest

from standalone.upstream_check import _NETWORK_GUARD


class UpstreamIsolationTests(unittest.TestCase):
    def test_loopback_works_and_external_audit_events_are_refused_before_any_io(self):
        program = _NETWORK_GUARD + r'''
import socket,threading
for event,args in (
 ('socket.getaddrinfo',('provider.example',443)),
 ('socket.getaddrinfo',('192.0.2.19',443)),
 ('socket.connect',(None,('198.51.100.9',443))),
 ('socket.connect',(None,('2001:db8::1',443,0,0))),
):
 try:sys.audit(event,*args)
 except PermissionError:pass
 else:raise AssertionError(('External event was allowed',event,args))
for host in ('127.0.0.1','::1','::ffff:127.0.0.1','localhost','localhost.'):
 sys.audit('socket.getaddrinfo',host,443)
 sys.audit('socket.connect',None,(host,443))
server=socket.socket();server.bind(('127.0.0.1',0));server.listen(1);server.settimeout(2)
errors=[]
def reply():
 try:
  connection,_=server.accept()
  with connection:
   assert connection.recv(4)==b'ping';connection.sendall(b'pong')
 except BaseException as error:errors.append(repr(error))
thread=threading.Thread(target=reply);thread.start()
try:
 with socket.create_connection(server.getsockname(),timeout=2) as client:
  client.sendall(b'ping');assert client.recv(4)==b'pong'
finally:
 thread.join(3);server.close()
assert not thread.is_alive() and not errors,errors
print('ok')
'''
        result = subprocess.run([sys.executable, "-I", "-S", "-c", program],
                                capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), "ok")
