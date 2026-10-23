"""Real subprocess ownership and signal behavior, without psutil or credentials."""

import os
from pathlib import Path
import signal
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import unittest

from standalone import processes


class ProcessTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="hermes-process-tests-")
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.children = []
        self.addCleanup(self.stop_children)

    def child(self, script="import time; time.sleep(60)"):
        environment = {"HOME": str(self.directory), "USERPROFILE": str(self.directory)}
        for name in ("SYSTEMROOT", "WINDIR"):
            if name in os.environ:
                environment[name] = os.environ[name]
        child = subprocess.Popen(
            [sys.executable, "-I", "-S", "-B", "-c", 'import sys; print("ready", flush=True); ' + script], env=environment,
            cwd=self.directory, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        )
        self.children.append(child)
        self.assertEqual(child.stdout.readline(), "ready\n")
        return child

    def stop_children(self):
        for child in self.children:
            if child.poll() is None:
                child.kill()
            child.communicate(timeout=5)

    def test_real_birth_parent_user_and_owned_child_enumeration(self):
        child = self.child()
        handle = processes.Process(child.pid)
        self.assertTrue(handle.is_running())
        self.assertEqual(handle.ppid(), os.getpid())
        self.assertEqual(handle.parent().pid, os.getpid())
        self.assertEqual(handle.username(), processes.Process().username())
        self.assertEqual(handle.create_time(), processes.Process(child.pid).create_time())
        self.assertIn(child.pid, [item.pid for item in processes.Process().children()])
        self.assertIn(os.getpid(), [item.pid for item in handle.parents()])
        self.assertLess(handle.create_time(), time.time() + 1)

    def test_wrong_birth_never_signals_the_live_child(self):
        child = self.child()
        handle = processes.Process(child.pid)
        handle._identity = ("intentionally-stale-birth",)
        with self.assertRaises(processes.NoSuchProcess):
            handle.kill()
        self.assertIsNone(child.poll())
        self.assertFalse(handle.is_running())
        self.assertIsNone(handle.wait(timeout=0))
        self.assertIsNone(child.poll())

    def test_system_statistics_report_real_memory_disk_and_cpu_samples(self):
        memory = processes.virtual_memory()
        self.assertGreater(memory.total, 0)
        self.assertGreaterEqual(memory.available, 0)
        self.assertLessEqual(memory.available, memory.total)
        self.assertGreaterEqual(memory.used, 0)
        self.assertEqual(memory.percent, round(100 * (memory.total - memory.available) / memory.total, 1))
        disk = processes.disk_usage(self.directory)
        self.assertEqual(disk.total, shutil.disk_usage(self.directory).total)
        self.assertGreaterEqual(disk.free, 0)
        self.assertEqual(disk.percent, round(100 * disk.used / (disk.used + disk.free), 1))
        started = time.monotonic()
        utilization = processes.cpu_percent(interval=0.02)
        self.assertGreaterEqual(time.monotonic() - started, 0.02)
        self.assertGreaterEqual(utilization, 0)
        self.assertLessEqual(utilization, 100)
        self.assertGreater(processes.Process().memory_info().rss, 0)
        with self.assertRaises(ValueError):
            processes.cpu_percent(interval=-1)
        process = processes.Process()
        self.assertGreaterEqual(sum(process.cpu_times()), 0)
        self.assertEqual(process.cpu_percent(), 0.0)
        self.assertGreaterEqual(process.cpu_percent(interval=0.01), 0)
        if os.name != "nt":
            self.assertGreaterEqual(process.num_fds(), 3)

    def test_wait_timeout_then_real_termination(self):
        child = self.child()
        handle = processes.Process(child.pid)
        with self.assertRaises(processes.TimeoutExpired):
            handle.wait(timeout=0.02)
        handle.terminate()
        handle.wait(timeout=5)
        self.assertFalse(handle.is_running())
        self.assertFalse(processes.pid_exists(child.pid))

    def test_process_cpu_sample_observes_real_child_work(self):
        child = self.child("import time\nvalue = 0\nwhile True:\n value = (value + 1) % 1000003")
        process = processes.Process(child.pid)
        before = sum(process.cpu_times())
        self.assertGreater(process.cpu_percent(interval=0.15), 0)
        self.assertGreater(sum(process.cpu_times()), before)

    @unittest.skipIf(os.name == "nt", "POSIX signal-state behavior")
    def test_suspend_resume_and_wait_procs(self):
        child = self.child()
        handle = processes.Process(child.pid)
        handle.suspend()
        deadline = time.monotonic() + 3
        while handle.status() != processes.STATUS_STOPPED and time.monotonic() < deadline:
            time.sleep(0.01)
        self.assertEqual(handle.status(), processes.STATUS_STOPPED)
        handle.resume()
        handle.kill()
        gone, alive = processes.wait_procs([handle], timeout=5)
        self.assertEqual(gone, [handle])
        self.assertEqual(alive, [])

    @unittest.skipUnless(sys.platform.startswith("linux"), "Linux procfs data")
    def test_child_command_environment_cwd_and_memory_are_kernel_data(self):
        child = self.child()
        handle = processes.Process(child.pid)
        self.assertEqual(handle.cmdline()[:4], [sys.executable, "-I", "-S", "-B"])
        self.assertEqual(handle.environ()["HOME"], str(self.directory))
        self.assertEqual(handle.cwd(), str(self.directory))
        self.assertGreater(handle.memory_info().vms, 0)
        self.assertGreaterEqual(handle.num_threads(), 1)

    @unittest.skipUnless(sys.platform.startswith("linux"), "Linux kernel socket-inode correlation")
    def test_network_descriptors_identify_only_owned_tcp_and_udp_sockets(self):
        with socket.socket() as listener, socket.socket(type=socket.SOCK_DGRAM) as udp:
            listener.bind(("127.0.0.1", 0))
            listener.listen()
            udp.bind(("127.0.0.1", 0))
            with socket.create_connection(listener.getsockname()) as client:
                peer, _ = listener.accept()
                with peer:
                    rows = processes.Process().net_connections(kind="inet4")
                    matching = {row.fd: row for row in rows}
                    self.assertEqual(matching[listener.fileno()].status, "LISTEN")
                    self.assertEqual(matching[listener.fileno()].laddr, listener.getsockname())
                    self.assertEqual(matching[client.fileno()].status, "ESTABLISHED")
                    self.assertEqual(matching[client.fileno()].raddr, listener.getsockname())
                    self.assertEqual(matching[udp.fileno()].status, "NONE")
                    self.assertEqual(matching[udp.fileno()].laddr, udp.getsockname())
                    self.assertEqual(matching[udp.fileno()].raddr, ())
                    self.assertTrue(all(row.family == socket.AF_INET for row in rows))

    @unittest.skipUnless(hasattr(os, "fork"), "POSIX zombie liveness contract")
    def test_zombie_pid_exists_but_status_identifies_exit(self):
        ready_read, ready_write = os.pipe()
        child = os.fork()
        if child == 0:
            os.close(ready_read)
            os.write(ready_write, b"ready")
            os._exit(0)
        os.close(ready_write)
        try:
            os.read(ready_read, 5)
            handle = processes.Process(child)
            deadline = time.monotonic() + 3
            while handle.status() != processes.STATUS_ZOMBIE and time.monotonic() < deadline:
                time.sleep(0.01)
            self.assertEqual(handle.status(), processes.STATUS_ZOMBIE)
            self.assertTrue(processes.pid_exists(child))
            self.assertTrue(handle.is_running())  # Same distinction as upstream psutil.
        finally:
            os.close(ready_read)
            os.waitpid(child, 0)


if __name__ == "__main__":
    unittest.main()
