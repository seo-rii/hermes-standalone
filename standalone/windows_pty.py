"""Qualified Windows ConPTY backend using only documented operating-system APIs.

CreatePseudoConsole + STARTUPINFOEX attach the actual child to a pseudoconsole.
A suspended launch enters an unnamed kill-on-close Job Object before it runs,
so cancellation owns descendants through a kernel handle, not a recycled PID.
The output pipe has its own draining thread; UTF-8 remains a byte stream.

API contract: https://learn.microsoft.com/windows/console/creating-a-pseudoconsole-session
"""

from __future__ import annotations

from collections import deque
import ctypes as ct
from functools import lru_cache
import os
import subprocess
import threading


DWORD, BOOL, WORD = ct.c_uint32, ct.c_int32, ct.c_uint16
HANDLE, SIZE_T = ct.c_void_p, ct.c_size_t


class COORD(ct.Structure):
    _fields_ = [("X", ct.c_int16), ("Y", ct.c_int16)]


class STARTUPINFOW(ct.Structure):
    _fields_ = [("cb", DWORD), ("lpReserved", ct.c_wchar_p), ("lpDesktop", ct.c_wchar_p),
                ("lpTitle", ct.c_wchar_p), ("dwX", DWORD), ("dwY", DWORD), ("dwXSize", DWORD),
                ("dwYSize", DWORD), ("dwXCountChars", DWORD), ("dwYCountChars", DWORD),
                ("dwFillAttribute", DWORD), ("dwFlags", DWORD), ("wShowWindow", WORD),
                ("cbReserved2", WORD), ("lpReserved2", HANDLE), ("hStdInput", HANDLE),
                ("hStdOutput", HANDLE), ("hStdError", HANDLE)]


class STARTUPINFOEXW(ct.Structure):
    _fields_ = [("StartupInfo", STARTUPINFOW), ("lpAttributeList", HANDLE)]


class PROCESS_INFORMATION(ct.Structure):
    _fields_ = [("hProcess", HANDLE), ("hThread", HANDLE), ("dwProcessId", DWORD), ("dwThreadId", DWORD)]


class FILETIME(ct.Structure):
    _fields_ = [("low", DWORD), ("high", DWORD)]


class JOBOBJECT_BASIC_LIMIT_INFORMATION(ct.Structure):
    _fields_ = [("PerProcessUserTimeLimit", ct.c_int64), ("PerJobUserTimeLimit", ct.c_int64),
                ("LimitFlags", DWORD), ("MinimumWorkingSetSize", SIZE_T), ("MaximumWorkingSetSize", SIZE_T),
                ("ActiveProcessLimit", DWORD), ("Affinity", SIZE_T), ("PriorityClass", DWORD), ("SchedulingClass", DWORD)]


class IO_COUNTERS(ct.Structure):
    _fields_ = [(name, ct.c_uint64) for name in ("ReadOperationCount", "WriteOperationCount", "OtherOperationCount",
                                               "ReadTransferCount", "WriteTransferCount", "OtherTransferCount")]


class JOBOBJECT_EXTENDED_LIMIT_INFORMATION(ct.Structure):
    _fields_ = [("BasicLimitInformation", JOBOBJECT_BASIC_LIMIT_INFORMATION), ("IoInfo", IO_COUNTERS),
                ("ProcessMemoryLimit", SIZE_T), ("JobMemoryLimit", SIZE_T),
                ("PeakProcessMemoryUsed", SIZE_T), ("PeakJobMemoryUsed", SIZE_T)]


def command_line(argv):
    if not isinstance(argv, (list, tuple)) or not argv or any(not isinstance(value, str) or "\x00" in value for value in argv) or not argv[0]:
        raise ValueError("ConPTY requires a nonempty string argument list without NULs")
    return subprocess.list2cmdline(argv)


def environment_block(environment):
    pairs, seen = [], set()
    for name, value in environment.items():
        if (not isinstance(name, str) or not name or "\x00" in name or "=" in name[1:]
                or not isinstance(value, str) or "\x00" in value):
            raise ValueError("Invalid Windows child environment entry")
        if name.casefold() in seen:
            raise ValueError("Duplicate Windows child environment name")
        seen.add(name.casefold())
        pairs.append((name, value))
    text = "\x00".join(name + "=" + value for name, value in sorted(pairs, key=lambda item: item[0].casefold())) + "\x00\x00"
    return text.encode("utf-16-le", errors="surrogatepass")


def _coord(rows, cols):
    if type(rows) is not int or type(cols) is not int or not 1 <= rows <= 32767 or not 1 <= cols <= 32767:
        raise ValueError("ConPTY dimensions must be integers between 1 and 32767")
    return COORD(cols, rows)


@lru_cache(maxsize=1)
def _api():
    if os.name != "nt":
        raise OSError("ConPTY is available only on Windows")
    kernel = ct.WinDLL("kernel32", use_last_error=True)
    functions = {
        "CreatePipe": ([ct.POINTER(HANDLE), ct.POINTER(HANDLE), HANDLE, DWORD], BOOL),
        "CreatePseudoConsole": ([COORD, HANDLE, HANDLE, DWORD, ct.POINTER(HANDLE)], ct.c_int32),
        "ResizePseudoConsole": ([HANDLE, COORD], ct.c_int32), "ClosePseudoConsole": ([HANDLE], None),
        "InitializeProcThreadAttributeList": ([HANDLE, DWORD, DWORD, ct.POINTER(SIZE_T)], BOOL),
        "UpdateProcThreadAttribute": ([HANDLE, DWORD, SIZE_T, HANDLE, SIZE_T, HANDLE, HANDLE], BOOL),
        "DeleteProcThreadAttributeList": ([HANDLE], None),
        "CreateProcessW": ([ct.c_wchar_p, HANDLE, HANDLE, HANDLE, BOOL, DWORD, HANDLE, ct.c_wchar_p, HANDLE,
                             ct.POINTER(PROCESS_INFORMATION)], BOOL),
        "CreateJobObjectW": ([HANDLE, ct.c_wchar_p], HANDLE),
        "SetInformationJobObject": ([HANDLE, ct.c_int32, HANDLE, DWORD], BOOL),
        "AssignProcessToJobObject": ([HANDLE, HANDLE], BOOL),
        "TerminateJobObject": ([HANDLE, DWORD], BOOL), "TerminateProcess": ([HANDLE, DWORD], BOOL),
        "ResumeThread": ([HANDLE], DWORD), "CloseHandle": ([HANDLE], BOOL),
        "WaitForSingleObject": ([HANDLE, DWORD], DWORD),
        "GetExitCodeProcess": ([HANDLE, ct.POINTER(DWORD)], BOOL),
        "GetProcessTimes": ([HANDLE, ct.POINTER(FILETIME), ct.POINTER(FILETIME), ct.POINTER(FILETIME), ct.POINTER(FILETIME)], BOOL),
        "ReadFile": ([HANDLE, HANDLE, DWORD, ct.POINTER(DWORD), HANDLE], BOOL),
        "WriteFile": ([HANDLE, HANDLE, DWORD, ct.POINTER(DWORD), HANDLE], BOOL),
        "OpenThread": ([DWORD, BOOL, DWORD], HANDLE), "GetCurrentThreadId": ([], DWORD),
        "CancelSynchronousIo": ([HANDLE], BOOL),
    }
    for name, (arguments, result) in functions.items():
        function = getattr(kernel, name)
        function.argtypes, function.restype = arguments, result
    return kernel


def _checked(result):
    if not result:
        raise ct.WinError(ct.get_last_error())
    return result


def _hresult(result):
    if result < 0:
        raise OSError(f"ConPTY operation failed with HRESULT 0x{result & 0xffffffff:08x}")


class PtyProcess:
    """The concrete spawn/read/write/resize/wait/terminate surface Hermes uses."""

    supports_bytes = True
    _BUFFER_LIMIT = 1024 * 1024

    @staticmethod
    def is_available():
        try:
            _api()
            return True
        except (OSError, AttributeError):
            return False

    @classmethod
    def spawn(cls, argv, *, cwd=None, env=None, dimensions=(24, 80)):
        text, size = command_line(argv), _coord(*dimensions)
        block = environment_block(os.environ if env is None else env)
        kernel = _api()
        input_read, input_write, output_read, output_write, pc = (HANDLE() for _ in range(5))
        process, attributes, initialized = PROCESS_INFORMATION(), None, False
        job, launched, transferred = None, False, False
        try:
            _checked(kernel.CreatePipe(ct.byref(input_read), ct.byref(input_write), None, 0))
            _checked(kernel.CreatePipe(ct.byref(output_read), ct.byref(output_write), None, 0))
            _hresult(kernel.CreatePseudoConsole(size, input_read, output_write, 0, ct.byref(pc)))
            required = SIZE_T()
            kernel.InitializeProcThreadAttributeList(None, 1, 0, ct.byref(required))
            if not 1 <= required.value <= 1024 * 1024:
                raise OSError("Invalid ConPTY process attribute size")
            attributes = ct.create_string_buffer(required.value)
            _checked(kernel.InitializeProcThreadAttributeList(attributes, 1, 0, ct.byref(required)))
            initialized = True
            _checked(kernel.UpdateProcThreadAttribute(attributes, 0, 0x00020016, pc, ct.sizeof(HANDLE), None, None))
            startup = STARTUPINFOEXW()
            startup.StartupInfo.cb = ct.sizeof(STARTUPINFOEXW)
            startup.lpAttributeList = ct.cast(attributes, HANDLE)
            command = ct.create_unicode_buffer(text)
            environment = ct.create_string_buffer(block)
            _checked(kernel.CreateProcessW(None, command, None, None, False, 0x00080000 | 0x00000400 | 0x00000004,
                                           environment, cwd, ct.byref(startup), ct.byref(process)))
            launched = True
            job = _checked(kernel.CreateJobObjectW(None, None))
            limits = JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
            limits.BasicLimitInformation.LimitFlags = 0x00002000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
            _checked(kernel.SetInformationJobObject(job, 9, ct.byref(limits), ct.sizeof(limits)))
            _checked(kernel.AssignProcessToJobObject(job, process.hProcess))
            times = [FILETIME() for _ in range(4)]
            _checked(kernel.GetProcessTimes(process.hProcess, *(ct.byref(value) for value in times)))
            birth = ((times[0].high << 32) | times[0].low) / 10_000_000 - 11_644_473_600
            result = cls.__new__(cls)
            result.pid, result.birth, result.exitstatus = int(process.dwProcessId), birth, None
            result._kernel, result._process, result._job = kernel, process.hProcess, job
            result._pc, result._input, result._output = pc.value, input_write.value, output_read.value
            result._condition = threading.Condition()
            result._process_lock, result._pc_lock, result._write_lock = (threading.Lock() for _ in range(3))
            result._chunks, result._buffered, result._writers = deque(), 0, {}
            result._closed, result._eof, result._draining = False, False, False
            result._reader_handle = None
            result._reader = threading.Thread(target=result._pump, name=f"hermes-conpty-{result.pid}", daemon=True)
            # Drain independently before resuming: even immediate output cannot
            # fill the synchronous console pipe while startup is finishing.
            result._reader.start()
            transferred = True
            if kernel.ResumeThread(process.hThread) == 0xffffffff:
                error = ct.get_last_error()
                result.close()
                raise ct.WinError(error)
            return result
        finally:
            if initialized:
                kernel.DeleteProcThreadAttributeList(attributes)
            for handle in (input_read.value, output_write.value, process.hThread):
                if handle:
                    kernel.CloseHandle(handle)
            if not transferred:
                if launched:
                    kernel.TerminateProcess(process.hProcess, 1)
                    kernel.WaitForSingleObject(process.hProcess, 5000)
                # Before a reader exists, close its endpoint first. ConPTY
                # itself can emit output even before the child is resumed.
                if output_read.value:
                    kernel.CloseHandle(output_read)
                if pc.value:
                    kernel.ClosePseudoConsole(pc)
                for handle in (job, process.hProcess, input_write.value):
                    if handle:
                        kernel.CloseHandle(handle)

    def _pump(self):
        kernel = self._kernel
        own_thread = kernel.OpenThread(1, False, kernel.GetCurrentThreadId())
        with self._condition:
            self._reader_handle = own_thread
        buffer, count = ct.create_string_buffer(65536), DWORD()
        try:
            while kernel.ReadFile(self._output, buffer, len(buffer), ct.byref(count), None) and count.value:
                content = buffer.raw[:count.value]
                with self._condition:
                    self._condition.wait_for(lambda: self._draining or self._closed or self._buffered < self._BUFFER_LIMIT)
                    if not self._closed:
                        self._chunks.append(content)
                        self._buffered += len(content)
                    self._condition.notify_all()
        finally:
            kernel.CloseHandle(self._output)
            with self._condition:
                self._output, self._reader_handle, self._eof = None, None, True
                if own_thread:
                    kernel.CloseHandle(own_thread)
                self._condition.notify_all()

    def _close_console(self):
        with self._pc_lock:
            if self._pc is None:
                return
            with self._condition:
                self._draining = True
                self._condition.notify_all()
            self._kernel.ClosePseudoConsole(self._pc)
            self._pc = None

    def isalive(self):
        with self._process_lock:
            if self._process is None or self._closed:
                return False
            status = self._kernel.WaitForSingleObject(self._process, 0)
            if status == 0x00000102:
                return True
            if status != 0:
                raise ct.WinError(ct.get_last_error())
        self._close_console()
        with self._condition:
            return bool(self._buffered or not self._eof)

    def read(self, size=4096):
        if type(size) is not int or size <= 0:
            raise ValueError("ConPTY read size must be positive")
        with self._condition:
            if not self._chunks and not self._closed and not self._eof:
                self._condition.wait(timeout=0.02)
            if not self._chunks:
                if self._closed or self._eof:
                    raise EOFError("ConPTY output closed")
                return b""
            chunk = self._chunks.popleft()
            content, remainder = chunk[:size], chunk[size:]
            if remainder:
                self._chunks.appendleft(remainder)
            self._buffered -= len(content)
            self._condition.notify_all()
            return content

    def write(self, data):
        content = data.encode("utf-8", errors="surrogatepass") if isinstance(data, str) else bytes(data)
        if not content:
            return 0
        with self._write_lock:
            kernel = self._kernel
            with self._condition:
                if self._closed or self._input is None:
                    raise EOFError("ConPTY input closed")
                thread_id = kernel.GetCurrentThreadId()
                own_thread = _checked(kernel.OpenThread(1, False, thread_id))
                self._writers[thread_id] = own_thread
                handle = self._input
            try:
                position, written = 0, DWORD()
                while position < len(content):
                    buffer = ct.create_string_buffer(content[position:position + 32768])
                    _checked(kernel.WriteFile(handle, buffer, len(buffer) - 1, ct.byref(written), None))
                    if not written.value:
                        raise EOFError("ConPTY input pipe made no progress")
                    position += written.value
                return position
            finally:
                with self._condition:
                    self._writers.pop(thread_id, None)
                    if self._closed and not self._writers and self._input is not None:
                        kernel.CloseHandle(self._input)
                        self._input = None
                    self._condition.notify_all()
                kernel.CloseHandle(own_thread)

    def setwinsize(self, rows, cols):
        size = _coord(rows, cols)
        with self._pc_lock:
            if self._pc is None:
                raise EOFError("ConPTY has closed")
            _hresult(self._kernel.ResizePseudoConsole(self._pc, size))

    def wait(self, timeout=None):
        milliseconds = 0xffffffff if timeout is None else min(0xfffffffe, max(0, int(float(timeout) * 1000)))
        with self._process_lock:
            if self._process is None:
                return self.exitstatus
            result = self._kernel.WaitForSingleObject(self._process, milliseconds)
            if result == 0x00000102:
                raise TimeoutError("ConPTY child did not exit")
            if result != 0:
                raise ct.WinError(ct.get_last_error())
            code = DWORD()
            _checked(self._kernel.GetExitCodeProcess(self._process, ct.byref(code)))
            self.exitstatus = int(code.value)
        self._close_console()
        return self.exitstatus

    def terminate(self, force=False):
        if not force:
            try:
                self.write(b"\x03")
                self.wait(timeout=0.2)
                return True
            except (OSError, EOFError, TimeoutError):
                return False
        self.close()
        return True

    def close(self):
        with self._condition:
            if self._closed:
                return
            self._closed, self._draining = True, True
            for thread in self._writers.values():
                self._kernel.CancelSynchronousIo(thread)
            self._condition.notify_all()
        # Stable job/process handles retain ownership even after a PID is reused.
        # Closing our noninherited final job handle also enforces kill-on-close
        # if TerminateJobObject could not report completion.
        job, self._job = self._job, None
        if job:
            self._kernel.TerminateJobObject(job, 1)
            self._kernel.CloseHandle(job)
        try:
            self.wait(timeout=5)
        except (OSError, TimeoutError):
            pass
        self._close_console()
        self._reader.join(timeout=1)
        with self._condition:
            if self._reader_handle:
                self._kernel.CancelSynchronousIo(self._reader_handle)
        self._reader.join(timeout=1)
        with self._condition:
            if not self._writers and self._input is not None:
                self._kernel.CloseHandle(self._input)
                self._input = None
        with self._process_lock:
            for name in ("_job", "_process"):
                handle = getattr(self, name)
                if handle:
                    self._kernel.CloseHandle(handle)
                    setattr(self, name, None)
