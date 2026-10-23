"""SCM buffer invariants on all hosts; operating-system queries only on Windows."""

import ctypes as C
import os
import struct
import unittest

from standalone.process_errors import AccessDenied, Error, NoSuchProcess
from standalone.windows_services import (
    ServiceConfig, ServiceEntry, ServiceStatus, WindowsService, _buffer_text,
    win_service_get, win_service_iter,
)


class ServiceBufferTests(unittest.TestCase):
    def test_dword_status_and_pointer_config_match_the_windows_abi(self):
        status = ServiceStatus.from_buffer_copy(struct.pack("<9I", 0x10, 4, 1, 0, 0, 7, 3000, 12345, 0))
        self.assertEqual((status.service_type, status.state, status.pid, status.wait_hint), (0x10, 4, 12345, 3000))
        self.assertEqual(C.sizeof(ServiceStatus), 36)
        pointer = C.sizeof(C.c_void_p)
        self.assertEqual(ServiceConfig.binpath.offset, 16 if pointer == 8 else 12)
        self.assertEqual(C.sizeof(ServiceConfig), 64 if pointer == 8 else 36)
        self.assertEqual(ServiceEntry.status.offset, 2 * pointer)

    def test_unicode_strings_are_decoded_only_inside_the_owned_response_buffer(self):
        buffer = C.create_string_buffer(256)
        text = 'C:\\Hermes 🔧\\hermes.py --profile 연구'
        payload = text.encode("utf-16-le") + b"\0\0"
        address = C.addressof(buffer) + 64
        C.memmove(address, payload, len(payload))
        config = ServiceConfig.from_buffer(buffer)
        config.binpath = address
        self.assertEqual(_buffer_text(buffer, config.binpath), text)
        for invalid in (C.addressof(buffer) - 2, C.addressof(buffer) + 256, address + 1):
            with self.assertRaises(Error):
                _buffer_text(buffer, invalid)
        self.assertEqual(_buffer_text(buffer, None), "")
        C.memset(address, 65, 192)
        with self.assertRaises(Error):
            _buffer_text(buffer, address)

    def test_invalid_names_and_nonwindows_queries_fail_before_native_loading(self):
        for name in ("", "a\0b", "a" * 257, None):
            with self.assertRaises(ValueError):
                WindowsService(name)
        if os.name != "nt":
            with self.assertRaises(AttributeError):
                win_service_get("fixture-never-loaded")
            with self.assertRaises(AttributeError):
                list(win_service_iter())


@unittest.skipUnless(os.name == "nt", "Real Windows SCM query; no emulated host evidence")
class NativeServiceTests(unittest.TestCase):
    def test_real_enumeration_and_named_query_report_one_accessible_service(self):
        for item in win_service_iter():
            try:
                service = win_service_get(item.name())
                snapshot = service.as_dict()
            except (AccessDenied, NoSuchProcess):
                continue
            self.assertEqual(snapshot["name"], item.name())
            self.assertIsInstance(snapshot["binpath"], str)
            self.assertIsInstance(snapshot["username"], str)
            self.assertIsInstance(snapshot["display_name"], str)
            self.assertIsInstance(snapshot["description"], str)
            self.assertIn(snapshot["status"], ("running", "stopped", "start_pending", "stop_pending",
                                               "pause_pending", "paused", "continue_pending"))
            self.assertIn(snapshot["start_type"], ("boot", "system", "automatic", "manual", "disabled"))
            return
        self.skipTest("No service exposes both its configuration and its status to this user")


if __name__ == "__main__":
    unittest.main()
