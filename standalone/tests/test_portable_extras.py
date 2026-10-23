"""Pure model interpreters are independent of original native wheel markers."""

import unittest
from unittest.mock import patch

from pm.extras import available, ensure_import, extra_supported, missing
from standalone.packages import replacement_anchors


class PortableExtraTests(unittest.TestCase):
    def test_bound_model_runtime_overrides_native_wheel_markers_only_for_ported_engines(self):
        excluded = {"wake-openwakeword": "sys_platform == 'linux'", "wake-sherpa": "sys_platform == 'linux'"}
        environment = {"sys_platform": "darwin", "platform_machine": "x86_64", "platform_system": "Darwin"}
        with patch("standalone.packages.active", return_value=True), patch("pm.extras._platform_gates", return_value=excluded), patch("pm.extras._evaluate_in_runtime", return_value=False):
            self.assertTrue(extra_supported("wake-openwakeword", environment=environment, importable=lambda _name:False))
            self.assertFalse(extra_supported("wake-sherpa", environment=environment, importable=lambda _name:False))
        self.assertEqual(replacement_anchors("wake-openwakeword"), ("standalone.wake_openword",))

    def test_source_install_keeps_original_sdk_marker_behavior(self):
        environment = {"sys_platform": "darwin", "platform_machine": "x86_64", "platform_system": "Darwin"}
        with patch("standalone.packages.active", return_value=False), patch("pm.extras._platform_gates", return_value={"wake-openwakeword":"sys_platform == 'linux'"}), patch("pm.extras._evaluate_in_runtime", return_value=False):
            self.assertFalse(extra_supported("wake-openwakeword", environment=environment, importable=lambda _name:False))

    def test_whisper_uses_qualified_runtime_without_a_native_sdk_or_installer(self):
        environments = (
            {"sys_platform":"darwin", "platform_system":"Darwin", "platform_machine":"x86_64"},
            {"sys_platform":"win32", "platform_system":"Windows", "platform_machine":"ARM64"},
        )
        present = {"standalone.local_whisper", "standalone.audio_device"}
        with patch("standalone.packages.active", return_value=True), patch("pm.extras._importable", side_effect=lambda name: name in present), patch("pm.client.sync_venv", side_effect=AssertionError("Runtime package installation attempted")):
            for extra in ("stt-whisper", "voice"):
                for environment in environments:
                    self.assertTrue(extra_supported(extra, environment=environment, importable=lambda _name:False))
                self.assertTrue(available(extra))
                self.assertEqual(missing(extra), ())
                ensure_import(extra)
            present.remove("standalone.local_whisper")
            self.assertFalse(available("stt-whisper"))
            self.assertEqual(missing("stt-whisper"), ("standalone.local_whisper",))
        with patch("standalone.packages.active", return_value=False), patch("pm.extras._importable", return_value=False):
            self.assertEqual(missing("stt-whisper"), ("faster_whisper",))
