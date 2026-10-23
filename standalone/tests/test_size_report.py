import hashlib
import io
from pathlib import Path
import tempfile
import unittest
import zipfile

from standalone import size_report
from standalone.tests.test_bootstrap import fixture_archive, write_artifact


class SizeReportTests(unittest.TestCase):
    def test_reads_both_envelopes_without_executing_the_application(self):
        with tempfile.TemporaryDirectory() as directory:
            artifact = Path(directory) / "one.py"
            original = fixture_archive()
            for compression in ("deflate", "lzma"):
                with self.subTest(compression=compression):
                    write_artifact(artifact, original, compression=compression)
                    info, restored = size_report.read_payload(artifact)
                    self.assertEqual(restored, original)
                    self.assertEqual(info["sha256"], hashlib.sha256(original).hexdigest())
                    self.assertEqual(list(Path(directory).iterdir()), [artifact])

    def test_reports_all_file_bytes_and_verifies_lossless_compression(self):
        original = fixture_archive({"app/skills/example/SKILL.md": b"skill document"})
        report = size_report.compare_archive(original)
        with zipfile.ZipFile(io.BytesIO(original)) as archive:
            self.assertEqual(report["file_count"], len(archive.infolist()))
            self.assertEqual(
                sum(item["bytes"] for item in report["categories"].values()),
                sum(item.file_size for item in archive.infolist()),
            )
        self.assertEqual(report["categories"]["app/skills"]["bytes"], len(b"skill document"))
        for item in report["encodings"].values():
            self.assertTrue(item["roundtrip_exact"])
            self.assertGreater(item["literal_layouts"]["100"]["source_bytes"], item["literal_layouts"]["8192"]["source_bytes"])


if __name__ == "__main__":
    unittest.main()
