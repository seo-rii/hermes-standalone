"""Original document boundaries with pure readers and deterministic wire fixtures."""

import importlib.util
import io
import json
import os
from pathlib import Path
import struct
import threading
import unittest
from unittest import mock
import zipfile
from email.parser import BytesParser
from email.policy import default as email_policy
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from standalone.tests import test_transport as _wire_fixture


class DocumentDependencyTests(_wire_fixture.TransportDependencyTests):
    CASES = "standalone.tests.test_documents.DocumentTests"
    DEPENDENCIES = ("pypdf", "striprtf", "olefile", "xlrd", "pyxlsb", "httpx", "defusedxml", "tlslite-ng")


def archive(parts):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as output:
        for name, data in parts.items():
            output.writestr(name, data)
    return buffer.getvalue()


def pdf(text="Original PDF fixture", *, blank_page=False):
    stream = b"BT /F1 12 Tf 72 720 Td (" + text.encode("ascii") + b") Tj ET"
    objects = [b"<< /Type /Catalog /Pages 2 0 R >>",
               b"<< /Type /Pages /Kids [3 0 R" + (b" 6 0 R" if blank_page else b"") + b"] /Count " + (b"2" if blank_page else b"1") + b" >>",
               b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>",
               b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
               b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"\nendstream"]
    if blank_page:
        objects.append(b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] >>")
    data, offsets = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n"), [0]
    for index, obj in enumerate(objects, 1):
        offsets.append(len(data))
        data.extend(str(index).encode() + b" 0 obj\n" + obj + b"\nendobj\n")
    start = len(data)
    data.extend(b"xref\n0 " + str(len(offsets)).encode() + b"\n0000000000 65535 f \n")
    for offset in offsets[1:]:
        data.extend(f"{offset:010d} 00000 n \n".encode())
    data.extend(b"trailer\n<< /Size " + str(len(offsets)).encode() + b" /Root 1 0 R >>\nstartxref\n" + str(start).encode() + b"\n%%EOF\n")
    return bytes(data)


@unittest.skipUnless(importlib.util.find_spec("pypdf"), "Runs in the reviewed dependency subprocess")
class DocumentTests(unittest.TestCase):
    def test_hosted_ocr_multipart_and_browser_fetch_wire(self):
        from standalone.documents import to_markdown_bytes, HostedError
        from standalone.browser_fetch import fetch_sync, FetchError

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_args):
                pass

            def do_POST(self):
                raw = self.rfile.read(int(self.headers["content-length"]))
                if self.path == "/v2/parse":
                    message = BytesParser(policy=email_policy).parsebytes(
                        b"Content-Type: " + self.headers["content-type"].encode() + b"\r\n\r\n" + raw)
                    body = {part.get_param("name", header="content-disposition"): part.get_payload(decode=True)
                            for part in message.iter_parts()}
                    result = {"success": True, "data": {"markdown": "# Hosted fixture document"}}
                else:
                    body = json.loads(raw)
                    result = {"status_code": 200, "body": "Browser fixture body", "headers": {}, "final_url": body["url"]}
                self.server.records.append((self.path, dict(self.headers), body))
                data = json.dumps(result).encode()
                self.send_response(302 if self.server.redirect else 200)
                self.send_header("Location", "/credential-sink")
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        server.records, server.redirect = [], False
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            endpoint = f"http://127.0.0.1:{server.server_port}"
            with self.assertRaises(HostedError):
                to_markdown_bytes(pdf(), ocr="hosted", api_url=endpoint)
            self.assertEqual(server.records, [])
            converted = to_markdown_bytes(pdf(), ocr="hosted", api_key="fixture-ocr", api_url=endpoint)
            self.assertIn("Hosted fixture", converted)
            path, headers, body = server.records[-1]
            self.assertEqual(path, "/v2/parse")
            self.assertEqual(headers["Authorization"], "Bearer fixture-ocr")
            self.assertEqual(body["file"], pdf())
            self.assertEqual(json.loads(body["options"]), {"formats": ["markdown"]})
            with mock.patch.dict(os.environ, {"BROWSER_USE_API_KEY": "fixture-browser", "FETCH_USE_URL": endpoint,
                                              "SESSION_ID": "fixture-session"}):
                response = fetch_sync("https://fixture.invalid/page", headers={"Accept": "text/plain"}, timeout_ms=5000)
                self.assertEqual(response.text, "Browser fixture body")
                self.assertEqual(server.records[-1][2]["session_id"], "fixture-session")
                self.assertEqual(server.records[-1][1]["X-Browser-Use-API-Key"], "fixture-browser")
                server.redirect = True
                before = len(server.records)
                with self.assertRaises(FetchError):
                    fetch_sync("https://fixture.invalid/page")
                self.assertEqual(len(server.records), before + 1)
                self.assertNotIn("/credential-sink", [record[0] for record in server.records])
        finally:
            server.shutdown()
            server.server_close()
            thread.join(3)

    def test_pdf_text_ocr_refusal_and_aes_encryption(self):
        from standalone import documents
        from pypdf import PdfReader, PdfWriter
        self.assertIn("Original PDF fixture", documents.to_markdown_bytes(pdf()))
        with self.assertRaises(documents.NeedsOcrError) as refusal:
            documents.to_markdown_bytes(pdf(blank_page=True))
        self.assertEqual(refusal.exception.pages, [2])
        for algorithm in ("RC4-128", "AES-128", "AES-256"):
            with self.subTest(algorithm=algorithm):
                writer = PdfWriter(clone_from=PdfReader(io.BytesIO(pdf())))
                writer.encrypt("fixture-password", algorithm=algorithm)
                encrypted = io.BytesIO()
                writer.write(encrypted)
                value = encrypted.getvalue()
                with self.assertRaises(documents.EncryptedError):
                    documents.to_markdown_bytes(value)
                reader = PdfReader(io.BytesIO(value))
                self.assertTrue(reader.decrypt("fixture-password"))
                self.assertIn("Original PDF fixture", reader.pages[0].extract_text())

    def test_odf_epub_pptx_order_tables_and_rtf_unicode(self):
        from standalone.documents import to_markdown_bytes
        odf = archive({"content.xml": '''<office:document-content xmlns:office="urn:oasis:names:tc:opendocument:xmlns:office:1.0" xmlns:text="urn:oasis:names:tc:opendocument:xmlns:text:1.0" xmlns:table="urn:oasis:names:tc:opendocument:xmlns:table:1.0"><office:body><office:text><text:h text:outline-level="2">제목</text:h><text:p>한글<text:s text:c="2"/>본문</text:p><table:table><table:table-row><table:table-cell><text:p>A</text:p></table:table-cell><table:table-cell><text:p>B</text:p></table:table-cell></table:table-row><table:table-row><table:table-cell><text:p>1</text:p></table:table-cell><table:table-cell><text:p>2</text:p></table:table-cell></table:table-row></table:table></office:text></office:body></office:document-content>'''})
        text = to_markdown_bytes(odf)
        self.assertIn("## 제목", text)
        self.assertIn("한글  본문", text)
        self.assertIn("| A | B |", text)
        epub = archive({"META-INF/container.xml": '<container><rootfiles><rootfile full-path="EPUB/package.opf"/></rootfiles></container>',
                        "EPUB/package.opf": '<package><manifest><item id="first" href="two.xhtml"/><item id="second" href="one.xhtml"/></manifest><spine><itemref idref="first"/><itemref idref="second"/></spine></package>',
                        "EPUB/two.xhtml": '<html><body><h1>First</h1><p>한글 &amp; text</p><script>hidden()</script></body></html>',
                        "EPUB/one.xhtml": '<html><body><p>Second</p></body></html>'})
        text = to_markdown_bytes(epub)
        self.assertLess(text.index("First"), text.index("Second"))
        self.assertNotIn("hidden", text)
        pptx = archive({"ppt/presentation.xml": '<p:presentation xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"><p:sldIdLst><p:sldId r:id="r2"/><p:sldId r:id="r1"/></p:sldIdLst></p:presentation>',
                         "ppt/_rels/presentation.xml.rels": '<Relationships><Relationship Id="r1" Target="slides/slide1.xml"/><Relationship Id="r2" Target="slides/slide2.xml"/></Relationships>',
                         **{f"ppt/slides/slide{i}.xml": f'<p:sld xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main" xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"><p:cSld><p:spTree><p:sp><p:txBody><a:p><a:r><a:t>Slide text {i}</a:t></a:r></a:p></p:txBody></p:sp></p:spTree></p:cSld></p:sld>' for i in (1, 2)}})
        text = to_markdown_bytes(pptx)
        self.assertLess(text.index("Slide text 2"), text.index("Slide text 1"))
        self.assertIn("Hello", to_markdown_bytes(b"{\\rtf1\\ansi Hello \\u54620?\\u44544?}", "rtf"))
        self.assertIn("한글", to_markdown_bytes(b"{\\rtf1\\ansi Hello \\u54620?\\u44544?}", "rtf"))

    def test_compound_word_piece_tables_and_powerpoint_atoms(self):
        from standalone.documents_legacy import _word, _powerpoint
        plain, unicode_text = b"First paragraph\r", "두 번째 문단\r".encode("utf-16-le")
        word = bytearray(2048)
        struct.pack_into("<HH", word, 0, 0xA5EC, 0xC1)
        struct.pack_into("<H", word, 32, 14)
        struct.pack_into("<H", word, 62, 22)
        characters = len(plain) + len(unicode_text) // 2
        struct.pack_into("<I", word, 76, characters)
        struct.pack_into("<H", word, 152, 34)
        word[1024:1024 + len(plain)] = plain
        word[1200:1200 + len(unicode_text)] = unicode_text
        pieces = struct.pack("<III", 0, len(plain), characters) + struct.pack("<HIH", 0, 0x40000000 | 2048, 0) + struct.pack("<HIH", 0, 1200, 0)
        table = b"\x02" + struct.pack("<I", len(pieces)) + pieces
        struct.pack_into("<II", word, 154 + 33 * 8, 0, len(table))
        class Compound:
            def __init__(self, streams): self.streams = streams
            def exists(self, name): return name in self.streams
            def openstream(self, name): return io.BytesIO(self.streams[name])
        text = _word(Compound({"WordDocument": word, "0Table": table}))
        self.assertEqual(text, "First paragraph\n두 번째 문단\n")
        body = struct.pack("<HHI", 0, 0xFA0, len(unicode_text)) + unicode_text
        body += struct.pack("<HHI", 0, 0xFA8, len(plain)) + plain
        container = struct.pack("<HHI", 15, 1000, len(body)) + body
        self.assertIn("First paragraph", _powerpoint(Compound({"PowerPoint Document": container})))

    def test_hostile_archive_entities_external_parts_and_encryption_fail_closed(self):
        from standalone import documents
        for data, hint in ((archive({"content.xml": '<!DOCTYPE x [<!ENTITY a "expansion">]><x>&a;</x>'}), "odt"),
                           (archive({"META-INF/container.xml": '<container><rootfile full-path="../../escape"/></container>'}), "epub")):
            with self.subTest(hint=hint), self.assertRaises(documents.ConvertError):
                documents.to_markdown_bytes(data, hint)
        encrypted = archive({"content.xml": "<x>hidden</x>", "META-INF/manifest.xml": '<manifest><encryption-data/></manifest>'})
        with self.assertRaises(documents.EncryptedError):
            documents.to_markdown_bytes(encrypted, "odt")
