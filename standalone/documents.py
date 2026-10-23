"""Document extraction boundary for the immutable standard-library runtime.

Text extraction stays local. Hosted OCR is used only when the original caller
explicitly requests it with its profile's credential. This module is qualified
application code, never installed under the native ``anydoc`` namespace.
"""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path
import zipfile


MAX_BYTES = 50 * 1024 * 1024
MAX_EXPANDED_BYTES = 100 * 1024 * 1024
MAX_TEXT_CHARS = 20_000_000


class ConvertError(ValueError):
    pass


class UnsupportedError(ConvertError):
    pass


class MalformedError(ConvertError):
    pass


class EncryptedError(ConvertError):
    pass


class ResourceLimitError(ConvertError):
    pass


class MissingPartError(ConvertError):
    pass


class HostedError(ConvertError):
    pass


class NeedsOcrError(ConvertError):
    def __init__(self, pages):
        self.pages = list(pages)
        super().__init__("PDF pages require OCR: " + ", ".join(map(str, self.pages)))


def open_zip(data):
    archive = zipfile.ZipFile(io.BytesIO(data))
    try:
        names, total = set(), 0
        for info in archive.infolist():
            if info.flag_bits & 1:
                raise EncryptedError("Encrypted document ZIP member")
            if info.filename in names:
                raise MalformedError("Document contains duplicate ZIP members")
            names.add(info.filename)
            total += info.file_size
            if len(names) > 10000 or total > MAX_EXPANDED_BYTES:
                raise ResourceLimitError("Document archive exceeds the expanded data limit")
    except BaseException:
        archive.close()
        raise
    return archive


def _format(data, hint):
    if hint:
        return str(hint).lower().lstrip(".")
    if data.startswith(b"%PDF-"):
        return "pdf"
    if data.lstrip().startswith(b"{\\rtf"):
        return "rtf"
    if data.startswith(b"PK"):
        with open_zip(data) as archive:
            names = set(archive.namelist())
            for part, kind in (("word/document.xml", "docx"), ("xl/workbook.xml", "xlsx"),
                               ("xl/workbook.bin", "xlsb"), ("ppt/presentation.xml", "pptx"),
                               ("META-INF/container.xml", "epub"), ("content.xml", "odt")):
                if part in names:
                    return kind
    if data.startswith(b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"):
        import olefile
        with olefile.OleFileIO(io.BytesIO(data)) as archive:
            for part, kind in (("WordDocument", "doc"), ("PowerPoint Document", "ppt"),
                               ("Workbook", "xls"), ("Book", "xls")):
                if archive.exists(part):
                    return kind
    raise UnsupportedError("Document format is not recognized")


def pdf_page_texts(data):
    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(data), strict=True)
    if reader.is_encrypted:
        try:
            unlocked = reader.decrypt("")
        except Exception as exc:
            raise EncryptedError("PDF is encrypted") from exc
        if not unlocked:
            raise EncryptedError("PDF requires a password")
    if len(reader.pages) > 10000:
        raise ResourceLimitError("PDF page count exceeds the extraction limit")
    texts, total = [], 0
    for page in reader.pages:
        contents = page.get_contents()
        if contents is not None and len(contents.get_data()) > MAX_EXPANDED_BYTES:
            raise ResourceLimitError("PDF content stream exceeds the extraction limit")
        text = page.extract_text() or ""
        total += len(text)
        if total > MAX_TEXT_CHARS:
            raise ResourceLimitError("Extracted document text exceeds the limit")
        texts.append(text)
    return texts


def _hosted(data, *, api_key, api_url=None):
    import httpx
    from agent.ssl_verify import resolve_httpx_verify

    if not api_key:
        raise HostedError("Hosted OCR requires the caller's profile credential")
    endpoint = (api_url or "https://api.firecrawl.dev").rstrip("/")
    with httpx.Client(verify=resolve_httpx_verify(base_url=endpoint), timeout=120, follow_redirects=False,
                      headers={"Authorization": "Bearer " + api_key}) as client:
        response = client.post(endpoint + "/v2/parse",
                               files={"file": ("document.pdf", data, "application/pdf"),
                                      "options": (None, json.dumps({"formats": ["markdown"]}), "application/json")})
        if not response.is_success:
            raise HostedError(f"Hosted OCR returned HTTP {response.status_code}")
        result = response.json()
        if not isinstance(result, dict) or result.get("success") is False:
            raise HostedError("Hosted OCR failed to convert the document")
        content = result.get("data", result) if isinstance(result, dict) else None
        if isinstance(content, list):
            markdown = "\n\n".join(item.get("markdown", "") for item in content if isinstance(item, dict))
        else:
            markdown = content.get("markdown", "") if isinstance(content, dict) else ""
        if not isinstance(markdown, str) or not markdown.strip():
            raise HostedError("Hosted OCR returned no document text")
        return markdown


def to_markdown(path, *, ocr=None, api_key=None, api_url=None):
    path = Path(path)
    if path.stat().st_size > MAX_BYTES:
        raise ResourceLimitError("Document exceeds the input byte limit")
    return to_markdown_bytes(path.read_bytes(), path.suffix, ocr=ocr, api_key=api_key, api_url=api_url)


def to_markdown_bytes(data, format=None, *, ocr=None, api_key=None, api_url=None):
    if len(data) > MAX_BYTES:
        raise ResourceLimitError("Document exceeds the input byte limit")
    if ocr not in (None, "hosted"):
        raise ValueError("Unsupported OCR mode")
    kind = _format(data, format)
    try:
        if kind == "pdf":
            if ocr == "hosted":
                return _hosted(data, api_key=api_key, api_url=api_url)
            texts = pdf_page_texts(data)
            missing = [index + 1 for index, text in enumerate(texts) if not text.strip()]
            if missing:
                raise NeedsOcrError(missing)
            output = "\n\n".join(texts)
        elif kind in ("docx", "docm", "xlsx", "xlsm"):
            from tools.read_extract import _temp_copy, _extract_docx, _extract_xlsx
            with open_zip(data):
                pass
            with _temp_copy(data, ".docx" if kind.startswith("doc") else ".xlsx") as temporary:
                output = (_extract_docx if kind.startswith("doc") else _extract_xlsx)(temporary)
        elif kind in ("pptx", "pptm", "ppsx", "ppsm", "odt", "ods", "odp", "epub"):
            from standalone.documents_xml import extract
            output = extract(data, kind)
        elif kind in ("doc", "ppt", "pps", "pot", "xls", "xlsb"):
            from standalone.documents_legacy import extract
            output = extract(data, kind)
        elif kind == "rtf":
            from striprtf.striprtf import rtf_to_text
            output = rtf_to_text(data.decode("latin-1"), errors="replace")
        elif kind == "csv":
            text = data.decode("utf-8-sig")
            try:
                dialect = csv.Sniffer().sniff(text[:8192])
            except csv.Error:
                dialect = csv.excel
            output = "\n".join("\t".join(row) for row in csv.reader(io.StringIO(text), dialect))
        else:
            raise UnsupportedError("Unsupported document format: " + kind)
    except ConvertError:
        raise
    except Exception as exc:
        raise MalformedError(f"Document could not be decoded: {type(exc).__name__}: {exc}") from exc
    if not isinstance(output, str) or not output.strip():
        raise MalformedError("Document contains no extractable text")
    if len(output) > MAX_TEXT_CHARS:
        raise ResourceLimitError("Extracted document text exceeds the limit")
    return output.rstrip() + "\n"
