"""OOXML presentation, OpenDocument and EPUB text extraction."""

from html.parser import HTMLParser
import posixpath
from urllib.parse import unquote, urlsplit

from defusedxml import ElementTree as ET

from standalone.documents import EncryptedError, MissingPartError, ResourceLimitError, open_zip


def _xml(archive, name):
    try:
        data = archive.read(name)
    except KeyError as exc:
        raise MissingPartError("Document is missing " + name) from exc
    return ET.fromstring(data)


def _part(base, target):
    url = urlsplit(target)
    if url.scheme or url.netloc:
        raise MissingPartError("External document parts are not fetched")
    path = posixpath.normpath(posixpath.join(base, unquote(url.path)))
    if path == ".." or path.startswith("../") or path.startswith("/"):
        raise MissingPartError("Document part escapes its archive")
    return path


def _text(node):
    local = node.tag.rsplit("}", 1)[-1]
    if local == "s":
        count = int(next((v for k, v in node.attrib.items() if k.rsplit("}", 1)[-1] == "c"), "1"))
        if not 1 <= count <= 10000:
            raise ResourceLimitError("Document space repeat exceeds the limit")
        result = " " * count
    elif local == "tab":
        result = "\t"
    elif local in ("br", "line-break"):
        result = "\n"
    else:
        result = node.text or ""
        result += "".join(_text(child) + (child.tail or "") for child in node)
    return result


def _table(rows):
    rows = [[value.replace("|", "\\|").replace("\n", "<br>") for value in row] for row in rows]
    if not rows:
        return ""
    width = max(map(len, rows))
    lines = ["| " + " | ".join(row + [""] * (width - len(row))) + " |" for row in rows]
    lines.insert(1, "| " + " | ".join(["---"] * width) + " |")
    return "\n".join(lines)


def _pptx(archive):
    ns = {"p": "http://schemas.openxmlformats.org/presentationml/2006/main",
          "a": "http://schemas.openxmlformats.org/drawingml/2006/main"}
    rel_ns = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}"
    relations = {rel.get("Id"): rel for rel in _xml(archive, "ppt/_rels/presentation.xml.rels")}
    lines = []
    for index, slide in enumerate(_xml(archive, "ppt/presentation.xml").findall("p:sldIdLst/p:sldId", ns), 1):
        rel = relations.get(slide.get(rel_ns + "id"))
        if rel is None or rel.get("TargetMode") == "External":
            raise MissingPartError("Presentation slide relationship is invalid")
        target = rel.get("Target", "")
        part = target.lstrip("/") if target.startswith("/ppt/") else _part("ppt", target)
        root = _xml(archive, part)
        lines.append(f"## Slide {index}")
        for shape in root.findall("p:cSld/p:spTree/*", ns):
            tables = shape.findall(".//a:tbl", ns)
            if tables:
                for table in tables:
                    lines.append(_table([["\n".join(_text(p) for p in cell.findall(".//a:p", ns)) for cell in row.findall("a:tc", ns)] for row in table.findall("a:tr", ns)]))
                continue
            title = any(ph.get("type") in ("title", "ctrTitle") for ph in shape.findall(".//p:ph", ns))
            for paragraph in shape.findall(".//a:p", ns):
                text = _text(paragraph).strip()
                if not text:
                    continue
                bullet = paragraph.find("a:pPr/a:buChar", ns) is not None
                lines.append(("### " if title else "- " if bullet else "") + text)
        lines.append("")
    return "\n\n".join(lines)


def _odf(archive):
    if "META-INF/manifest.xml" in archive.namelist():
        if any(node.tag.rsplit("}", 1)[-1] == "encryption-data" for node in _xml(archive, "META-INF/manifest.xml").iter()):
            raise EncryptedError("OpenDocument is encrypted")
    root = _xml(archive, "content.xml")
    lines = []
    def visit(node):
        local = node.tag.rsplit("}", 1)[-1]
        if local in ("h", "p"):
            level = next((value for key, value in node.attrib.items() if key.rsplit("}", 1)[-1] == "outline-level"), "1")
            lines.append(("#" * max(1, min(6, int(level))) + " " if local == "h" else "") + _text(node))
        elif local == "table":
            rows = []
            for row in node.iter():
                if row.tag.rsplit("}", 1)[-1] != "table-row":
                    continue
                cells = []
                for cell in row:
                    if cell.tag.rsplit("}", 1)[-1] not in ("table-cell", "covered-table-cell"):
                        continue
                    repeat = int(next((v for k, v in cell.attrib.items() if k.rsplit("}", 1)[-1] == "number-columns-repeated"), "1"))
                    if not 1 <= repeat <= 256 or len(cells) + repeat > 256:
                        raise ResourceLimitError("OpenDocument table has too many columns")
                    cells.extend([_text(cell).strip()] * repeat)
                repeat = int(next((v for k, v in row.attrib.items() if k.rsplit("}", 1)[-1] == "number-rows-repeated"), "1"))
                if not 1 <= repeat <= 5000 or len(rows) + repeat > 5000:
                    raise ResourceLimitError("OpenDocument table has too many rows")
                rows.extend([cells] * repeat)
            lines.append(_table(rows))
        else:
            for child in node:
                visit(child)
    visit(root)
    return "\n\n".join(lines)


class _HTML(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts, self.skip = [], 0

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style"):
            self.skip += 1
        if self.skip:
            return
        if tag in ("p", "div", "section", "article", "tr"):
            self.parts.append("\n\n")
        elif tag == "li":
            self.parts.append("\n- ")
        elif tag == "br":
            self.parts.append("\n")
        elif tag in ("h1", "h2", "h3", "h4", "h5", "h6"):
            self.parts.append("\n\n" + "#" * int(tag[1]) + " ")
        elif tag in ("td", "th"):
            self.parts.append("\t")

    def handle_endtag(self, tag):
        if tag in ("script", "style"):
            self.skip = max(0, self.skip - 1)
        elif not self.skip and tag in ("p", "div", "section", "article", "h1", "h2", "h3", "h4", "h5", "h6"):
            self.parts.append("\n\n")

    def handle_data(self, text):
        if not self.skip:
            self.parts.append(text)


def _epub(archive):
    container = _xml(archive, "META-INF/container.xml")
    rootfile = next((node.get("full-path") for node in container.iter() if node.tag.rsplit("}", 1)[-1] == "rootfile"), None)
    if not rootfile:
        raise MissingPartError("EPUB has no package rootfile")
    rootfile = _part("", rootfile)
    package = _xml(archive, rootfile)
    manifest = {node.get("id"): node.get("href") for node in package.iter() if node.tag.rsplit("}", 1)[-1] == "item"}
    chapters = []
    for item in package.iter():
        if item.tag.rsplit("}", 1)[-1] != "itemref":
            continue
        href = manifest.get(item.get("idref"))
        if href is None:
            raise MissingPartError("EPUB spine references a missing chapter")
        path = _part(posixpath.dirname(rootfile), href)
        parser = _HTML()
        parser.feed(archive.read(path).decode("utf-8-sig"))
        parser.close()
        chapters.append("".join(parser.parts).strip())
    return "\n\n".join(chapters)


def extract(data, kind):
    with open_zip(data) as archive:
        if kind.startswith(("ppt", "pps")):
            return _pptx(archive)
        return _epub(archive) if kind == "epub" else _odf(archive)
