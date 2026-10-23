"""Bounded text readers for binary Word/PowerPoint/Excel document containers.

Word piece-table and PowerPoint atom layouts follow Microsoft's MS-DOC and
MS-PPT open file-format specifications. Formatting and revision-history layout
are not reproduced by these text readers.
"""

import io
import struct

from standalone.documents import EncryptedError, MalformedError, MAX_EXPANDED_BYTES, ResourceLimitError, UnsupportedError, open_zip


def _clean(text):
    # Field instructions precede the separator; their displayed result follows.
    output, fields = [], []
    for character in text:
        if character == "\x13":
            fields.append(False)
        elif character == "\x14" and fields:
            fields[-1] = True
        elif character == "\x15" and fields:
            fields.pop()
        elif not fields or all(fields):
            if character in ("\r", "\x0b", "\x0c"):
                output.append("\n")
            elif character == "\x07":
                output.append("\t")
            elif character in ("\t", "\n") or ord(character) >= 32:
                output.append(character)
    return "".join(output)


def _word(archive):
    word = archive.openstream("WordDocument").read()
    if len(word) < 154 or word[:2] != b"\xec\xa5":
        raise MalformedError("Invalid binary Word FIB header")
    version, flags = struct.unpack_from("<H", word, 2)[0], struct.unpack_from("<H", word, 10)[0]
    if flags & 0x8100:
        raise EncryptedError("Binary Word document is encrypted or obfuscated")
    if version < 0xC1:
        raise UnsupportedError("Word versions before Word 97 are not supported")
    cursor = 32
    csw = struct.unpack_from("<H", word, cursor)[0]
    cursor += 2 + csw * 2
    if cursor + 2 > len(word):
        raise MalformedError("Truncated Word FIB")
    cslw = struct.unpack_from("<H", word, cursor)[0]
    lw_start = cursor + 2
    cursor = lw_start + cslw * 4
    if cursor + 2 > len(word) or cslw < 4:
        raise MalformedError("Truncated Word FIB long fields")
    text_chars = struct.unpack_from("<I", word, lw_start + 12)[0]
    count = struct.unpack_from("<H", word, cursor)[0]
    array = cursor + 2
    if count <= 33 or array + count * 8 > len(word):
        raise MalformedError("Word FIB has no text piece-table location")
    fc, length = struct.unpack_from("<II", word, array + 33 * 8)
    table_name = "1Table" if flags & 0x0200 else "0Table"
    if not archive.exists(table_name):
        raise MalformedError("Word document has no table stream")
    table = archive.openstream(table_name).read()
    if fc + length > len(table) or not length:
        raise MalformedError("Invalid Word piece-table bounds")
    clx, offset = table[fc:fc + length], 0
    while offset < len(clx) and clx[offset] == 1:
        if offset + 3 > len(clx):
            raise MalformedError("Truncated Word formatting run")
        offset += 3 + struct.unpack_from("<H", clx, offset + 1)[0]
    if offset + 5 > len(clx) or clx[offset] != 2:
        raise MalformedError("Word text piece table is missing")
    size = struct.unpack_from("<I", clx, offset + 1)[0]
    pieces = clx[offset + 5:offset + 5 + size]
    if len(pieces) != size or size < 4 or (size - 4) % 12:
        raise MalformedError("Invalid Word text piece table size")
    count = (size - 4) // 12
    cps = struct.unpack_from("<" + "I" * (count + 1), pieces)
    if any(right < left for left, right in zip(cps, cps[1:])) or cps[0] != 0:
        raise MalformedError("Word character positions are not ordered")
    output = []
    for index in range(count):
        if cps[index] >= text_chars:
            break
        characters = min(cps[index + 1], text_chars) - cps[index]
        encoded_fc = struct.unpack_from("<I", pieces, (count + 1) * 4 + index * 8 + 2)[0]
        compressed = bool(encoded_fc & 0x40000000)
        start = (encoded_fc & 0x3FFFFFFF) // (2 if compressed else 1)
        end = start + characters * (1 if compressed else 2)
        if end > len(word):
            raise MalformedError("Word text piece exceeds its document stream")
        output.append(word[start:end].decode("cp1252" if compressed else "utf-16-le", "replace"))
    return _clean("".join(output))


def _powerpoint(archive):
    data = archive.openstream("PowerPoint Document").read()
    output, records = [], 0
    def read(start, end, depth):
        nonlocal records
        if depth > 64:
            raise ResourceLimitError("PowerPoint container nesting exceeds the limit")
        offset = start
        while offset < end:
            if end - offset < 8:
                raise MalformedError("Truncated PowerPoint record header")
            flags, kind, size = struct.unpack_from("<HHI", data, offset)
            records += 1
            if records > 1_000_000 or size > end - offset - 8:
                raise ResourceLimitError("PowerPoint record exceeds the input bounds")
            body, stop = offset + 8, offset + 8 + size
            if kind == 0x2F14:
                raise EncryptedError("PowerPoint contains an encryption session")
            if flags & 15 == 15:
                read(body, stop, depth + 1)
            elif kind in (0x0FA0, 0x0FA8):
                if kind == 0x0FA0 and size % 2:
                    raise MalformedError("PowerPoint Unicode atom has an odd size")
                output.append(_clean(data[body:stop].decode("utf-16-le" if kind == 0x0FA0 else "cp1252", "replace")))
            offset = stop
    read(0, len(data), 0)
    return "\n\n".join(output)


def _xls(data):
    import xlrd
    workbook = xlrd.open_workbook(file_contents=data, on_demand=True)
    try:
        output = []
        for sheet in workbook.sheets():
            if sheet.visibility:
                continue
            if sheet.nrows > 5000 or sheet.ncols > 256:
                raise ResourceLimitError("Binary Excel sheet exceeds the extraction limits")
            output.append("## " + sheet.name)
            for index in range(sheet.nrows):
                values = []
                for cell in sheet.row(index):
                    if cell.ctype == xlrd.XL_CELL_DATE:
                        value = xlrd.xldate.xldate_as_datetime(cell.value, workbook.datemode).isoformat()
                    elif cell.ctype == xlrd.XL_CELL_BOOLEAN:
                        value = "TRUE" if cell.value else "FALSE"
                    elif cell.ctype == xlrd.XL_CELL_NUMBER:
                        value = str(int(cell.value)) if cell.value == int(cell.value) else str(cell.value)
                    elif cell.ctype == xlrd.XL_CELL_ERROR:
                        value = xlrd.error_text_from_code.get(cell.value, "#ERROR")
                    else:
                        value = str(cell.value)
                    values.append(value)
                output.append("\t".join(values))
        return "\n".join(output)
    finally:
        workbook.release_resources()


def _xlsb(data):
    import pyxlsb
    with open_zip(data):
        pass
    output = []
    with pyxlsb.open_workbook(io.BytesIO(data)) as workbook:
        for name in workbook.sheets:
            output.append("## " + name)
            with workbook.get_sheet(name) as sheet:
                for index, row in enumerate(sheet.rows()):
                    if index >= 5000 or len(row) > 256:
                        raise ResourceLimitError("Binary Excel sheet exceeds the extraction limits")
                    output.append("\t".join("" if cell.v is None else str(cell.v) for cell in row))
    return "\n".join(output)


def extract(data, kind):
    if kind in ("xls", "xlsb"):
        return _xls(data) if kind == "xls" else _xlsb(data)
    import olefile
    with olefile.OleFileIO(io.BytesIO(data), raise_defects=olefile.DEFECT_INCORRECT) as archive:
        for stream in archive.listdir(streams=True, storages=False):
            if archive.get_size(stream) > MAX_EXPANDED_BYTES:
                raise ResourceLimitError("Compound document stream exceeds the extraction limit")
        return _word(archive) if kind == "doc" else _powerpoint(archive)
