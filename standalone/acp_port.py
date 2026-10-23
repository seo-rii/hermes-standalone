"""Reproduce the private ACP transport from its pinned, uninstalled wheel.

Run with --wheel PATH. This tool does not fetch inputs, execute build scripts
or install packages. Protocol/schema sources retain their upstream originals;
only explicit validation imports and the private import namespace change.
"""

import argparse
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import re
import zipfile


PIN = {
    "name": "agent-client-protocol", "version": "0.9.0",
    "filename": "agent_client_protocol-0.9.0-py3-none-any.whl",
    "sha256": "06911500b51d8cb69112544e2be01fc5e7db39ef88fecbc3848c5c6f194798ee",
    "url": "https://files.pythonhosted.org/packages/8f/ed/c284543c08aa443a4ef2c8bd120be51da8433dd174c01749b5d87c333f22/agent_client_protocol-0.9.0-py3-none-any.whl",
}
PRIVATE_PACKAGE = "acp_adapter._protocol"


def _port(data):
    text = data.decode("utf-8")
    text = re.sub(r"(?m)^(\s*from) acp(?=[ .])", r"\1 " + PRIVATE_PACKAGE, text)
    text = text.replace('"acp.core:', '"' + PRIVATE_PACKAGE + '.core:')
    lines = []
    for line in text.splitlines(keepends=True):
        if line.startswith("from pydantic import "):
            names = line[len("from pydantic import "):].strip().split(", ")
            rewritten = []
            for name in names:
                if name == "BaseModel":
                    rewritten.append("RequestModel as BaseModel")
                elif name == "BaseModel as _BaseModel":
                    rewritten.append("RequestModel as _BaseModel")
                elif name in {"Field", "RootModel", "ConfigDict", "ValidationError"}:
                    rewritten.append(name)
                else:
                    raise ValueError("Unreviewed ACP validation import: " + name)
            line = "from standalone.web_schema import " + ", ".join(rewritten) + "\n"
        lines.append(line)
    result = "".join(lines).encode("utf-8")
    if result != data:
        result = (b"# Modified for Hermes standalone: private imports and stdlib validation.\n"
                  b"# Exact upstream sources and change hashes are retained in PROVENANCE.json.\n" + result)
    return result


def port(wheel, destination):
    data = Path(wheel).read_bytes()
    if hashlib.sha256(data).hexdigest() != PIN["sha256"]:
        raise ValueError("ACP wheel does not match the reviewed source pin")
    destination = Path(destination)
    expected = {}
    sources = []
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        for name in sorted(archive.namelist()):
            path = PurePosixPath(name)
            if path.is_absolute() or ".." in path.parts:
                raise ValueError("Unsafe ACP archive path")
            if name.startswith("acp/") and (name.endswith(".py") or name.endswith("py.typed")):
                source = archive.read(name)
                relative = name[len("acp/"):]
                result = _port(source) if name.endswith(".py") else source
                expected[relative] = result
                record = {"upstream": name, "path": relative,
                          "upstream_sha256": hashlib.sha256(source).hexdigest(),
                          "sha256": hashlib.sha256(result).hexdigest()}
                if result != source:
                    record["retained_original"] = "upstream_sources/" + relative + ".original"
                    expected[record["retained_original"]] = source
                sources.append(record)
        metadata = "agent_client_protocol-0.9.0.dist-info/"
        for source, target in (("METADATA", "METADATA.upstream"),
                               ("RECORD", "RECORD.upstream"), ("licenses/LICENSE", "LICENSE")):
            expected[target] = archive.read(metadata + source)
    expected["PROVENANCE.json"] = (json.dumps({
        "format_version": 1, "source": PIN, "private_package": PRIVATE_PACKAGE,
        "changes": ["Relocate absolute ACP imports to the private package namespace.",
                    "Replace explicit Pydantic validation imports with standalone.web_schema."],
        "files": sources,
    }, sort_keys=True, indent=2) + "\n").encode()
    if destination.exists():
        existing = {path.relative_to(destination).as_posix(): path.read_bytes()
                    for path in destination.rglob("*") if path.is_file() and "__pycache__" not in path.parts}
        if existing != expected:
            raise ValueError("Existing private ACP sources differ; inspect before replacing")
    else:
        for name, content in expected.items():
            target = destination / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
    return len(sources)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheel", type=Path, required=True)
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).resolve().parents[1] / "acp_adapter/_protocol")
    args = parser.parse_args()
    print(f"Verified {port(args.wheel, args.output)} private ACP source files")


if __name__ == "__main__":
    main()
