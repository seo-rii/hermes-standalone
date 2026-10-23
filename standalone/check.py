#!/usr/bin/env python3
"""Run standalone checks with the standard library, independently of upstream PM."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import unittest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, help="Also test this generated single-file distribution")
    parser.add_argument("--artifact-only", action="store_true", help="Run only generated-file acceptance checks")
    args = parser.parse_args()
    if args.artifact_only and not args.artifact:
        parser.error("--artifact-only requires --artifact")
    root = Path(__file__).resolve().parents[1]
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(root))
    if args.artifact:
        artifact = args.artifact.resolve()
        if not artifact.is_file():
            parser.error(f"Artifact does not exist: {artifact}")
        from standalone.tests import test_artifact
        test_artifact.ARTIFACT = artifact
    if args.artifact_only:
        suite = unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromName(name) for name in (
            "standalone.tests.test_artifact", "standalone.tests.test_artifact_chat",
        ))
    else:
        suite = unittest.defaultTestLoader.discover(
            str(root / "standalone" / "tests"), pattern="test_*.py", top_level_dir=str(root),
        )
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
