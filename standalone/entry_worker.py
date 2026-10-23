"""Re-enter an allowlisted original module with the launcher's profile fallback."""

import argparse
import os
from pathlib import Path
import runpy
import sys


def main() -> None:
    from standalone.runtime import current_bundle, is_active_bundle

    source = Path(__file__).resolve().parents[1]
    if not is_active_bundle(source):
        raise SystemExit("A bundled entry worker requires verified runtime initialization")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--module", required=True)
    parser.add_argument("--home", required=True)
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.module not in current_bundle()["manifest"]["internal_modules"] or args.module == "standalone.entry_worker":
        parser.error("target module is not in the bundled allowlist")
    os.environ["HERMES_HOME"] = os.environ.get("HERMES_HOME") or args.home
    arguments = args.arguments[1:] if args.arguments[:1] == ["--"] else args.arguments
    sys.argv = [args.module, *arguments]
    runpy.run_module(args.module, run_name="__main__", alter_sys=True)


if __name__ == "__main__":
    main()
