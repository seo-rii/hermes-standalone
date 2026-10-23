"""Run the original session kernel inside the initialized standalone interpreter."""

import argparse
from pathlib import Path
import sys


def main():
    from standalone.runtime import is_active_bundle

    if not is_active_bundle(Path(__file__).resolve().parents[1]):
        raise SystemExit("The code kernel requires initialized bundle context")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--staging", required=True)
    args = parser.parse_args()
    staging = Path(args.staging).resolve(strict=True)
    if not staging.is_dir() or not (staging / "hermes_tools.py").is_file():
        parser.error("kernel staging must contain its generated tool RPC client")
    # Only the generated RPC client lives here. The runner itself always comes
    # from the verified application and retains its original cell authority.
    sys.path.insert(0, str(staging))
    from tools.code_kernel import KERNEL_RUNNER_SOURCE

    filename = str(staging / "hermes_kernel_runner.py")
    sys.argv = [filename]
    exec(compile(KERNEL_RUNNER_SOURCE, filename, "exec"), {"__name__": "__main__", "__file__": filename})


if __name__ == "__main__":
    main()
