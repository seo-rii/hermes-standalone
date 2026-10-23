"""Run original profile maintenance inside this verified artifact's runtime."""

import json
from pathlib import Path


def main():
    from standalone.runtime import is_active_bundle

    if not is_active_bundle(Path(__file__).resolve().parents[1]):
        raise SystemExit("Profile maintenance requires verified standalone initialization")
    from tools.skills_sync import sync_skills

    print(json.dumps(sync_skills(quiet=True)), flush=True)


if __name__ == "__main__":
    main()
