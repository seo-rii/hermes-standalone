"""Internal diagnostic worker used to verify a cold bundle process."""

import importlib.metadata
import json
import sys


def main() -> None:
    def offline(event, args):
        if event in ("socket.connect", "socket.getaddrinfo", "subprocess.Popen", "os.exec", "os.system"):
            raise RuntimeError(f"Diagnostic worker attempted an external operation: {event}")

    sys.addaudithook(offline)
    import hermes_bootstrap
    from hermes_yaml import safe_load
    from tools.ansi_strip import strip_ansi

    print(json.dumps({
        "isolated": bool(sys.flags.isolated),
        "site_disabled": bool(sys.flags.no_site),
        "yaml": safe_load("enabled: yes\n") == {"enabled": True},
        "ansi": strip_ansi("\x1b[31mworker\x1b[0m"),
        "httpx": importlib.metadata.version("httpx"),
        "upstream_bootstrap": bool(hermes_bootstrap._standalone_bundle),
        "pm_loaded": any(name == "pm" or name.startswith("pm.") for name in sys.modules),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
