"""The single-file distribution has an immutable dependency set."""

from pathlib import Path
import sys


# The pinned Hub SDK's original HTTP/LFS path replaces optional native Xet.
HUGGINGFACE_DISABLE_XET = True

# These model interpreters have no CPU/OS-specific native wheel requirement.
PORTABLE_EXTRAS = frozenset({"wake", "wake-openwakeword", "voice", "stt-whisper", "piper"})


def active() -> bool:
    runtime = sys.modules.get("standalone.runtime")
    return runtime is not None and runtime.is_active_bundle(Path(__file__).resolve().parents[1])


def replacement_anchors(extra: str):
    return {
        "anthropic": ("standalone.transport_messages",),
        "web": ("standalone.web", "starlette", "uvicorn", "multipart"),
        "computer-use": ("standalone.mcp_client",),
        "acp": ("acp_adapter._protocol",),
        "doc-extract": ("standalone.documents", "pypdf", "olefile", "xlrd", "pyxlsb", "striprtf"),
        "vision": ("standalone.raster", "pyjpeg"),
        "exa": ("standalone.web_providers",),
        "firecrawl": ("standalone.web_providers",),
        "parallel-web": ("standalone.web_providers",),
        "ddgs": ("standalone.web_ddgs",),
        "matrix": ("mautrix", "standalone.matrix_olm", "aiosqlite", "markdown", "aiohttp_socks"),
        "audio-io": ("standalone.audio_device",),
        "voice": ("standalone.local_whisper", "standalone.audio_device"),
        "stt-whisper": ("standalone.local_whisper",),
        "piper": ("standalone.local_piper_voice",),
        "silk": ("standalone.silk",),
        "wake": ("standalone.wake_openword",),
        "wake-openwakeword": ("standalone.wake_openword",),
        "dingtalk": ("dingtalk_stream",),
        "feishu": ("lark_oapi",),
        "azure-identity": ("azure.identity",),
        "teams": ("standalone.platform_teams",),
        "modal": ("modal",),
        "vercel": ("standalone.cloud_vercel", "vercel_oidc"),
        "daytona": ("standalone.cloud_daytona",),
        "google-chat": ("standalone.google_pubsub", "google.auth", "googleapiclient", "google_auth_httplib2"),
        "trace-upload": ("huggingface_hub",),
        "mistral": ("standalone.transport_vendor_media",),
        "tts-premium": ("standalone.transport_vendor_media",),
        "google-meet": ("standalone.meet_browser", "websockets"),
    }.get(extra)


def refuse_install(package: str) -> None:
    from pm.package import InstallError

    raise InstallError(
        package,
        "This Hermes single-file distribution embeds its dependencies; runtime package installation is disabled",
        "rebuild or replace hermes.py with a distribution containing the required feature",
    )
