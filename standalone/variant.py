"""Feature availability belongs to the verified artifact, not a Hermes profile."""

from collections.abc import Mapping
from types import MappingProxyType


_EXCLUDED_FEATURES = frozenset({"local-piper-model", "google-static-discovery"})
_FULL = MappingProxyType({"name": "full", "excluded_features": ()})
_UNAVAILABLE = {
    "local-piper-model": (
        "Local Piper speech synthesis is not included in the lite build. "
        "Use the full build for local Piper, or select a cloud TTS provider."
    ),
    "google-static-discovery": (
        "The full Google static discovery catalog is not included in the lite build; "
        "Google Chat's document remains embedded. Use the full build for other static "
        "documents, or request dynamic discovery explicitly."
    ),
}


class UnavailableFeatureError(ImportError):
    """An optional feature was intentionally excluded from this artifact."""


def normalize_variant(value: object) -> Mapping[str, object]:
    """Validate and freeze metadata copied from an authenticated manifest."""
    if (
        not isinstance(value, Mapping)
        or not {"name", "excluded_features"}.issubset(value)
        or set(value) - {"name", "excluded_features", "generated_files"}
    ):
        raise ValueError("invalid standalone build variant")
    name = value["name"]
    excluded = value["excluded_features"]
    if (
        name not in ("full", "lite")
        or not isinstance(excluded, (list, tuple))
        or not all(isinstance(feature, str) for feature in excluded)
        or len(excluded) != len(set(excluded))
        or set(excluded) != (_EXCLUDED_FEATURES if name == "lite" else set())
    ):
        raise ValueError("invalid standalone build variant")
    result = {"name": name, "excluded_features": tuple(sorted(excluded))}
    if "generated_files" in value:
        generated = value["generated_files"]
        if (
            not isinstance(generated, (list, tuple))
            or tuple(generated) not in ((), ("app/standalone/capabilities.json",))
            or (name == "full" and generated)
        ):
            raise ValueError("invalid standalone generated variant files")
        result["generated_files"] = tuple(generated)
    return MappingProxyType(result)


def manifest_variant(manifest: Mapping[str, object]) -> Mapping[str, object]:
    """Older artifacts without variant metadata retain the full distribution."""
    return normalize_variant(manifest["build_variant"]) if "build_variant" in manifest else _FULL


def current_variant() -> Mapping[str, object]:
    from standalone.runtime import current_bundle

    context = current_bundle()
    return manifest_variant(context["manifest"]) if context is not None else _FULL


def unavailable_reason(feature: str) -> str | None:
    if feature in current_variant()["excluded_features"]:
        return _UNAVAILABLE[feature]
    return None


def excluded_feature_details(variant: Mapping[str, object]) -> dict[str, str]:
    return {feature: _UNAVAILABLE[feature] for feature in variant["excluded_features"]}


def require_feature(feature: str) -> None:
    reason = unavailable_reason(feature)
    if reason is not None:
        raise UnavailableFeatureError(reason)
