"""Feature flags for optional MC DTF Pro modules."""

ENABLE_VECTOR_EXPORT = False
ENABLE_AI_UPSCALE = False
ENABLE_OBJECT_REMOVAL = False
ENABLE_SUBLIMATION = False
ENABLE_DTF_UV = False
ENABLE_DESIGN_TOOLS = False
ENABLE_MOCKUPS = True
ENABLE_BATCH = True
ENABLE_BUSINESS_TOOLS = False
ENABLE_API = False


def is_enabled(flag_name: str) -> bool:
    """Return whether a feature flag is enabled."""
    return bool(globals().get(flag_name, False))


def enabled_flags() -> dict[str, bool]:
    """Return all public feature flags."""
    return {key: bool(value) for key, value in globals().items() if key.startswith("ENABLE_")}
