from __future__ import annotations


def detection_value(detection: dict[str, object], key: str, default: object = "-") -> object:
    """Read detector fields safely across app redeploys and older session state."""
    value = detection.get(key, default)
    return default if value is None else value
