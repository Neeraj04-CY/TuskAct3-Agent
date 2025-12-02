"""Helpers for screenshot naming and placeholder rendering."""

from __future__ import annotations

from typing import Optional


def generate_screenshot_name(prefix: str, index: int) -> str:
    """Return a deterministic screenshot filename."""

    return f"{prefix}_{index:03d}.png"


def empty_screenshot(width: int = 1280, height: int = 720) -> bytes:
    """Return a placeholder PNG header when real screenshots are unavailable."""

    # Minimal PNG header for a transparent image. This is intentionally tiny.
    return bytes.fromhex(
        "89504E470D0A1A0A0000000D4948445200000001000000010806000000" "1F15C4890000000A49444154789C6360000002000154A24F5D0000000049454E44AE426082"
    )
