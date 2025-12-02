"""Utilities for loading engine configuration files."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml

DEFAULT_SETTINGS_PATH = Path(__file__).resolve().parents[1] / "config" / "settings.yaml"


def load_settings(path: Path | None = None) -> Dict[str, Any]:
    """Return parsed settings YAML as a dictionary."""

    file_path = path or DEFAULT_SETTINGS_PATH
    if not file_path.exists():
        raise FileNotFoundError(f"Missing settings file at {file_path}")
    with file_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    return data
