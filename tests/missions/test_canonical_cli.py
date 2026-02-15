from __future__ import annotations

import json
from pathlib import Path

import pytest

from eikon_engine.missions import cli


def test_load_canonical_spec_reads_manifest(tmp_path: Path) -> None:
    manifest = tmp_path / "canonical.json"
    manifest.write_text(json.dumps({"missions": [{"slug": "demo", "instruction": "Run"}]}), encoding="utf-8")

    entry = cli._load_canonical_spec("demo", manifest)

    assert entry["instruction"] == "Run"


def test_parse_args_requires_mission_or_canonical() -> None:
    with pytest.raises(SystemExit):
        cli.parse_args([])

    args = cli.parse_args(["--canonical", "demo"])
    assert args.canonical == "demo"
