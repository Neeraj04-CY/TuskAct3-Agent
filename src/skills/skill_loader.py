from __future__ import annotations

import importlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class SkillMetadata:
    name: str
    source: str
    description: str
    version: str
    rate_limit: Optional[str]
    auth_methods: List[str]
    endpoints: List[Dict[str, Any]]
    example_usage: str


class SkillLoader:
    """
    Loads skill metadata and, optionally, executable skill modules.

    v1: only metadata parsing from configs/skills.json and simple Python modules.
    """

    def __init__(self, skills_config_path: str = "configs/skills.json") -> None:
        self._skills_config_path = Path(skills_config_path)
        self._skills: Dict[str, SkillMetadata] = {}
        self._load_metadata()

    def _load_metadata(self) -> None:
        if not self._skills_config_path.exists():
            return

        data = json.loads(self._skills_config_path.read_text(encoding="utf-8"))
        for item in data.get("skills", []):
            meta = SkillMetadata(
                name=item.get("name", ""),
                source=item.get("source", ""),
                description=item.get("description", ""),
                version=item.get("version", "0.1.0"),
                rate_limit=item.get("rate_limit"),
                auth_methods=item.get("auth_methods", []),
                endpoints=item.get("endpoints", []),
                example_usage=item.get("example_usage", ""),
            )
            self._skills[meta.name] = meta

    def list_skills(self) -> List[SkillMetadata]:
        return list(self._skills.values())

    def get_skill(self, name: str) -> Optional[SkillMetadata]:
        return self._skills.get(name)

    def import_skill_module(self, dotted_path: str) -> Any:
        """
        Dynamically import a Python module that implements a skill.
        """
        return importlib.import_module(dotted_path)