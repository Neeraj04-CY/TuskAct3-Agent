import json
from pathlib import Path
from typing import List

from .memory_store import MissionMemory

MEMORY_DIR = Path("memory_logs")


def load_all_memories() -> List[MissionMemory]:
	memories = []
	if not MEMORY_DIR.exists():
		return memories

	for file in MEMORY_DIR.glob("*.json"):
		with open(file, "r", encoding="utf-8") as f:
			data = json.load(f)
			memories.append(MissionMemory(**data))

	return memories
