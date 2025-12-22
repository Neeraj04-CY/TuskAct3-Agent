import json
from pathlib import Path

from .memory_store import MissionMemory

MEMORY_DIR = Path("memory_logs")


def save_mission_memory(memory: MissionMemory) -> None:
	MEMORY_DIR.mkdir(exist_ok=True)
	file_path = MEMORY_DIR / f"{memory.mission_id}.json"

	with open(file_path, "w", encoding="utf-8") as f:
		json.dump(memory.__dict__, f, indent=2)
