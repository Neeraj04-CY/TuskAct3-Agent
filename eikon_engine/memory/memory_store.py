from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional


@dataclass
class MissionMemory:
	mission_id: str
	mission_text: str
	url: Optional[str]
	status: str
	skills_used: List[str]
	artifacts_path: str
	timestamp: str
