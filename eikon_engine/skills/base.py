from abc import ABC, abstractmethod
from typing import Any, Dict


class Skill(ABC):
    name: str

    @abstractmethod
    async def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        pass
