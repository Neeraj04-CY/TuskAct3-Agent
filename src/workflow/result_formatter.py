from __future__ import annotations

from typing import Any, Dict, List

from src.common_types import WorkerStepResult


class ResultFormatter:
    """
    Format Worker results into human-friendly or machine-friendly outputs.
    """

    def to_text(self, results: List[WorkerStepResult]) -> str:
        lines: List[str] = []
        for res in results:
            status_symbol = "✅" if res.status == "success" else "❌"
            lines.append(f"{status_symbol} {res.step} [{res.status}] - retries: {res.retry_count}")
        return "\n".join(lines)

    def to_dict(self, results: List[WorkerStepResult]) -> Dict[str, Any]:
        return {
            "steps": [res.__dict__ for res in results]
        }