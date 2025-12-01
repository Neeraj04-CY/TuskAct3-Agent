from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional, Set


_DEFAULT_DANGEROUS_PATTERNS = (
    "rm -rf",
    "delete",
    "format drive",
    "shutdown",
    "poweroff",
)


@dataclass(slots=True)
class ToolPolicyDecision:
    allowed: bool
    reason: Optional[str] = None


class ToolPolicy:
    """Simple allow/deny policy for task execution."""

    def __init__(
        self,
        allowed_workers: Iterable[str] | None = None,
        max_iterations: int = 20,
        max_tokens: int = 4096,
        dangerous_patterns: Iterable[str] | None = None,
    ) -> None:
        self.allowed_workers: Optional[Set[str]] = set(allowed_workers or []) or None
        self.max_iterations = max_iterations
        self.max_tokens = max_tokens
        self.dangerous_patterns = tuple(dangerous_patterns or _DEFAULT_DANGEROUS_PATTERNS)

    def evaluate(
        self,
        step: Dict[str, Any],
        *,
        iteration: int = 0,
        token_count: Optional[int] = None,
    ) -> ToolPolicyDecision:
        worker = step.get("worker")
        description = str(step.get("description", ""))
        command = str(step.get("command", ""))

        if self.allowed_workers is not None and worker not in self.allowed_workers:
            return ToolPolicyDecision(False, f"Worker '{worker}' is not allowlisted")

        if iteration >= self.max_iterations:
            return ToolPolicyDecision(False, "Iteration limit exceeded")

        if token_count is not None and token_count > self.max_tokens:
            return ToolPolicyDecision(False, "Token budget exceeded")

        lower_blob = f"{description} {command}".lower()
        if any(pattern in lower_blob for pattern in self.dangerous_patterns):
            return ToolPolicyDecision(False, "Dangerous command detected")

        return ToolPolicyDecision(True, None)
