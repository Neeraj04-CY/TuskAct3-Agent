"""Reflection worker that evaluates whether goals are satisfied."""

from __future__ import annotations

from typing import Any, Dict

from eikon_engine.core.completion import build_completion


class ReflectionWorker:
    """Tiny heuristic reflection worker used to stop loops early."""

    async def execute(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        """Return a completion payload when success heuristics are met."""

        dom = (observation.get("dom_snapshot") or "").lower()
        success = "secure area" in dom or "you logged into a secure area" in dom
        completion = build_completion(
            complete=success,
            reason="secure area detected" if success else "goal not yet satisfied",
            payload={"dom_length": len(dom)},
        )
        return {"completion": completion}
