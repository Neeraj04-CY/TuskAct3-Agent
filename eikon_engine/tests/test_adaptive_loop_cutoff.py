from __future__ import annotations

from eikon_engine.api import llm_repair
from eikon_engine.core.adaptive_controller import AdaptiveController


def test_adaptive_loop_cutoff(monkeypatch) -> None:
    controller = AdaptiveController(max_corrections=1)
    failure_report = {"step_id": "s1", "error": "timeout"}

    def fake_request(_report):
        return {"type": "navigate", "payload": {"url": "https://retry"}}

    monkeypatch.setattr(llm_repair, "request_llm_fix", fake_request)
    assert controller.should_call_llm(failure_report)
    delta = controller.propose_fix(failure_report)
    controller.apply_fix({"tasks": []}, delta)
    assert controller.corrections == 1
    assert controller.should_call_llm(failure_report) is False
