from __future__ import annotations

from eikon_engine.core.adaptive_controller import AdaptiveController


def test_adaptive_loop_cutoff(monkeypatch) -> None:
    controller = AdaptiveController(budget=2)
    failure_report = {"step_id": "s1", "error": "timeout"}
    monkeypatch.setattr(controller, "_should_call_llm", lambda _: True)
    assert controller.should_call_llm(failure_report)
    controller.record_failure("timeout")
    controller.record_failure("timeout")
    assert controller.remaining_budget == 0
    assert not controller.should_call_llm(failure_report)
