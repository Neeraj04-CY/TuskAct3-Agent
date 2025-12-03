from __future__ import annotations

from run_record_demo import build_record_dataset


def make_result() -> dict:
    return {
        "steps": [
            {
                "step": {
                    "step_id": "s1",
                    "task_id": "task_a",
                    "action_payload": {"action": "click", "selector": "#cta", "url": "https://site"},
                },
                "result": {
                    "completion": {"complete": True, "reason": "ok"},
                    "steps": [{"status": "blocked", "block_reason": "click_blocked_risky"}],
                    "failure_dom_path": "artifacts/dom.html",
                    "failure_screenshot_path": "artifacts/screenshot.png",
                },
            }
        ]
    }


def test_build_record_dataset_marks_blocked() -> None:
    dataset = build_record_dataset(make_result())
    assert len(dataset) == 1
    record = dataset[0]
    assert record["blocked"] is True
    assert record["dom_path"] == "artifacts/dom.html"
    assert record["screenshot"] == "artifacts/screenshot.png"


def test_build_record_dataset_inline_dom() -> None:
    sample = make_result()
    sample["steps"][0]["result"]["dom_snapshot"] = "<html></html>"
    dataset = build_record_dataset(sample, inline_dom=True)
    assert "dom_snapshot" in dataset[0]
    assert dataset[0]["dom_snapshot"].startswith("<html>")
