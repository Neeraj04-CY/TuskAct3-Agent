from __future__ import annotations

from eikon_engine.planning.memory_store import MemoryStore


def test_memory_store_short_and_long_term_behavior() -> None:
    store = MemoryStore()
    for idx in range(7):
        store.remember({"summary": f"step {idx}", "result": {"idx": idx}})

    assert len(store.short_term) == 5
    assert store.short_term[-1]["summary"] == "step 6"

    summary = store.summarize()
    assert summary
    assert store.long_term

    retrieved = store.retrieve("step 6")
    assert retrieved
    assert any("step" in record.text for record in retrieved)
