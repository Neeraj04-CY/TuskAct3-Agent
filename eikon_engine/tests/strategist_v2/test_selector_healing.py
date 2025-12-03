import pytest

from eikon_engine.strategist.selector_healing import HealingEntry, heal_selector


def make_el(selector: str, tag: str, text: str, clickable: bool = True, role: str = ""):
    return {"selector": selector, "tag": tag, "text": text, "clickable": clickable, "role": role}


def test_fuzzy_text_match():
    dom = [
        make_el("#btn1", "button", "Sign In"),
        make_el("#btn2", "button", "Register"),
    ]
    entry = heal_selector(dom, "#broken", {"text": "Sign-In"})
    assert isinstance(entry, HealingEntry)
    assert entry.selector == "#btn1"
    assert entry.reason == "fuzzy_text_match"
    assert entry.confidence > 0.55


def test_role_fallback_pick_button():
    dom = [
        make_el("#submit", "button", "Go", clickable=True, role="button"),
        make_el("#other", "div", "Stuff", clickable=False),
    ]
    entry = heal_selector(dom, "#missing", {"role": "button"})
    assert isinstance(entry, HealingEntry)
    assert entry.selector == "#submit"
    assert entry.reason == "role_fallback"


def test_downgrade_fallback_returns_tag_candidate():
    dom = [
        make_el("#s1", "button", "Proceed", clickable=True),
        make_el("#s2", "a", "Proceed Link", clickable=True),
    ]
    # broken selector looks like "button#submit-123"
    entry = heal_selector(dom, "button#submit-123", None)
    assert isinstance(entry, HealingEntry)
    assert entry.selector in {"#s1", "#s2"}
    assert entry.reason in {"downgrade_fallback", "first_clickable_fallback", "nearest_clickable_fallback"}


def test_nearest_clickable_when_no_text():
    dom = [
        make_el("#a1", "button", "Proceed", clickable=True),
        make_el("#a2", "button", "Abort", clickable=True),
    ]
    entry = heal_selector(dom, "#missing", None)
    assert isinstance(entry, HealingEntry)
    assert entry.selector in {"#a1", "#a2"}
    assert entry.reason.startswith("first_clickable") or entry.reason.startswith("nearest_clickable")


def test_no_clickables_returns_none():
    dom = [
        {"selector": "#x", "tag": "div", "text": "hi", "clickable": False},
    ]
    entry = heal_selector(dom, "#missing", {"text": "something"})
    assert entry is None
