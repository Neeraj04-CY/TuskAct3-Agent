from __future__ import annotations

from eikon_engine.strategist.page_intent import classify_page_intent

FORM_HTML = """
<form>
  <input type="text" name="first" />
  <input type="text" name="last" />
  <input type="email" name="email" />
  <button type="submit">Continue</button>
</form>
"""

SEARCH_HTML = """
<div>
  <input type="search" name="q" />
  <div class="results">
    <a href="/1">Search result 1</a>
    <a href="/2">Search result 2</a>
    <a href="/3">Search result 3</a>
    <a href="/4">Search result 4</a>
    <a href="/5">Search result 5</a>
  </div>
</div>
"""


def test_form_entry_intent_detected() -> None:
    intent = classify_page_intent(FORM_HTML)
    assert intent.intent == "form_entry"
    assert intent.confidence > 0.5


def test_search_results_intent_detected() -> None:
    intent = classify_page_intent(SEARCH_HTML)
    assert intent.intent == "search_results"
    assert intent.confidence > 0.5
