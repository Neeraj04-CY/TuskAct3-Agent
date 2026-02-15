from eikon_engine.page_intent import PageIntent, PageIntentResult
from eikon_engine.strategist.strategist_v2 import StrategistV2


class PlannerStub:
    async def create_plan(self, goal: str, last_result=None):
        return {"tasks": []}


def _make_strategist(goal: str = "") -> StrategistV2:
    strategist = StrategistV2(planner=PlannerStub())
    strategist._goal = goal or "Investigate startup listings"
    return strategist


def _step_meta(action: str) -> dict:
    action_payload = {"action": action}
    return {"step_id": f"{action}-1", "action_payload": action_payload, "action": action}


def test_listing_intent_blocks_dom_presence_and_requests_skill() -> None:
    strategist = _make_strategist(goal="Find a startup listing")
    run_ctx: dict = {}
    intent = PageIntentResult(intent=PageIntent.LISTING_PAGE, confidence=0.74, signals={"card_repetition": 3})
    strategist._handle_page_intent(run_ctx, _step_meta("navigate"), intent, action_label="navigate")

    assert run_ctx["dom_presence_blocked"] is True
    assert run_ctx["page_intents"][0]["intent"] == PageIntent.LISTING_PAGE.value
    assert run_ctx["requested_skills"][0]["name"] == "listing_extraction_skill"


def test_login_intent_clears_dom_presence_block() -> None:
    strategist = _make_strategist()
    run_ctx = {"dom_presence_blocked": True}
    intent = PageIntentResult(intent=PageIntent.LOGIN_FORM, confidence=0.81, signals={})
    strategist._handle_page_intent(run_ctx, _step_meta("navigate"), intent, action_label="navigate")

    assert run_ctx["dom_presence_blocked"] is False


def test_dom_presence_step_is_skipped_when_blocked() -> None:
    strategist = _make_strategist()
    run_ctx = {"dom_presence_blocked": True}
    planned_step = _step_meta("dom_presence_check")

    assert strategist.should_skip_step(run_ctx, planned_step) is True
