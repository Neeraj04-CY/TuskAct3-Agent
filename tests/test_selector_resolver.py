import pytest

from eikon_engine.browser.selector_resolver import SelectorResolver


def test_selector_resolver_prioritizes_username_input_by_keywords():
    html = """
    <form>
        <label for="primary">Email Address</label>
        <input id="primary" name="identifier" type="email" placeholder="Email" />
        <input id="secondary" name="search" type="text" placeholder="Search" />
    </form>
    """
    resolver = SelectorResolver(html)

    candidates = resolver.resolve_username()

    assert candidates, "expected at least one username candidate"
    assert candidates[0].selector == "#primary"
    assert candidates[0].score >= candidates[-1].score


def test_selector_resolver_prefers_password_inputs():
    html = """
    <form>
        <input id="pass-field" name="passwd" type="password" />
        <input id="text-field" name="note" type="text" />
    </form>
    """
    resolver = SelectorResolver(html)

    candidates = resolver.resolve_password()

    assert candidates[0].selector == "#pass-field"
    assert candidates[0].score > 0


def test_selector_resolver_scores_login_buttons_by_text():
    html = """
    <div>
        <button id="confirm" type="button">Cancel</button>
        <button id="login-btn" type="submit">Sign In</button>
    </div>
    """
    resolver = SelectorResolver(html)

    candidates = resolver.resolve_login_button()

    assert candidates[0].selector == "#login-btn"
    assert candidates[0].score >= 3.5


def test_selector_resolver_returns_fallback_when_missing_buttons():
    html = "<div><p>No forms here</p></div>"
    resolver = SelectorResolver(html)

    candidates = resolver.resolve_login_button()

    assert candidates[0].selector == "button[type=submit]"
    assert candidates[0].score == pytest.approx(0.2)


def test_selector_resolver_injects_login_bundle_for_heroku_demo():
    resolver = SelectorResolver(
        "<html></html>",
        current_url="https://the-internet.herokuapp.com/login",
    )

    bundle = resolver.get_login_selector_bundle()

    assert "#username" in bundle.get("username", [])
    overrides = resolver.login_override_candidates()
    assert overrides, "expected login overrides for Heroku page"
    assert overrides[0].selector == "#username"


def test_selector_resolver_mission_text_triggers_login_bundle():
    resolver = SelectorResolver(
        "<html></html>",
        mission_text="Login to the dashboard",
        current_url="https://example.com",
    )

    assert resolver.has_login_context()
    assert "#password" in resolver.get_login_selector_bundle().get("password", [])
