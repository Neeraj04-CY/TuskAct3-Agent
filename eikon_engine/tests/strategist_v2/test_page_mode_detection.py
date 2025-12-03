from __future__ import annotations

from eikon_engine.strategist.state_detector import detect_page_mode

LOGIN_HTML = """
<form>
  <input type="email" name="username" />
  <input type="password" name="password" />
  <button>Log In</button>
</form>
"""

DASHBOARD_HTML = """
<div class="nav-bar"></div>
<h1>Team Dashboard</h1>
<button class="card">Open App</button>
<button class="card">Reports</button>
"""

COOKIE_HTML = """
<div class="cookie-banner">
  <p>We use cookies to personalize content</p>
  <button class="accept">Accept All</button>
</div>
"""


def test_detects_login_page() -> None:
    assert detect_page_mode(LOGIN_HTML, "https://example.com/login") == "login_page"


def test_detects_dashboard_page() -> None:
    assert detect_page_mode(DASHBOARD_HTML, "https://example.com/app") == "dashboard_page"


def test_detects_cookie_popup() -> None:
    assert detect_page_mode(COOKIE_HTML, "https://example.com") == "cookie_popup"
