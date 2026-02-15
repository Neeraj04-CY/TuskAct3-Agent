from eikon_engine.page_intent import PageIntent, classify_page_intent

LISTING_DOM = """
<html>
  <body>
    <section class="company-card card">
      <h2>Atlas Robotics</h2>
      <p>Robots for every warehouse.</p>
      <a href="/companies/atlas-robotics">Details</a>
    </section>
    <section class="company-card card">
      <h2>Zephyr Labs</h2>
      <p>AI generated food insights.</p>
      <a href="/companies/zephyr-labs">Details</a>
    </section>
  </body>
</html>
"""

LOGIN_DOM = """
<html>
  <body>
    <form>
      <input type="text" name="username" />
      <input type="password" name="password" />
      <button>Log In</button>
    </form>
  </body>
</html>
"""


def test_classifier_detects_listing_page() -> None:
    result = classify_page_intent(LISTING_DOM, url="https://www.ycombinator.com/companies")
    assert result.intent is PageIntent.LISTING_PAGE
    assert result.confidence >= 0.4
    assert result.signals["card_repetition"] >= 1


def test_classifier_detects_login_form() -> None:
    result = classify_page_intent(LOGIN_DOM, url="https://example.com/login")
    assert result.intent is PageIntent.LOGIN_FORM
    assert result.confidence >= 0.7
