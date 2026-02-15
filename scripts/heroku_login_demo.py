from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Final

from playwright.async_api import Page, TimeoutError as PlaywrightTimeoutError, async_playwright

FAST_DEMO_MODE: Final[bool] = True
LOGIN_URL: Final[str] = "https://practicetestautomation.com/practice-test-login/"
USERNAME: Final[str] = "student"
PASSWORD: Final[str] = "Password123"
FLASH_SUCCESS_SELECTOR: Final[str] = "h1.post-title"
SUCCESS_TEXT: Final[str] = "Logged In Successfully"
ARTIFACTS_DIR: Final[Path] = Path("artifacts_demo")
SCREENSHOT_NAME: Final[str] = "secure_area.png"

NAV_TIMEOUT_MS: Final[int] = 12000 if FAST_DEMO_MODE else 20000
FIELD_TIMEOUT_MS: Final[int] = 8000 if FAST_DEMO_MODE else 8000
SECURE_TIMEOUT_MS: Final[int] = 8000 if FAST_DEMO_MODE else 7000
NAV_ATTEMPTS: Final[int] = 1 if FAST_DEMO_MODE else 3
WAIT_UNTIL_STATE: Final[str] = "commit" if FAST_DEMO_MODE else "domcontentloaded"


class DemoStepError(RuntimeError):
    """Raised when the fast demo fails at a known step."""


def _fail(step: str, reason: str) -> DemoStepError:
    message = f"Login demo failed at {step} ({reason})"
    _log(message)
    return DemoStepError(message)


def _log(message: str) -> None:
    print(f"[DEMO] {message}")


def _prepare_screenshot_path() -> Path:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    return ARTIFACTS_DIR / SCREENSHOT_NAME


async def _navigate_to_login(page: Page) -> None:
    attempts = NAV_ATTEMPTS
    for attempt in range(1, attempts + 1):
        try:
            _log(f"Navigating to {LOGIN_URL} ...")
            await page.goto(LOGIN_URL, wait_until=WAIT_UNTIL_STATE, timeout=NAV_TIMEOUT_MS)
        except PlaywrightTimeoutError as exc:
            if FAST_DEMO_MODE or attempt == attempts:
                raise _fail("navigation", "timeout") from exc
            _log(f"Navigation attempt {attempt} timed out; retrying...")
            continue

        try:
            await page.wait_for_selector("input#username", timeout=FIELD_TIMEOUT_MS)
            return
        except PlaywrightTimeoutError as exc:
            raise _fail("username field", "timeout") from exc


async def _run_login_flow(page: Page, screenshot_path: Path) -> None:
    await _navigate_to_login(page)

    _log("Filling username")
    try:
        await page.fill("input#username", USERNAME)
    except PlaywrightTimeoutError as exc:
        raise _fail("username field", "timeout") from exc

    _log("Filling password")
    try:
        await page.wait_for_selector("input#password", timeout=FIELD_TIMEOUT_MS)
        await page.fill("input#password", PASSWORD)
    except PlaywrightTimeoutError as exc:
        raise _fail("password field", "timeout") from exc

    _log("Submitting login form")
    try:
        await page.wait_for_selector("button#submit", timeout=FIELD_TIMEOUT_MS)
        await page.click("button#submit")
    except PlaywrightTimeoutError as exc:
        raise _fail("login button", "timeout") from exc

    _log("Waiting for success text...")
    try:
        await page.wait_for_selector(FLASH_SUCCESS_SELECTOR, timeout=SECURE_TIMEOUT_MS)
    except PlaywrightTimeoutError as exc:
        raise _fail("success text", "timeout") from exc

    flash_text = (await page.text_content(FLASH_SUCCESS_SELECTOR) or "").strip()
    current_url = page.url
    if SUCCESS_TEXT.lower() not in flash_text.lower() or "logged-in-successfully" not in current_url:
        raise RuntimeError("Login success verification failed")

    _log("Capturing secure area screenshot")
    await page.screenshot(path=str(screenshot_path), full_page=True)


async def _await_manual_close(prompt: str) -> None:
    await asyncio.to_thread(input, prompt)


async def run_demo() -> None:
    screenshot_path = _prepare_screenshot_path()
    error: Exception | None = None
    async with async_playwright() as playwright:
        _log("Launching Chromium (headful)...")
        browser = await playwright.chromium.launch(
            headless=False,
            args=[
                "--disable-gpu",
                "--disable-dev-shm-usage",
                "--no-sandbox",
            ],
        )
        context = await browser.new_context()
        page = await context.new_page()
        try:
            await _run_login_flow(page, screenshot_path)
            _log("✅ Heroku login demo completed.")
            _log(f"📸 Screenshot saved to {screenshot_path}")
        except DemoStepError as exc:
            error = exc
        except (PlaywrightTimeoutError, RuntimeError) as exc:
            error = exc
            _log(f"Login demo failed: {exc}")
        except Exception as exc:  # noqa: BLE001 - surface unexpected issues
            error = exc
            _log(f"Unexpected failure: {exc}")
        finally:
            await _await_manual_close("[DEMO] Press Enter to close the browser...")
            await context.close()
            await browser.close()
    if error:
        raise error


def main() -> None:
    asyncio.run(run_demo())


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: BLE001 - CLI entry point reporting
        print(f"Mission demo failed: {exc}", file=sys.stderr)
        sys.exit(1)
