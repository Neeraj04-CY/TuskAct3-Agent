from __future__ import annotations

import json
import os
import re
from contextlib import suppress
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.telemetry import Telemetry

try:  # pragma: no cover - optional dependency
    from playwright.async_api import (  # type: ignore
        BrowserContext,
        Error as PlaywrightError,
        Page,
        TimeoutError as PlaywrightTimeoutError,
        async_playwright,
    )
except ImportError:  # pragma: no cover - optional dependency
    PlaywrightError = RuntimeError  # type: ignore[assignment]
    PlaywrightTimeoutError = RuntimeError  # type: ignore[assignment]
    async_playwright = None  # type: ignore[assignment]
    BrowserContext = Any  # type: ignore[assignment]
    Page = Any  # type: ignore[assignment]


_ALLOW_EXTERNAL_ENV = "EIKON_ALLOW_EXTERNAL"
_ALLOW_SENSITIVE_ENV = "EIKON_ALLOW_SENSITIVE"
_BYPASS_DRY_RUN_ENV = "PLAYWRIGHT_BYPASS_DRY_RUN"


@dataclass(slots=True)
class BrowserAction:
    action: str
    url: Optional[str] = None
    selector: Optional[str] = None
    value: Optional[str] = None
    name: Optional[str] = None
    timeout: Optional[int] = None

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "BrowserAction":
        action = payload.get("action")
        if not action:
            raise ValueError("Browser action missing 'action' field")
        return cls(
            action=str(action).lower(),
            url=payload.get("url"),
            selector=payload.get("selector"),
            value=payload.get("value"),
            name=payload.get("name"),
            timeout=payload.get("timeout"),
        )


@dataclass
class BrowserSession:
    playwright: Any
    browser: Any
    context: Any
    page: Any

    async def close(self) -> None:
        if self.page:
            with suppress(Exception):
                await self.page.close()
        if self.context:
            with suppress(Exception):
                await self.context.close()
        if self.browser:
            with suppress(Exception):
                await self.browser.close()
        if self.playwright:
            with suppress(Exception):
                await self.playwright.stop()


class BrowserWorker:
    """Executes declarative browser steps via Playwright with dry-run safeguards."""

    session_key = "browser"

    def __init__(
        self,
        *,
        telemetry: Optional[Telemetry] = None,
        allow_external: Optional[bool] = None,
    ) -> None:
        self.telemetry = telemetry or Telemetry()
        env_allow = os.getenv(_ALLOW_EXTERNAL_ENV, "0") == "1"
        self.allow_external = allow_external if allow_external is not None else env_allow

    async def run(
        self,
        description: str,
        prev_results: Dict[str, Any],
        *,
        dry_run: bool = True,
        allow_sensitive: Optional[bool] = None,
        allow_external: Optional[bool] = None,
        session: Optional[BrowserSession] = None,
        reuse_session: bool = False,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        del prev_results  # unused but retained for interface compatibility
        actions = self._parse_actions(description)
        if not actions:
            return {
                "steps": [],
                "screenshots": [],
                "dom": None,
                "error": "No browser actions were parsed",
                "dry_run": dry_run,
                "completion": {"success": False, "complete": False, "reason": "No browser actions were parsed"},
            }

        allow_sensitive = (
            allow_sensitive
            if allow_sensitive is not None
            else (os.getenv(_ALLOW_SENSITIVE_ENV, "0") == "1")
        )
        manual_allow_external = allow_external if allow_external is not None else kwargs.get("allow_external")
        dry_run = dry_run and os.getenv(_BYPASS_DRY_RUN_ENV, "0") != "1"

        if not self._sensitive_allowed(actions, allow_sensitive, manual_allow_external):
            reason = (
                "External browsing is disabled. Set --allow-sensitive or "
                f"export {_ALLOW_SENSITIVE_ENV}=1 to enable."
            )
            return {
                "steps": [],
                "screenshots": [],
                "dom": None,
                "error": reason,
                "dry_run": True,
                "completion": {"success": False, "complete": False, "reason": reason},
            }

        completion_state = {"success": False, "complete": False, "reason": ""}
        result: Dict[str, Any] = {
            "steps": [],
            "screenshots": [],
            "dom": None,
            "error": None,
            "dry_run": dry_run,
            "completion": completion_state,
        }

        if dry_run:
            for idx, action in enumerate(actions, start=1):
                step = {
                    "id": f"s{idx}",
                    "action": action.action,
                    "status": "dry_run",
                    "selector": action.selector,
                    "url": action.url,
                }
                result["steps"].append(step)
            completion_state["reason"] = "Dry run only"
            return result

        if async_playwright is None:
            completion_state["reason"] = "Playwright missing"
            return {
                **result,
                "error": "Playwright is not installed. Run 'pip install playwright' and 'playwright install'.",
            }

        reuse_live_session = reuse_session and not dry_run
        managed_sessions: List[BrowserSession] = []
        live_session: Optional[BrowserSession] = session if reuse_live_session else None

        if not reuse_live_session:
            live_session = await self._create_session()
            managed_sessions.append(live_session)
        elif live_session is None:
            live_session = await self._create_session()

        if live_session is None:
            return {**result, "error": "Failed to initialize browser session."}

        run_root = self._prepare_run_dir()
        screenshot_dir = run_root / "screenshots"
        screenshot_dir.mkdir(parents=True, exist_ok=True)

        page = live_session.page

        try:
            for idx, action in enumerate(actions, start=1):
                step_id = f"s{idx}"
                step_entry = {
                    "id": step_id,
                    "action": action.action,
                    "status": "pending",
                    "selector": action.selector,
                    "url": action.url,
                }
                result["steps"].append(step_entry)
                await self._emit_event("browser_step_scheduled", step_entry)

                try:
                    await self._execute_action(
                        page,
                        action,
                        screenshot_dir,
                        result,
                    )
                    step_entry["status"] = "ok"
                    await self._emit_event("browser_step_completed", step_entry)
                except (PlaywrightError, PlaywrightTimeoutError) as exc:  # pragma: no cover - integration
                    step_entry["status"] = "error"
                    result["error"] = str(exc)
                    await self._emit_event(
                        "browser_error",
                        {"action": action.action, "reason": str(exc)},
                    )
                    break
        finally:
            for managed in managed_sessions:  # pragma: no cover - defensive cleanup
                await managed.close()

        if reuse_live_session and live_session:
            result["_session"] = live_session

        if result.get("error"):
            completion_state["reason"] = result["error"] or "Browser worker encountered an error"
        else:
            completion_state.update({
                "success": True,
                "complete": True,
                "reason": "Browser actions completed successfully",
            })

        return result

    def _parse_actions(self, description: str) -> List[BrowserAction]:
        description = (description or "").strip()
        if not description:
            return []

        try:
            payload = json.loads(description)
            if isinstance(payload, dict):
                payload = [payload]
            if isinstance(payload, list):
                return [BrowserAction.from_dict(item) for item in payload]
        except json.JSONDecodeError:
            pass

        actions: List[BrowserAction] = []
        lowered = description.lower()
        nav_match = re.search(r"navigate to (?P<url>\S+)", description, flags=re.IGNORECASE)
        if nav_match:
            actions.append(BrowserAction(action="navigate", url=nav_match.group("url")))
        fill_match = re.search(r"fill (?P<selector>#[^ ]+|\.[^ ]+|[\w-]+) with (?P<value>.+)", description, flags=re.IGNORECASE)
        if fill_match:
            selector = fill_match.group("selector")
            if not selector.startswith(("#", ".")):
                selector = f"#{selector}"
            actions.append(
                BrowserAction(
                    action="fill",
                    selector=selector,
                    value=fill_match.group("value").strip(),
                )
            )
        click_match = re.search(r"click (?:the )?(?P<label>[\w\s]+)", description, flags=re.IGNORECASE)
        if click_match:
            label = click_match.group("label").strip()
            actions.append(BrowserAction(action="click", selector=f"text={label}"))
        if "screenshot" in lowered:
            actions.append(BrowserAction(action="screenshot", name="step.png"))
        if "extract dom" in lowered:
            actions.append(BrowserAction(action="extract_dom"))
        return actions

    def _sensitive_allowed(
        self,
        actions: List[BrowserAction],
        allow_sensitive: bool,
        manual_allow_external: Optional[bool],
    ) -> bool:
        if allow_sensitive or manual_allow_external or self.allow_external:
            return True
        for action in actions:
            if action.url and action.url.startswith(("http://", "https://")):
                return False
        return True

    def _prepare_run_dir(self) -> Path:
        root = Path("runs") / datetime.now().strftime("%Y%m%d_%H%M%S")
        root.mkdir(parents=True, exist_ok=True)
        return root

    async def _create_session(self) -> BrowserSession:
        if async_playwright is None:  # pragma: no cover - defensive
            raise RuntimeError("Playwright is not available")
        playwright = await async_playwright().start()
        browser = await playwright.chromium.launch(headless=True)
        context = await browser.new_context()
        page = await context.new_page()
        return BrowserSession(playwright=playwright, browser=browser, context=context, page=page)

    async def _execute_action(
        self,
        page: Any,
        action: BrowserAction,
        screenshot_dir: Path,
        result: Dict[str, Any],
    ) -> None:
        if action.action == "navigate":
            if not action.url:
                raise ValueError("Navigate action missing url")
            await page.goto(action.url, wait_until="load")
        elif action.action == "fill":
            if not action.selector or action.value is None:
                raise ValueError("Fill action requires selector and value")
            await page.fill(action.selector, action.value)
        elif action.action == "click":
            if not action.selector:
                raise ValueError("Click action missing selector")
            await page.click(action.selector)
        elif action.action == "screenshot":
            name = action.name or f"step_{len(result['screenshots']) + 1}.png"
            path = screenshot_dir / name
            await page.screenshot(path=str(path))
            result["screenshots"].append(str(path))
        elif action.action == "extract_dom":
            result["dom"] = await page.content()
        elif action.action == "wait":
            if not action.selector:
                raise ValueError("Wait action missing selector")
            timeout = action.timeout or 5000
            await page.wait_for_selector(action.selector, timeout=timeout)
        else:
            raise ValueError(f"Unsupported browser action: {action.action}")

    async def _emit_event(self, event_type: str, payload: Dict[str, Any]) -> None:
        if self.telemetry:
            await self.telemetry.trace_event(event_type, payload)
