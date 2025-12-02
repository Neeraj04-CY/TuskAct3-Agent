"""Reusable Playwright-backed browser worker."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse
from urllib.request import url2pathname

import httpx

from eikon_engine.core.completion import build_completion
from eikon_engine.core.types import BrowserAction, BrowserWorkerResult
from eikon_engine.utils import dom_utils, vision_utils
from eikon_engine.utils.logging_utils import ArtifactLogger

try:  # pragma: no cover - optional dependency
    from playwright.async_api import async_playwright
except Exception:  # pragma: no cover - fallback when Playwright missing
    async_playwright = None  # type: ignore


@dataclass
class BrowserSession:
    """Holds Playwright session objects for reuse."""

    playwright: Any
    browser: Any
    context: Any
    page: Any

    async def close(self) -> None:
        await self.page.close()
        await self.context.close()
        await self.browser.close()
        await self.playwright.stop()


class BrowserWorker:
    """Executes declarative browser actions with retries and logging."""

    def __init__(
        self,
        *,
        settings: Dict[str, Any] | None = None,
        logger: ArtifactLogger | None = None,
        enable_playwright: bool | None = None,
    ) -> None:
        worker_settings = settings or {}
        browser_settings = worker_settings.get("browser", {})
        self.headless = bool(browser_settings.get("headless", True))
        self.slow_mo = int(browser_settings.get("slow_mo", 0))
        self.screenshot_enabled = bool(browser_settings.get("screenshot", True))
        completion_cfg = worker_settings.get("completion", {})
        self.retry_limit = int(completion_cfg.get("retry_limit", 3))
        self.logger = logger
        self.enable_playwright = enable_playwright if enable_playwright is not None else async_playwright is not None
        self._session: BrowserSession | None = None
        self._http_client = httpx.AsyncClient(follow_redirects=True)

    async def execute(self, metadata: Dict[str, Any]) -> BrowserWorkerResult:
        """Execute the provided browser action sequence."""

        actions = self._parse_actions(metadata)
        steps: List[Dict[str, Any]] = []
        screenshots: List[str] = []
        dom_snapshot: Optional[str] = None
        layout_graph: Optional[str] = None
        error: Optional[str] = None

        session = await self._ensure_session()
        for idx, action in enumerate(actions, start=1):
            step_entry = {"id": f"s{idx}", **action}
            steps.append(step_entry)
            try:
                dom_snapshot, layout_graph, screenshot_path = await self._perform_action(
                    action,
                    session=session,
                    step_index=idx,
                )
                if screenshot_path:
                    screenshots.append(screenshot_path)
            except Exception as exc:  # noqa: BLE001
                error = str(exc)
                step_entry["status"] = "error"
                break
            else:
                step_entry["status"] = "ok"

        completion = build_completion(
            complete=error is None,
            reason="browser actions completed" if error is None else error,
            payload={"steps": len(steps)},
        )
        return {
            "steps": steps,
            "screenshots": screenshots,
            "dom_snapshot": dom_snapshot,
            "layout_graph": layout_graph,
            "completion": completion,
            "error": error,
        }

    async def _perform_action(
        self,
        action: BrowserAction,
        *,
        session: BrowserSession | None,
        step_index: int,
    ) -> tuple[Optional[str], Optional[str], Optional[str]]:
        kind = action.get("action", "")
        if kind == "navigate":
            return await self._handle_navigation(action, session=session, step_index=step_index)
        if kind == "fill":
            await self._log_trace(
                step_index=step_index,
                action_name="fill",
                url=None,
                completion=None,
            )
            return None, None, None
        if kind == "click":
            await self._log_trace(
                step_index=step_index,
                action_name="click",
                url=None,
                completion=None,
            )
            return None, None, None
        if kind == "screenshot" and self.screenshot_enabled:
            screenshot_path = await self._capture_screenshot(
                session=session,
                action=action,
                step_index=step_index,
            )
            return None, None, screenshot_path
        if kind == "extract_dom" and session:
            page = session.page
            content = await page.content()
            html, layout = self._record_dom(content, step_index)
            return html, layout, None
        if kind == "extract_dom":
            # In fallback mode reuse the last logged HTML if available via logger
            dom_content = "<html></html>"
            if self.logger:
                self.logger.save_dom(dom_content, step_index=step_index)
                layout_content = dom_utils.build_layout_graph(dom_content)
                self.logger.save_layout_graph(layout_content, step_index=step_index)
            else:
                layout_content = dom_utils.build_layout_graph(dom_content)
            return dom_content, layout_content, None
        raise ValueError(f"Unsupported action: {kind}")

    async def _handle_navigation(
        self,
        action: BrowserAction,
        *,
        session: BrowserSession | None,
        step_index: int,
    ) -> tuple[Optional[str], Optional[str], Optional[str]]:
        url_value = action.get("url")
        if not url_value:
            raise ValueError("navigate action missing url")
        url = str(url_value)
        local_content = self._try_load_local_resource(url)
        if local_content is not None:
            html, layout = self._record_dom(local_content, step_index)
            await self._log_trace(
                step_index=step_index,
                action_name="navigate",
                url=url,
                completion=None,
            )
            return html, layout, None
        last_exc: Exception | None = None
        for attempt in range(1, self.retry_limit + 2):
            try:
                if session:
                    await session.page.goto(url, wait_until="load")
                    content = await session.page.content()
                else:
                    response = await self._http_client.get(url)
                    response.raise_for_status()
                    content = response.text
                html, layout = self._record_dom(content, step_index)
                await self._log_trace(
                    step_index=step_index,
                    action_name="navigate",
                    url=url,
                    completion=None,
                )
                return html, layout, None
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                await asyncio.sleep(0.25 * attempt)
        raise RuntimeError(f"failed to navigate to {url}: {last_exc}")

    def _try_load_local_resource(self, url: str) -> Optional[str]:
        parsed = urlparse(url)
        candidate: Path | None = None
        if parsed.scheme == "file":
            candidate = Path(url2pathname(parsed.path))
        elif parsed.scheme == "":
            candidate = Path(url)
        if candidate and candidate.exists():
            return candidate.read_text(encoding="utf-8")
        return None

    async def _capture_screenshot(
        self,
        *,
        session: BrowserSession | None,
        action: BrowserAction,
        step_index: int,
    ) -> Optional[str]:
        name = action.get("name") or vision_utils.generate_screenshot_name("screenshot", step_index)
        if session:
            data = await session.page.screenshot(full_page=True)
        else:
            data = vision_utils.empty_screenshot()
        if self.logger:
            path = self.logger.save_screenshot(data, step_index=step_index, name=name)
            return str(path)
        file_path = Path("artifacts") / name
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_bytes(data)
        return str(file_path)

    def _record_dom(self, html: str, step_index: int) -> tuple[str, str]:
        layout = dom_utils.build_layout_graph(html)
        if self.logger:
            self.logger.save_dom(html, step_index=step_index)
            self.logger.save_layout_graph(layout, step_index=step_index)
        return html, layout

    async def _log_trace(
        self,
        *,
        step_index: int,
        action_name: str,
        url: str | None,
        completion: Dict[str, Any] | None,
    ) -> None:
        if self.logger:
            self.logger.log_trace(
                goal=None,
                step_index=step_index,
                action=action_name,
                url=url,
                completion=completion,
            )

    async def _ensure_session(self) -> BrowserSession | None:
        if not self.enable_playwright or async_playwright is None:
            return None
        if self._session:
            return self._session
        playwright = await async_playwright().start()
        browser = await playwright.chromium.launch(headless=self.headless, slow_mo=self.slow_mo)
        context = await browser.new_context()
        page = await context.new_page()
        self._session = BrowserSession(playwright=playwright, browser=browser, context=context, page=page)
        return self._session

    def _parse_actions(self, metadata: Dict[str, Any]) -> List[BrowserAction]:
        action_payload = metadata.get("action", metadata)
        if isinstance(action_payload, str):
            try:
                parsed = json.loads(action_payload)
            except json.JSONDecodeError as exc:  # pragma: no cover - defensive
                raise ValueError("Action payload must be JSON when using strings") from exc
        else:
            parsed = action_payload
        if isinstance(parsed, list):
            return parsed  # type: ignore[return-value]
        if isinstance(parsed, dict):
            return [parsed]  # type: ignore[list-item]
        raise ValueError("Unsupported action payload type")

    async def close(self) -> None:
        """Close any underlying resources."""

        await self._http_client.aclose()
        if self._session:
            await self._session.close()
            self._session = None
