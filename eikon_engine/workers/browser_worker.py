"""Reusable Playwright-backed browser worker."""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
from datetime import UTC, datetime
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Mapping
from urllib.parse import urlparse
from urllib.request import url2pathname
from time import perf_counter

import httpx

from eikon_engine.browser.selector_resolver import (
    BUTTON_TOKENS,
    PASSWORD_TOKENS,
    SelectorCandidate,
    SelectorResolver,
    USERNAME_TOKENS,
)
from eikon_engine.core.completion import build_completion
from eikon_engine.core.types import BrowserAction, BrowserWorkerResult
from eikon_engine.utils import dom_utils, vision_utils
from eikon_engine.utils.logging_utils import ArtifactLogger
from eikon_engine.utils.safety_guardrails import SafetyGuardrails

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
        try:
            await self.page.close()
        finally:
            try:
                await self.context.close()
            finally:
                try:
                    await self.browser.close()
                finally:
                    await self.playwright.stop()


@dataclass
class ActionResult:
    dom_snapshot: Optional[str] = None
    layout_graph: Optional[str] = None
    screenshot: Optional[str] = None
    status: str = "ok"
    error: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)
    payload: Optional[Dict[str, Any]] = None

    def failed(self) -> bool:
        return self.status != "ok"


class BrowserWorker:
    """Executes declarative browser actions with retries and logging."""

    def __init__(
        self,
        *,
        settings: Dict[str, Any] | None = None,
        logger: ArtifactLogger | None = None,
        enable_playwright: bool | None = None,
        show_browser: bool | None = None,
    ) -> None:
        worker_settings = settings or {}
        browser_settings = worker_settings.get("browser", {})
        resolved_headless = bool(browser_settings.get("headless", True))
        if show_browser is True:
            resolved_headless = False
        elif show_browser is False:
            resolved_headless = True
        self.headless = resolved_headless
        self.slow_mo = int(browser_settings.get("slow_mo", 0))
        self.screenshot_enabled = bool(browser_settings.get("screenshot", True))
        completion_cfg = worker_settings.get("completion", {})
        self.retry_limit = int(completion_cfg.get("retry_limit", 3))
        self.logger = logger
        self.enable_playwright = enable_playwright if enable_playwright is not None else async_playwright is not None
        self._session: BrowserSession | None = None
        self._http_client = httpx.AsyncClient(follow_redirects=True)
        guard_settings = worker_settings.get("guardrails", {})
        self.guardrails = SafetyGuardrails(guard_settings)
        self._trace_logger = logging.getLogger(__name__)
        self._last_dom_html: str = "<html></html>"
        self._mission_instruction: Optional[str] = None
        self._subgoal_description: Optional[str] = None
        self._current_goal: Optional[str] = None
        self._current_url: Optional[str] = None
        self._teardown_required = False
        self._shutdown_complete = False
        self._secure_area_info: Dict[str, Any] | None = None
        self._login_flow_state: Dict[str, bool] = {}

    async def execute(self, metadata: Dict[str, Any]) -> BrowserWorkerResult:
        """Execute the provided browser action sequence."""

        run_goal: Optional[str] = None
        if isinstance(metadata, dict):
            run_goal = metadata.get("goal") or metadata.get("description")
        self._current_goal = run_goal or self._subgoal_description or None
        self._current_url = None
        self._secure_area_info = None
        self._reset_login_flow_state()

        actions = self._parse_actions(metadata)
        steps: List[Dict[str, Any]] = []
        screenshots: List[str] = []
        dom_snapshot: Optional[str] = None
        layout_graph: Optional[str] = None
        dom_presence_events: List[Dict[str, Any]] = []
        error: Optional[str] = None
        failure_dom_path: Optional[str] = None
        failure_screenshot_path: Optional[str] = None

        session = await self._ensure_session()
        secure_area_detection: Dict[str, Any] | None = None
        current_url: Optional[str] = None
        for idx, action in enumerate(actions, start=1):
            action_kind = (action.get("action") or "").lower()
            step_entry = {"id": f"s{idx}", **action}
            steps.append(step_entry)
            allowed, block_reason = self.guardrails.check(action, current_url=current_url)
            if not allowed:
                step_entry["status"] = "blocked"
                step_entry["block_reason"] = block_reason
                continue
            try:
                action_result = await self._perform_action(
                    action,
                    session=session,
                    step_index=idx,
                    last_dom=dom_snapshot,
                )
                if action_result.dom_snapshot is not None:
                    dom_snapshot = action_result.dom_snapshot
                if action_result.layout_graph is not None:
                    layout_graph = action_result.layout_graph
                if action_kind == "navigate" and action.get("url") and action_result.status == "ok":
                    current_url = str(action.get("url"))
                    self._current_url = current_url
                if action_result.screenshot:
                    if action_result.failed():
                        failure_screenshot_path = failure_screenshot_path or action_result.screenshot
                    else:
                        screenshots.append(action_result.screenshot)
            except Exception as exc:  # noqa: BLE001
                error = str(exc)
                action_result = ActionResult(status="failed", error=error, details={"exception": error})
                step_entry["status"] = "error"
                step_entry["error"] = error
                failure_dom_path, failure_screenshot_path = await self._capture_failure_artifacts(
                    session=session,
                    step_index=idx,
                    last_dom=dom_snapshot,
                )
                self._log_step_entry(self._build_step_metadata(step_entry, action), action_result)
                await self._log_trace(
                    step_index=idx,
                    action_name=action_kind,
                    url=action.get("url") if isinstance(action.get("url"), str) else None,
                    completion=None,
                    action_result=action_result,
                )
                break
            else:
                if action_result.details:
                    step_entry["details"] = action_result.details
                step_entry["status"] = action_result.status
                if action_result.error:
                    step_entry["error"] = action_result.error
                if action_kind == "dom_presence_check":
                    dom_presence_events.append({
                        "step": step_entry["id"],
                        "status": action_result.status,
                        **action_result.details,
                    })
                self._log_step_entry(self._build_step_metadata(step_entry, action), action_result)
                await self._log_trace(
                    step_index=idx,
                    action_name=action_kind,
                    url=action.get("url") if isinstance(action.get("url"), str) else None,
                    completion=None,
                    action_result=action_result,
                )
                self._update_login_flow_state(action, action_result)
                if session and not secure_area_detection:
                    detection = await self._maybe_handle_secure_area(
                        action=action,
                        session=session,
                        step_index=idx,
                    )
                    if detection:
                        secure_area_detection = detection
                        screenshot_path = detection.get("screenshot")
                        if screenshot_path and screenshot_path not in screenshots:
                            screenshots.append(screenshot_path)
                if action_result.failed():
                    error = action_result.error or "action_failed"
                    failure_dom_path = failure_dom_path or (action_result.details or {}).get("dom_path")
                    failure_screenshot_path = failure_screenshot_path or action_result.screenshot
                    if not failure_dom_path or not failure_screenshot_path:
                        captured_dom, captured_shot = await self._capture_failure_artifacts(
                            session=session,
                            step_index=idx,
                            last_dom=dom_snapshot,
                        )
                        failure_dom_path = failure_dom_path or captured_dom
                        failure_screenshot_path = failure_screenshot_path or captured_shot
                    break

        completion_reason = "browser actions completed" if error is None else error
        if secure_area_detection:
            completion_reason = "secure_area_detected"
        completion = build_completion(
            complete=error is None,
            reason=completion_reason,
            payload={"steps": len(steps)},
        )
        result: Dict[str, Any] = {
            "steps": steps,
            "screenshots": screenshots,
            "dom_snapshot": dom_snapshot,
            "layout_graph": layout_graph,
            "completion": completion,
            "error": error,
            "failure_dom_path": failure_dom_path,
            "failure_screenshot_path": failure_screenshot_path,
        }
        if dom_presence_events:
            result["dom_presence"] = dom_presence_events
        if secure_area_detection:
            result["secure_area"] = secure_area_detection
        if error:
            self._teardown_required = True
        return result

    def set_mission_context(
        self,
        *,
        mission_instruction: str | None = None,
        subgoal_description: str | None = None,
    ) -> None:
        """Optional context used for selector heuristics."""

        self._mission_instruction = mission_instruction
        self._subgoal_description = subgoal_description

    async def _perform_action(
        self,
        action: BrowserAction,
        *,
        session: BrowserSession | None,
        step_index: int,
        last_dom: Optional[str],
    ) -> ActionResult:
        kind = (action.get("action") or "").lower()
        if kind == "navigate":
            return await self._handle_navigation(action, session=session, step_index=step_index)
        if kind == "fill":
            return await self._handle_fill(action, session=session)
        if kind == "click":
            return await self._handle_click(action, session=session)
        if kind in {"wait_for", "wait_for_navigation"}:
            return await self._handle_wait_for(action, session=session)
        if kind == "wait_for_selector":
            return await self._handle_selector_wait(action, session=session)
        if kind == "dom_presence_check":
            return await self._handle_dom_presence(
                action,
                session=session,
                step_index=step_index,
                last_dom=last_dom,
            )
        if kind == "screenshot" and self.screenshot_enabled:
            screenshot_path = await self._capture_screenshot(
                session=session,
                action=action,
                step_index=step_index,
            )
            return ActionResult(screenshot=screenshot_path)
        if kind == "extract_dom" and session:
            page = session.page
            content = await page.content()
            html, layout = self._record_dom(content, step_index)
            return ActionResult(dom_snapshot=html, layout_graph=layout)
        if kind == "extract_dom":
            dom_content = last_dom or "<html></html>"
            layout_content = dom_utils.build_layout_graph(dom_content)
            if self.logger:
                self.logger.save_dom(dom_content, step_index=step_index)
                self.logger.save_layout_graph(layout_content, step_index=step_index)
            self._last_dom_html = dom_content
            return ActionResult(dom_snapshot=dom_content, layout_graph=layout_content)
        if kind in {"retry", "reload_if_failed"}:
            return ActionResult()
        raise ValueError(f"Unsupported action: {kind}")

    async def _handle_navigation(
        self,
        action: BrowserAction,
        *,
        session: BrowserSession | None,
        step_index: int,
    ) -> ActionResult:
        url_value = action.get("url")
        if not url_value:
            raise ValueError("navigate action missing url")
        url = str(url_value)
        local_content = self._try_load_local_resource(url)
        if local_content is not None:
            html, layout = self._record_dom(local_content, step_index)
            return ActionResult(dom_snapshot=html, layout_graph=layout)
        last_exc: Exception | None = None
        for attempt in range(1, self.retry_limit + 2):
            try:
                if session:
                    page = session.page
                    await page.goto(url, wait_until="load")
                    await page.wait_for_load_state("domcontentloaded", timeout=10000)
                    try:
                        await page.wait_for_load_state("networkidle", timeout=15000)
                    except Exception:  # noqa: BLE001 - continue on slow pages
                        self._trace_logger.debug("networkidle timeout for %s", url)
                    content = await page.content()
                else:
                    response = await self._http_client.get(url)
                    response.raise_for_status()
                    content = response.text
                html, layout = self._record_dom(content, step_index)
                return ActionResult(dom_snapshot=html, layout_graph=layout)
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                await asyncio.sleep(0.25 * attempt)
        raise RuntimeError(f"failed to navigate to {url}: {last_exc}")

    async def _handle_fill(
        self,
        action: BrowserAction,
        *,
        session: BrowserSession | None,
    ) -> ActionResult:
        selector = (action.get("selector") or "").strip()
        value = action.get("text")
        if value is None:
            value = action.get("value")
        if selector == "":
            return ActionResult(status="failed", error="fill_missing_selector")
        text_value = str(value or "")
        details = {"selector": selector, "value": text_value}
        if not session:
            details["mode"] = "dry_run"
            return ActionResult(details=details)
        page = session.page
        timeout = int(action.get("timeout") or 5000)
        try:
            await page.fill(selector, text_value, timeout=timeout)
        except Exception as exc:  # noqa: BLE001
            return ActionResult(status="failed", error="fill_failed", details={**details, "exception": str(exc)})
        return ActionResult(details=details)

    async def _handle_click(
        self,
        action: BrowserAction,
        *,
        session: BrowserSession | None,
    ) -> ActionResult:
        selector = (action.get("selector") or "").strip()
        if selector == "":
            return ActionResult(status="failed", error="click_missing_selector")
        details = {"selector": selector}
        if not session:
            details["mode"] = "dry_run"
            return ActionResult(details=details)
        timeout = int(action.get("timeout") or 5000)
        page = session.page
        try:
            await page.click(selector, timeout=timeout)
        except Exception as exc:  # noqa: BLE001
            return ActionResult(status="failed", error="click_failed", details={**details, "exception": str(exc)})
        return ActionResult(details=details)

    async def _handle_wait_for(
        self,
        action: BrowserAction,
        *,
        session: BrowserSession | None,
    ) -> ActionResult:
        action_name = (action.get("action") or "").lower()
        default_state = "networkidle" if action_name == "wait_for_navigation" else "load"
        mode = (action.get("state") or default_state).lower()
        timeout = int(action.get("timeout") or 8000)
        if not session:
            return ActionResult(details={"state": mode, "mode": "dry_run", "timeout_ms": timeout})
        try:
            await session.page.wait_for_load_state(mode, timeout=timeout)
        except Exception as exc:  # noqa: BLE001
            return ActionResult(status="failed", error="wait_failed", details={"state": mode, "exception": str(exc)})
        return ActionResult(details={"state": mode, "timeout_ms": timeout})

    async def _handle_selector_wait(
        self,
        action: BrowserAction,
        *,
        session: BrowserSession | None,
    ) -> ActionResult:
        selector = (action.get("selector") or "").strip()
        if not selector:
            return ActionResult(status="failed", error="wait_missing_selector")
        timeout = int(action.get("timeout") or 5000)
        state = (action.get("state") or "visible").lower()
        if not session:
            return ActionResult(details={"selector": selector, "state": state, "mode": "dry_run"})
        try:
            await session.page.wait_for_selector(selector, timeout=timeout, state=state)
        except Exception as exc:  # noqa: BLE001
            return ActionResult(status="failed", error="wait_selector_failed", details={"selector": selector, "exception": str(exc)})
        return ActionResult(details={"selector": selector, "state": state, "timeout_ms": timeout})

    async def _handle_dom_presence(
        self,
        action: BrowserAction,
        *,
        session: BrowserSession | None,
        step_index: int,
        last_dom: Optional[str],
    ) -> ActionResult:
        selector = (action.get("selector") or "").strip()
        if not selector:
            return ActionResult(
                status="failed",
                error="dom_presence_failed",
                details={"reason": "missing_selector"},
            )
        selectors = self.expand_selectors(selector)
        resolver_details: Optional[Dict[str, Any]] = None
        resolver_html: Optional[str] = last_dom or self._last_dom_html
        if not resolver_html and session:
            try:
                resolver_html = await session.page.content()
            except Exception:  # noqa: BLE001 - resolver is best-effort
                resolver_html = None
        selectors, resolver_details = self._apply_selector_resolver(
            selector=selector,
            selectors=selectors,
            html_source=resolver_html,
            action=action,
        )
        timeout_ms = int(action.get("timeout") or 8000)
        page = session.page if session else None
        if page:
            try:
                matched = await self.wait_for_dom(page, selectors, timeout_ms=timeout_ms)
                details = {
                    "matched_selector": matched,
                    "selectors": selectors,
                    "wait_timeout_ms": timeout_ms,
                    "mode": "live",
                }
                resolver_payload = self._format_resolver_details(resolver_details, matched)
                if resolver_payload:
                    details["selector_resolver"] = resolver_payload
                return ActionResult(details=details)
            except TimeoutError as exc:
                dom_path, screenshot = await self._capture_dom_failure_artifacts(
                    session=session,
                    step_index=step_index,
                    last_dom=last_dom,
                )
                details = {
                    "reason": str(exc),
                    "selectors": selectors,
                    "dom_path": dom_path,
                    "wait_timeout_ms": timeout_ms,
                }
                resolver_payload = self._format_resolver_details(resolver_details, None)
                if resolver_payload:
                    details["selector_resolver"] = resolver_payload
                if screenshot:
                    details["screenshot"] = screenshot
                return ActionResult(
                    status="failed",
                    error="dom_presence_failed",
                    screenshot=screenshot,
                    details=details,
                )
        html_source = last_dom or self._last_dom_html
        matched = self._match_selector_in_html(selectors, html_source)
        if matched:
            details = {
                "matched_selector": matched,
                "selectors": selectors,
                "mode": "snapshot",
            }
            resolver_payload = self._format_resolver_details(resolver_details, matched)
            if resolver_payload:
                details["selector_resolver"] = resolver_payload
            return ActionResult(details=details)
        return ActionResult(
            status="failed",
            error="dom_presence_failed",
            details=self._build_snapshot_failure_details(selectors, resolver_details),
        )

    async def wait_for_dom(self, page: Any, selectors: List[str], *, timeout_ms: int = 8000) -> str:
        failures: List[str] = []
        for selector in selectors:
            started = perf_counter()
            try:
                await page.wait_for_selector(selector, timeout=timeout_ms, state="attached")
                elapsed = round((perf_counter() - started) * 1000, 2)
                self._trace_logger.debug("DOM selector success", extra={"selector": selector, "elapsed_ms": elapsed})
                return selector
            except Exception as exc:  # noqa: BLE001
                elapsed = round((perf_counter() - started) * 1000, 2)
                self._trace_logger.debug(
                    "DOM selector failure",
                    extra={"selector": selector, "elapsed_ms": elapsed, "error": str(exc)},
                )
                failures.append(str(exc))
        raise TimeoutError(f"DOM presence failed. Tried: {selectors}")

    def expand_selectors(self, selector: str) -> List[str]:
        raw = selector.strip()
        if not raw:
            return []
        candidates: List[str] = [raw]
        token = raw.lstrip("#.")
        token_root = self._selector_token_root(token)
        title_token = token.capitalize()
        if raw.startswith("#"):
            candidates.extend([
                f"#{token}",
                f"input#{token}",
                f'[id="{token}"]',
                f'[name="{token}"]',
                f'input[name="{token}"]',
                f'input[id="{token}"]',
                f"//input[contains(@id,\"{token_root}\")]",
                f"//*[contains(text(),\"{title_token}\")]",
            ])
        elif raw.startswith("."):
            candidates.extend([
                f"button{raw}",
                f"a{raw}",
                f"div{raw}",
                f"span{raw}",
                f'//*[@class and contains(@class,\"{token_root}\")]',
            ])
        else:
            candidates.extend([
                f"#{token}",
                f".{token}",
                f'input[name="{token}"]',
                f'button[name="{token}"]',
                f"//input[contains(@name,\"{token_root}\")]",
                f"//*[contains(text(),\"{title_token}\")]",
            ])
        return self._dedupe_selectors(candidates)

    def _match_selector_in_html(self, selectors: List[str], html: str) -> Optional[str]:
        lowered = html.lower()
        for selector in selectors:
            token = self._selector_token(selector)
            if token and token in lowered:
                return selector
        return None

    def _apply_selector_resolver(
        self,
        *,
        selector: str,
        selectors: List[str],
        html_source: Optional[str],
        action: BrowserAction,
    ) -> tuple[List[str], Optional[Dict[str, Any]]]:
        mission_hint = self._mission_instruction or self._current_goal
        resolver = SelectorResolver(
            html_source or "",
            mission_text=mission_hint,
            goal_text=self._current_goal,
            current_url=self._current_url,
            base_selector=selector,
        )
        role = self._infer_selector_role(selector, action, resolver)
        role_candidates: List[SelectorCandidate] = []
        if html_source:
            if role == "username":
                role_candidates = resolver.resolve_username()
            elif role == "password":
                role_candidates = resolver.resolve_password()
            elif role == "login_button":
                role_candidates = resolver.resolve_login_button()
        login_candidates = resolver.login_override_candidates()
        combined_candidates = self._dedupe_candidate_objects(login_candidates + role_candidates)
        if not combined_candidates:
            return selectors, None
        selectors = self._dedupe_selectors([cand.selector for cand in combined_candidates] + selectors)
        top_candidates = combined_candidates[:5]
        resolver_details = {
            "role": role,
            "candidates": [
                {
                    "selector": cand.selector,
                    "score": cand.score,
                    "metadata": cand.metadata,
                }
                for cand in top_candidates
            ],
            "candidate_selectors": [cand.selector for cand in top_candidates],
            "login_bundle": resolver.get_login_selector_bundle(),
            "context": self._resolver_context_snapshot(resolver),
        }
        return selectors, resolver_details

    def _format_resolver_details(
        self,
        resolver_details: Optional[Dict[str, Any]],
        matched: Optional[str],
    ) -> Optional[Dict[str, Any]]:
        if not resolver_details:
            return None
        payload = dict(resolver_details)
        candidates = payload.get("candidate_selectors", [])
        payload["matched_from_resolver"] = bool(matched and matched in candidates)
        payload["attempted"] = True
        return payload

    def _build_snapshot_failure_details(
        self,
        selectors: List[str],
        resolver_details: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        details: Dict[str, Any] = {
            "reason": "selector_missing_in_snapshot",
            "selectors": selectors,
        }
        resolver_payload = self._format_resolver_details(resolver_details, None)
        if resolver_payload:
            details["selector_resolver"] = resolver_payload
        return details

    @staticmethod
    def _dedupe_selectors(selectors: List[str]) -> List[str]:
        seen: set[str] = set()
        ordered: List[str] = []
        for candidate in selectors:
            normalized = candidate.strip()
            if normalized and normalized not in seen:
                ordered.append(normalized)
                seen.add(normalized)
        return ordered

    @staticmethod
    def _dedupe_candidate_objects(candidates: List[SelectorCandidate]) -> List[SelectorCandidate]:
        if not candidates:
            return []
        best: Dict[str, SelectorCandidate] = {}
        for candidate in candidates:
            selector = candidate.selector.strip()
            if not selector:
                continue
            current = best.get(selector)
            if current is None or current.score < candidate.score:
                best[selector] = candidate
        return sorted(best.values(), key=lambda cand: cand.score, reverse=True)

    def _resolver_context_snapshot(self, resolver: SelectorResolver) -> Dict[str, Any]:
        return {
            "login_context": resolver.has_login_context(),
            "heroku_override": resolver.is_heroku_login(),
            "current_url": self._current_url,
        }

    def _infer_selector_role(
        self,
        selector: str,
        action: Optional[BrowserAction] = None,
        resolver: SelectorResolver | None = None,
    ) -> Optional[str]:
        lowered = selector.lower()
        if self._selector_mentions_tokens(lowered, USERNAME_TOKENS):
            return "username"
        if self._selector_mentions_tokens(lowered, PASSWORD_TOKENS):
            return "password"
        if self._selector_mentions_tokens(lowered, BUTTON_TOKENS) or "type=submit" in lowered:
            return "login_button"
        action_field = str((action or {}).get("field") or "").lower()
        if action_field:
            if any(token in action_field for token in USERNAME_TOKENS):
                return "username"
            if any(token in action_field for token in PASSWORD_TOKENS):
                return "password"
        if resolver and resolver.has_login_context():
            return "username"
        return None

    @staticmethod
    def _selector_mentions_tokens(selector_value: str, tokens: tuple[str, ...]) -> bool:
        return any(token in selector_value for token in tokens)

    def _selector_token(self, selector: str) -> Optional[str]:
        raw = selector.strip()
        if not raw:
            return None
        if raw.startswith("#"):
            return f'id="{raw[1:].lower()}"'
        if raw.startswith("."):
            return raw[1:].lower()
        if "contains(@id" in raw:
            fragment = self._extract_between(raw, 'contains(@id,', ")")
            return (fragment or "").strip('\"').lower()
        if "contains(@name" in raw:
            fragment = self._extract_between(raw, 'contains(@name,', ")")
            return (fragment or "").strip('\"').lower()
        if "contains(text()" in raw:
            fragment = self._extract_between(raw, 'contains(text(),', ")")
            return (fragment or "").strip('\"').lower()
        if "[name=" in raw:
            fragment = self._extract_between(raw, '[name="', '"]')
            return (fragment or "").lower()
        if "[id=" in raw:
            fragment = self._extract_between(raw, '[id="', '"]')
            return (fragment or "").lower()
        return raw.lower()

    @staticmethod
    def _extract_between(value: str, start: str, end: str) -> Optional[str]:
        if start not in value or end not in value:
            return None
        prefix, _, remainder = value.partition(start)
        if not remainder:
            return None
        fragment, _, _ = remainder.partition(end)
        return fragment

    @staticmethod
    def _selector_token_root(token: str) -> str:
        lowered = token.lower()
        for splitter in ("-", "_", "."):
            if splitter in lowered:
                lowered = lowered.split(splitter)[0]
                break
        return lowered

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
        file_path = self._ensure_fallback_dir(step_index) / name
        file_path.write_bytes(data)
        return str(file_path)

    async def _capture_failure_artifacts(
        self,
        *,
        session: BrowserSession | None,
        step_index: int,
        last_dom: Optional[str],
    ) -> tuple[Optional[str], Optional[str]]:
        html_snapshot: Optional[str] = None
        if session:
            try:
                html_snapshot = await session.page.content()
            except Exception:  # noqa: BLE001 - best-effort capture
                html_snapshot = None
        if not html_snapshot:
            html_snapshot = last_dom or "<html></html>"
        dom_path = self._save_dom_artifact(html_snapshot, step_index=step_index, name="failure_dom.html")
        screenshot_path = await self._capture_screenshot(
            session=session,
            action={"name": f"failure_step_{step_index:03d}.png"},
            step_index=step_index,
        )
        return dom_path, screenshot_path

    async def _capture_dom_failure_artifacts(
        self,
        *,
        session: BrowserSession | None,
        step_index: int,
        last_dom: Optional[str],
    ) -> tuple[Optional[str], Optional[str]]:
        html_snapshot: Optional[str] = None
        if session:
            try:
                html_snapshot = await session.page.content()
            except Exception:  # noqa: BLE001
                html_snapshot = None
        if not html_snapshot:
            html_snapshot = last_dom or "<html></html>"
        dom_path = self._save_dom_artifact(html_snapshot, step_index=step_index, name="dom_failure.html")
        screenshot = await self._capture_screenshot(
            session=session,
            action={"name": "dom_failure.png"},
            step_index=step_index,
        )
        return dom_path, screenshot

    def _record_dom(self, html: str, step_index: int) -> tuple[str, str]:
        layout = dom_utils.build_layout_graph(html)
        if self.logger:
            self.logger.save_dom(html, step_index=step_index)
            self.logger.save_layout_graph(layout, step_index=step_index)
        self._last_dom_html = html
        return html, layout

    def _save_dom_artifact(self, html: str, *, step_index: int, name: str) -> str:
        if self.logger:
            path = self.logger.save_dom(html, step_index=step_index, name=name)
            return str(path)
        file_path = self._ensure_fallback_dir(step_index) / name
        file_path.write_text(html, encoding="utf-8")
        return str(file_path)

    def _ensure_fallback_dir(self, step_index: int) -> Path:
        step_dir = Path("artifacts") / f"step_{step_index:03d}"
        step_dir.mkdir(parents=True, exist_ok=True)
        return step_dir

    async def _log_trace(
        self,
        *,
        step_index: int,
        action_name: str,
        url: str | None,
        completion: Dict[str, Any] | None,
        action_result: ActionResult | Mapping[str, Any] | None = None,
    ) -> None:
        if not self.logger:
            return
        log_fn = self.logger.log_trace
        accepts_result = "action_result" in inspect.signature(log_fn).parameters
        kwargs = {
            "goal": None,
            "step_index": step_index,
            "action": action_name,
            "url": url,
            "completion": completion,
        }
        if accepts_result:
            kwargs["action_result"] = action_result
        log_fn(**kwargs)

    def _log_step_entry(
        self,
        metadata: Dict[str, Any],
        action_result: ActionResult | Mapping[str, Any] | None,
    ) -> None:
        if not self.logger or not hasattr(self.logger, "log_step"):
            return
        log_fn = self.logger.log_step  # type: ignore[attr-defined]
        accepts_result = "action_result" in inspect.signature(log_fn).parameters
        kwargs = {"metadata": metadata, "goal": None}
        if accepts_result:
            kwargs["action_result"] = action_result
        log_fn(**kwargs)

    def _build_step_metadata(self, step_entry: Dict[str, Any], action: BrowserAction) -> Dict[str, Any]:
        payload = dict(action)
        metadata: Dict[str, Any] = {
            "action": step_entry.get("action"),
            "payload": payload,
            "step_id": step_entry.get("id"),
            "status": step_entry.get("status"),
        }
        if "error" in step_entry:
            metadata["error"] = step_entry["error"]
        if "details" in step_entry:
            metadata["details"] = step_entry["details"]
        return metadata

    def _reset_login_flow_state(self) -> None:
        self._login_flow_state = {
            "username_filled": False,
            "password_filled": False,
            "submit_clicked": False,
        }

    def _update_login_flow_state(self, action: BrowserAction, result: ActionResult) -> None:
        if result.failed():
            return
        selector = (action.get("selector") or "").lower()
        kind = (action.get("action") or "").lower()
        if kind == "fill":
            if "#username" in selector:
                self._login_flow_state["username_filled"] = True
            if "#password" in selector:
                self._login_flow_state["password_filled"] = True
        elif kind == "click":
            if "submit" in selector or "login" in selector:
                self._login_flow_state["submit_clicked"] = True

    async def _maybe_handle_secure_area(
        self,
        *,
        action: BrowserAction,
        session: BrowserSession | None,
        step_index: int,
    ) -> Optional[Dict[str, Any]]:
        if self._secure_area_info or session is None:
            return None
        if not (
            self._login_flow_state.get("username_filled")
            and self._login_flow_state.get("password_filled")
            and self._login_flow_state.get("submit_clicked")
        ):
            return None
        action_kind = (action.get("action") or "").lower()
        if action_kind not in {"wait_for_navigation", "wait_for", "screenshot", "click"}:
            return None
        # secure area success termination condition
        detected, current_url = await self._detect_secure_area(session=session)
        if not detected:
            return None
        screenshot_path = await self._capture_screenshot(
            session=session,
            action={"name": "secure_area.png"},
            step_index=step_index,
        )
        payload = {
            "detected": True,
            "timestamp": datetime.now(UTC).isoformat(),
            "step_index": step_index,
            "url": current_url,
            "screenshot": screenshot_path,
        }
        self._secure_area_info = payload
        print("[SUCCESS] Secure area detected — mission complete")
        return payload

    async def _detect_secure_area(self, *, session: BrowserSession) -> tuple[bool, Optional[str]]:
        page = session.page
        try:
            current_url = page.url
        except Exception:  # noqa: BLE001 - best effort url capture
            current_url = None
        url_match = bool(current_url and "/secure" in current_url.lower())
        text_match = False
        try:
            text_match = await page.evaluate(
                """
                () => {
                    const phrases = ["secure area", "you logged into a secure area!"];
                    const selectors = ["h2", "div"];
                    return selectors.some((selector) => {
                        return Array.from(document.querySelectorAll(selector)).some((node) => {
                            const text = (node.innerText || "").toLowerCase();
                            if (!text) {
                                return false;
                            }
                            return phrases.some((phrase) => text.includes(phrase));
                        });
                    });
                }
                """
            )
        except Exception:  # noqa: BLE001 - DOM evaluation best effort
            text_match = False
        return (url_match or bool(text_match)), current_url

    async def await_manual_close(self) -> None:
        session = self._session
        if not session:
            return
        page = session.page
        try:
            await page.wait_for_event("close")
        except Exception:  # noqa: BLE001 - user may close browser via other means
            await asyncio.sleep(0.1)
        finally:
            await self.shutdown()

    async def _ensure_session(self) -> BrowserSession | None:
        if not self.enable_playwright or async_playwright is None:
            return None
        if self._session:
            return self._session
        playwright = await async_playwright().start()
        launch_args = [
            "--disable-gpu",
            "--disable-dev-shm-usage",
            "--no-sandbox",
            "--disable-setuid-sandbox",
        ]
        browser = await playwright.chromium.launch(
            headless=self.headless,
            slow_mo=self.slow_mo,
            args=launch_args,
        )
        print("[DEBUG] Chromium launched with sandbox disabled")
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

    async def shutdown(self) -> None:
        """Flush artifacts, capture final state, and close browser resources."""

        if self._shutdown_complete:
            return
        summary_writer = getattr(self.logger, "write_summary", None)
        if callable(summary_writer):
            try:
                summary_writer({
                    "event": "browser_shutdown",
                    "timestamp": datetime.now(UTC).isoformat(),
                })
            except Exception:  # noqa: BLE001 - logging should not raise
                self._trace_logger.debug("artifact summary flush failed", exc_info=True)
        session = self._session
        if session and self._teardown_required and self.screenshot_enabled:
            try:
                await self._capture_screenshot(
                    session=session,
                    action={"name": "final_state.png"},
                    step_index=0,
                )
            except Exception:  # noqa: BLE001 - best effort capture
                self._trace_logger.debug("final screenshot capture failed", exc_info=True)
        if session:
            try:
                await session.close()
            finally:
                self._session = None
        try:
            await self._http_client.aclose()
        except Exception:  # noqa: BLE001 - closing client is best effort
            self._trace_logger.debug("http client shutdown failed", exc_info=True)
        self._shutdown_complete = True
        self._teardown_required = False

    async def close(self) -> None:
        """Backward-compatible alias for shutdown."""

        await self.shutdown()
