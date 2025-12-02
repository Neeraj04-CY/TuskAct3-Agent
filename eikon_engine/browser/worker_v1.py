"""Planner v3 aware Browser Worker v1 implementation."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, List, Sequence
from urllib.parse import urlparse
from urllib.request import url2pathname

import httpx

from eikon_engine.browser.schema_v1 import RunSummary, RunTrace, StepAction
from eikon_engine.utils import dom_utils, vision_utils
from eikon_engine.utils import file_ops
from eikon_engine.utils.logging_utils import ArtifactLogger

try:  # pragma: no cover - optional runtime dependency
    from playwright.async_api import async_playwright
except Exception:  # pragma: no cover - Playwright not installed
    async_playwright = None  # type: ignore[assignment]


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


class BrowserWorkerV1:
    """Executes Planner v3 tasks sequentially and records run traces."""

    def __init__(
        self,
        *,
        settings: Dict[str, Any] | None = None,
        logger: ArtifactLogger | None = None,
        enable_playwright: bool | None = None,
    ) -> None:
        config = settings or {}
        browser_cfg = config.get("browser", {})
        logging_cfg = config.get("logging", {})
        self.logger = logger
        self.enable_playwright = enable_playwright if enable_playwright is not None else async_playwright is not None
        self.headless = bool(browser_cfg.get("headless", True))
        self.slow_mo = int(browser_cfg.get("slow_mo", 0))
        self.allow_external = bool(browser_cfg.get("allow_external", True))
        self.screenshot_enabled = bool(browser_cfg.get("screenshot", True))
        self._max_wait = float(browser_cfg.get("max_wait_seconds", 0.5))
        artifact_root = Path(logging_cfg.get("artifact_root", "artifacts"))
        self._artifact_base = artifact_root / "browser_worker_v1"
        self._http_client = httpx.AsyncClient(follow_redirects=True)
        self._session: BrowserSession | None = None
        self._state: Dict[str, Any] = {
            "url": "about:blank",
            "dom": "<html></html>",
            "layout": "document",
        }
        self._local_step_index = 0

    async def run_plan(self, plan: Dict[str, Any], *, goal: str | None = None) -> RunSummary:
        """Execute every BrowserWorker task contained in the plan."""

        plan_id = str(plan.get("plan_id") or "plan")
        plan_goal = goal or str(plan.get("goal") or "")
        actions = self._flatten_actions(plan)
        if not actions:
            raise ValueError("BrowserWorkerV1 requires at least one planner action")

        traces: List[RunTrace] = []
        failure_reason: str | None = None
        failures = 0
        start_ts = perf_counter()

        for action in actions:
            step_number = self._log_step(action, goal=plan_goal)
            trace, error = await self._execute_action(action, step_number=step_number)
            traces.append(trace)
            self._emit_trace_log(action, trace, goal=plan_goal)
            if error and failure_reason is None:
                failure_reason = f"{action.get('action')}: {error}"
            if error:
                failures += 1
                break

        duration = round(perf_counter() - start_ts, 4)
        summary: RunSummary = {
            "plan_id": plan_id,
            "goal": plan_goal,
            "total_steps": len(traces),
            "failures": failures,
            "recovery_steps": sum(1 for trace in traces if trace.get("recovery_applied")),
            "final_url": str(self._state.get("url") or "about:blank"),
            "run_duration": duration,
            "traces": traces,
            "first_failure_type": failure_reason,
            "run_output": self._build_run_output(traces, failures),
        }
        if self.logger:
            self.logger.write_summary(summary)  # type: ignore[arg-type]
        return summary

    async def close(self) -> None:
        """Release network and browser resources."""

        await self._http_client.aclose()
        if self._session:
            await self._session.close()
            self._session = None

    async def _execute_action(self, action: StepAction, *, step_number: int) -> tuple[RunTrace, str | None]:
        action_name = str(action.get("action") or "").lower()
        started_at = self._timestamp()
        trace: RunTrace = {
            "step_id": action.get("id") or f"step_{step_number:03d}",
            "step_index": step_number,
            "action": action_name,
            "status": "ok",
            "start_time": started_at,
            "end_time": started_at,
            "screenshot_path": None,
            "dom_path": None,
            "error": None,
            "delta_state": {},
            "recovery_applied": bool(action.get("_recovery")),
        }
        error_message: str | None = None
        try:
            if action_name == "navigate":
                trace["dom_path"] = await self._handle_navigation(action, step_number=step_number)
            elif action_name == "fill":
                await self._handle_fill(action)
            elif action_name == "click":
                await self._handle_click(action)
            elif action_name in {"wait_for", "wait_for_selector"}:
                await self._handle_wait(action)
            elif action_name == "dom_presence_check":
                await self._handle_dom_check(action)
            elif action_name == "screenshot":
                trace["screenshot_path"] = await self._handle_screenshot(action, step_number=step_number)
            elif action_name in {"extract_dom", "extract"}:
                trace["dom_path"] = await self._handle_dom_extract(step_number=step_number)
            elif action_name in {"retry", "reload_if_failed"}:
                await self._handle_recovery(action)
            elif not action_name:
                raise ValueError("action field is required")
            else:
                raise ValueError(f"Unsupported action: {action_name}")
        except Exception as exc:  # noqa: BLE001
            error_message = str(exc)
            trace["status"] = "error"
            trace["error"] = error_message
        finally:
            trace["end_time"] = self._timestamp()
            trace["delta_state"] = {
                "url": self._state.get("url"),
                "task_id": action.get("task_id"),
                "bucket": action.get("bucket"),
            }
        return trace, error_message

    async def _handle_navigation(self, action: StepAction, *, step_number: int) -> str:
        url_value = action.get("url")
        if not url_value:
            raise ValueError("navigate action missing url")
        session = await self._ensure_session()
        html: str
        if session:
            await session.page.goto(str(url_value), wait_until="load")
            html = await session.page.content()
            current_url = session.page.url
        else:
            current_url = str(url_value)
            html = await self._load_html(current_url)
        dom_path = self._save_dom(html, step_index=step_number)
        layout = dom_utils.build_layout_graph(html)
        self._save_layout(layout, step_index=step_number)
        self._state.update({"url": current_url, "dom": html, "layout": layout})
        return str(dom_path)

    async def _handle_fill(self, action: StepAction) -> None:
        fields = action.get("fields") or []
        if fields:
            for field in fields:
                if not field.get("selector"):
                    raise ValueError("fill fields require selector")
        else:
            if not action.get("selector"):
                raise ValueError("fill action missing selector")

    async def _handle_click(self, action: StepAction) -> None:
        if not action.get("selector"):
            raise ValueError("click action missing selector")

    async def _handle_wait(self, action: StepAction) -> None:
        timeout = action.get("timeout")
        seconds = 0.0
        if timeout is not None:
            try:
                seconds = max(float(timeout) / 1000.0, 0.0)
            except (TypeError, ValueError):  # pragma: no cover - defensive
                seconds = 0.0
        await asyncio.sleep(min(seconds, self._max_wait))

    async def _handle_dom_check(self, action: StepAction) -> None:
        selector = action.get("selector")
        if not selector:
            raise ValueError("dom_presence_check requires selector")
        dom_snapshot = self._state.get("dom") or ""
        if selector not in dom_snapshot:
            raise ValueError(f"selector {selector} not found in DOM")

    async def _handle_screenshot(self, action: StepAction, *, step_number: int) -> str | None:
        if not self.screenshot_enabled:
            return None
        session = await self._ensure_session()
        if session:
            payload = await session.page.screenshot(full_page=True)
        else:
            payload = vision_utils.empty_screenshot()
        name = action.get("name") or vision_utils.generate_screenshot_name("screenshot", step_number)
        path = self._save_screenshot(payload, step_index=step_number, name=name)
        return str(path)

    async def _handle_dom_extract(self, *, step_number: int) -> str:
        html = self._state.get("dom") or "<html></html>"
        dom_path = self._save_dom(html, step_index=step_number, name="extract_dom.html")
        layout = self._state.get("layout") or dom_utils.build_layout_graph(html)
        self._save_layout(layout, step_index=step_number)
        return str(dom_path)

    async def _handle_recovery(self, action: StepAction) -> None:
        _ = action
        await asyncio.sleep(0)

    def _flatten_actions(self, plan: Dict[str, Any]) -> List[StepAction]:
        actions: List[StepAction] = []
        tasks: Sequence[Dict[str, Any]] = plan.get("tasks") or []
        for task in tasks:
            if task.get("tool") and task.get("tool") != "BrowserWorker":
                continue
            inputs = task.get("inputs") or {}
            for raw in inputs.get("actions", []):
                cloned: StepAction = dict(raw)  # type: ignore[assignment]
                cloned.setdefault("id", f"step_{len(actions) + 1:03d}")
                cloned.setdefault("task_id", task.get("id"))
                cloned.setdefault("bucket", task.get("bucket"))
                cloned.setdefault("metadata", {})
                actions.append(cloned)
        if not actions:
            for raw in plan.get("actions", []):
                cloned = dict(raw)
                cloned.setdefault("id", f"step_{len(actions) + 1:03d}")
                cloned.setdefault("metadata", {})
                actions.append(cloned)
        return actions

    def _log_step(self, action: StepAction, *, goal: str | None) -> int:
        if self.logger:
            return self.logger.log_step({"action": action.get("action"), "payload": dict(action)}, goal=goal)
        self._local_step_index += 1
        return self._local_step_index

    def _emit_trace_log(self, action: StepAction, trace: RunTrace, *, goal: str | None) -> None:
        if not self.logger:
            return
        self.logger.log_trace(
            goal=goal,
            step_index=trace["step_index"],
            action=action.get("action"),
            url=action.get("url"),
            completion={"status": trace["status"], "error": trace.get("error")},
        )

    def _build_run_output(self, traces: Sequence[RunTrace], failures: int) -> str:
        if failures:
            failing = next((trace for trace in traces if trace.get("status") == "error"), None)
            if failing:
                return f"failed on {failing['action']} at step {failing['step_index']}"
            return "failed"
        return f"completed {len(traces)} steps"

    async def _load_html(self, url: str) -> str:
        local = self._try_load_local_resource(url)
        if local is not None:
            return local
        if not self.allow_external:
            raise RuntimeError("external navigation disabled")
        response = await self._http_client.get(url)
        response.raise_for_status()
        return response.text

    def _try_load_local_resource(self, url: str) -> str | None:
        parsed = urlparse(url)
        candidate: Path | None = None
        if parsed.scheme == "file":
            candidate = Path(url2pathname(parsed.path))
        elif parsed.scheme == "":
            candidate = Path(url)
        if candidate and candidate.exists():
            return candidate.read_text(encoding="utf-8")
        return None

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

    def _save_dom(self, html: str, *, step_index: int, name: str | None = None) -> Path:
        if self.logger:
            return self.logger.save_dom(html, step_index=step_index, name=name)
        path = self._fallback_step_dir(step_index) / (name or "dom.html")
        file_ops.write_text(path, html)
        return path

    def _save_layout(self, layout: str, *, step_index: int) -> Path:
        if self.logger:
            return self.logger.save_layout_graph(layout, step_index=step_index)
        path = self._fallback_step_dir(step_index) / "layout_graph.txt"
        file_ops.write_text(path, layout)
        return path

    def _save_screenshot(self, payload: bytes, *, step_index: int, name: str | None = None) -> Path:
        if self.logger:
            return self.logger.save_screenshot(payload, step_index=step_index, name=name)
        path = self._fallback_step_dir(step_index) / (name or "screenshot.png")
        file_ops.write_bytes(path, payload)
        return path

    def _fallback_step_dir(self, step_index: int) -> Path:
        step_dir = self._artifact_base / f"step_{step_index:03d}"
        step_dir.mkdir(parents=True, exist_ok=True)
        return step_dir

    @staticmethod
    def _timestamp() -> str:
        return datetime.now(UTC).isoformat()
