<div align="center">

# TuskAct3 v4.6 — Goal-Driven Autonomous Research Engine

**Status: Finalized Core Engine**

[![CI](https://github.com/Neeraj04-CY/TustAct3-Agent/actions/workflows/ci.yml/badge.svg)](https://github.com/Neeraj04-CY/TustAct3-Agent/actions/workflows/ci.yml)
[![Docs](https://img.shields.io/badge/docs-live-blue)](docs/index.html)
[![Demo Assets](https://img.shields.io/badge/demo-heroku_sample-success)](docs/artifacts.md)

> Goal-driven autonomous research engine with real-browser execution and deterministic evidence for every run.

</div>

## Why it matters

- **Deterministic autonomy** – Strategist + Browser Worker pairings run with Playwright dry-runs by default so YC reviewers can replay with zero external access.
- **Artifact-first storytelling** – Every demo emits screenshots, DOM captures, JSON traces, and a markdown summary that publish directly to GitHub Pages.
- **Tool discovery + memory fusion** – Worker feedback loops update the memory store so the agent ranks skills, retries failures, and keeps improving.

Read the architecture notes in `docs/overview.md` and the YC-ready positioning in `docs/yc_pitch.md`.

---

## Demo at a glance

| Asset | Location | Why it matters |
| --- | --- | --- |
| 📺 **Landing page** | `docs/index.html` | Polished cards showcasing each browser step with CTA buttons for JSON + artifacts. |
| 🧠 **Strategist plan** | `docs/json_viewer.html` (loads `docs/artifacts/heroku_sample/result.json`) | Inspect every chain-of-thought action without running code. |
| 📁 **Artifacts bundle** | `docs/artifacts/heroku_sample/` | Screenshots, DOM, logs, and the generated `run_summary.md` for recruiters. |
| 🎥 **Video script** | `docs/quick_video.md` | 90-second narration ready for Loom / YC video. |
| ✅ **Reviewer checklist** | `docs/reviewer_guide.md` | Five-minute inspection instructions for PMs or partners. |

Need more context? `docs/artifacts.md` explains each file, and `docs/demo_quickstart.md` shows how to refresh them.

---

## One-command quickstart

### Windows PowerShell

```powershell
Set-ExecutionPolicy -Scope Process RemoteSigned
./run_quick_demo.ps1
```

### macOS / Linux

```bash
chmod +x run_quick_demo.sh
./run_quick_demo.sh
```

What this does:

1. Creates/activates `.venv`, installs dependencies, and provisions Playwright browsers.
2. Runs `run_autonomy_demo.py` in deterministic dry-run mode (no external network).
3. Copies the latest `artifacts/autonomy_demo_*` folder into `docs/artifacts/heroku_sample/`.
4. Rebuilds the GIF (`docs/assets/demo.gif`) and `docs/artifacts/heroku_sample/run_summary.md` via `scripts/make_demo_gif.py` and `scripts/generate_run_summary.py`.

When it finishes you can open `docs/index.html` locally or push to GitHub Pages for the exact same experience viewers will see.

---

## Runtime requirements

- **Python 3.10.x only.** Playwright and the BrowserWorker now assert `sys.version_info < (3, 11)` so missions fail fast if you try to run under Python 3.11+ (3.13 is known-broken).
- **Pinned toolchain.** Use `pyenv local 3.10.7` (or the bundled `.venv`) before running demos, then `python -m playwright install` to provision Chromium.
- **Diagnostics.** The BrowserWorker logs the Python executable, Playwright version, Chromium binary, headless mode, and OS once per mission. Check the mission logs if startup fails.

See `docs/dev_notes.md` for founder-facing context on the runtime lock and troubleshooting tips.

---

## Manual setup (if you prefer to step through)

```bash
python -m venv .venv && source .venv/bin/activate  # .venv\Scripts\activate on Windows
pip install -r requirements.txt
python -m playwright install
python run_demo_goal.py --goal "Log into the Heroku test site and capture the Secure Area banner" --summary runs/latest_summary.json
```

Copy the resulting `runs/<timestamp>` folder into `docs/artifacts/<slug>` and update `docs/assets/demo.gif` / `docs/json_viewer.html` paths as needed. The helper script `scripts/generate_demo_assets.py` shows the exact copying rules.

---

## Resume halted missions (v4.1)

- Any `halted` or `ask_human` stop now writes `resume_checkpoint.json` inside the mission artifacts folder with pending subgoals, last intent/url, and capability snapshots.
- Resume a prior run with `python -m eikon_engine.missions.cli --resume <checkpoint-path-or-mission-id> --artifacts-dir artifacts`, which restores the mission id, skips completed subgoals, and replays the lifecycle in the trace.
- Trace summaries gain a Lifecycle section showing `mission_halted`, `resume_loaded`, and `resume_completed` events for auditors.
- Learning logs mark resumed runs as `resumed_success` or `resumed_failure` so analytics can separate fresh vs. resumed performance.

## Quick Heroku Login Demo (Recording-Friendly)

Need a lightweight walkthrough for screen recordings? Run the direct Playwright driver:

```bash
python scripts/heroku_login_demo.py
```

What happens:

- Launches Chromium headfully with Windows-safe flags and opens the Heroku sample login page.
- Autofills the demo credentials (`tomsmith` / `SuperSecretPassword!`) and waits for the "You logged into a secure area!" banner.
- Captures `artifacts_demo/secure_area.png` so you have a crisp slide-ready screenshot.
- Keeps the browser open until you press Enter in the terminal, making it perfect for recording live narration.

Use this when you only need a fast visual proof. The full autonomy missions/agent pipeline remains available through the CLI (`python -m eikon_engine.missions.cli ...`) for end-to-end demos, rollouts, and artifact generation.

---

## v2: Memory & Skills

- **Mission memory ledger** – every autonomy run now persists a `MissionMemory` JSON file under `memory_logs/`. Entries capture mission text, detected URL, status, artifacts path, and the skills that fired. Nothing is lost between runs.
- **Memory reader hooks Strategist V2** – before planning a new subgoal, the Strategist inspects prior memories (matching URLs or login intents) and surfaces reusable skill hints so the agent never starts cold.
- **Reusable skills** – `LoginFormSkill` lives in `eikon_engine/skills/login.py` and drives Playwright directly. `BrowserWorker.run_skill()` hands the live `page` object plus credentials so skills can manipulate the DOM without rebuilding pipelines.
- **Skill-aware execution** – when a remembered login mission repeats, `MissionExecutor` invokes `login_form_skill` ahead of the subgoal loop and records the reuse inside both the artifacts and the mission memory entry.
- **Proof it works** – rerun any mission twice. The first run writes a memory entry; the second detects the prior success, loads the login skill, and completes faster while logging the reused skill.

This fulfills the v2 mandate: *“TuskAct3 remembers past missions and reuses learned skills instead of starting from scratch.”*

## v3: Execution Traces

- **Always-on recorder** – every mission now spins up an `ExecutionTraceRecorder` at launch. The recorder captures mission metadata, subgoal attempts, atomic browser actions, skills fired, and the artifacts that land on disk. Trace payloads are versioned (`trace_version: "v3.1"`) and saved under `traces/trace_<timestamp>_<mission_id>/trace.json`.
- **Structured subgoal + action history** – each retry creates a `SubgoalTrace` with its own attempt number, timestamps, actions, and errors. BrowserWorker streams `ActionTrace` entries with action type, selector/target, masked inputs, outcome, duration, and raw metadata so reviewers can replay every click/fill/wait.
- **Failure + skill forensics** – mission timeouts, planner explosions, retry loops, and secure-area aborts emit `FailureRecord` rows (including retryable flags) while skills push `SkillUsage` entries tied back to the subgoal that invoked them. Login skills, repairs, and future tools all surface in the same `skills_used[]` feed.
- **Persistence-first contract** – mission summaries now embed the trace path, and `run_mission` aborts if the trace cannot be written (`No trace = no mission`). Planning failures and crashes still flush their trace before the exception bubbles, giving you an auditable artifact even when runs die early.
- **Regression guardrail tests** – `tests/trace/test_execution_trace.py` covers creation, retry logging, skill recording, and failure persistence (`test_execution_trace_created`, `test_subgoal_trace_written`, `test_skill_usage_recorded`, `test_trace_persisted_on_failure`). Add them to CI to block regressions.

This fulfills the Phase 1 charter for v3: *“Every mission must emit a replayable, machine-readable execution trace.”*

### v3.1 Trace Stabilization

- **Readable schema** – every trace node now exposes `id`, `type`, `started_at`, `ended_at`, and `duration_ms`, plus docstrings that explain when `ExecutionTrace`, `SubgoalTrace`, `ActionTrace`, `FailureRecord`, `SkillUsage`, and `ArtifactRecord` are created and sealed.
- **Deterministic ordering + completeness** – recorder-enforced ordering sorts subgoals by their first timestamp, actions by execution sequence, and failures by the exact moment they occurred. Completeness checks flag unfinished attempts, missing action statuses, retry failures without reasons, or subgoal-less skill usages, marking traces as `incomplete: true` with top-level warnings when violations occur.
- **Trace summaries** – each persisted trace now includes a sibling `trace_summary.txt` generated solely from the JSON (duration, counts, skill inventory, recovered failures, final status). Demo tooling can link to both the machine-readable and human summaries.

## v4 (In Progress): Learning & Adaptation

TuskAct3 v4 introduces a learning layer that observes completed missions, scores skill effectiveness, and persists memory for future decisions.

Learning is:
- Post-execution only
- Deterministic
- Fully inspectable
- Replay-safe

New scaffolding (Phase 0):
- `learning_logs/<mission_id>.json` written after mission completion with skill usage, failures, confidence, and trace references.
- `eikon_engine/learning` contains the record schema, scorer, recorder, and reader utilities for human-readable inspection.
- Strategist exposes a no-op `learning_hints` hook to accept hints later without altering current behavior.
- **Integrity tests** – `tests/trace/test_trace_integrity.py` runs a mocked mission, ensures action traces exist, and asserts both `trace.json` and `trace_summary.txt` are emitted with the expected version header.

### How TuskAct3 Learns (v4)

- **What updates** – the learning layer only ranks skills per mission-type. Each run appends evidence (success rate, steps saved, observed confidence) to `learning_logs/` and recalculates priorities plus average confidence per skill.
- **What never changes** – strategist logic, mission goals, and worker code paths are untouched by learning. There is no hidden mutation of prompts or plans; biasing simply reorders the `preferred_skills` list that already exists.
- **When it happens** – mission execution finishes, traces persist, and only then does the recorder score the run and update the learning index. Live executions never mutate mid-flight.
- **Why replay stays deterministic** – every trace still captures the exact actions and metadata used during execution. Learning artifacts (`learning_diff.json`, `learning_summary.txt`) are written alongside the mission directory without editing the trace or result payloads, so replays still consume the original immutable data.
- **Guardrails against silent regressions** – artifact diffs list only real metric shifts (priority, confidence, success rate). Empty diffs prove no change, and the summary explains the same numbers in plain English. CI tests assert that learning-only artifacts do not modify traces, so any regression would fail loudly instead of altering missions silently.

### v3.2 Deterministic Replay & Trust Signals

- **Replay CLI (`python -m eikon_engine.replay --trace traces/.../trace.json`)** – rehydrate any mission headlessly or headfully, re-run every recorded action, replay the attached skill, and emit fresh artifacts under `replay_artifacts/<mission_id>/`. Divergences (`selector mismatch`, `skill output mismatch`, etc.) halt execution immediately and surface inside `replay_summary.txt`.
- **Decision attribution log (`trace_decisions.json`)** – every trace now ships with a structured ledger of page intents, skill invocations, subgoal outcomes, and failure metadata plus aggregated confidence and risk flags. Reviewer dashboards can consume this JSON directly without scraping Markdown summaries.
- **Confidence + risk scoring** – the decision report normalizes page-intent confidences, adds `low_confidence`, `failures_recorded`, `retries_detected`, or `trace_incomplete` flags, and classifies failures into human-readable buckets (timeout, planner, strategy violation, retryable, etc.).
- **Canonical mission manifest** – `config/canonical_missions.json` locks the review-ready missions (`yc_listing`, `heroku_secure_area`, `docs_artifact_audit`). Run any of them via `python -m eikon_engine.missions.cli --canonical yc_listing --artifacts-dir artifacts/yc_run_v3` to guarantee identical planner inputs across contributors.
- **Replay tests + failure demo** – `tests/replay/test_deterministic_replay.py` stubs the browser worker to ensure the engine validates per-step behavior and surfaces divergences. Pair it with `python -m eikon_engine.replay --trace ...` to demo a captured failure end-to-end.

## v4 Phase 2 — Human Approval Loop

- **Gated execution** – pass `--require-approval` to `run_mission.py` (or the mission CLI) to pause before risky subgoals. Configure `--approval-timeout` (seconds) and `--auto-approve-low-risk` for low-risk auto-passes. All approval requests land in `approval_request.json` inside the mission artifacts directory and are linked from `trace_summary.txt` and `mission_result.json`.
- **Resolve via CLI** – approve or reject by id with `python -m eikon_engine.approval approve <approval_id>` or `... reject <approval_id>`. The tool scans `artifacts/` (or `logging.artifact_root` from settings) to locate the pending request and writes the decision back with timestamps and resolver info.
- **Trace accountability** – approval requests and resolutions are recorded in the execution trace (`approvals_requested[]`, `approvals_resolved[]`) and show up in summaries, keeping governance auditable alongside capability and learning signals.

## v4 Phase 2 — Aggressive Learning Mode

- **Learning override** – planner outputs are advisory; the executor runs a learning review that can reorder, replace, or skip steps based on the Learning Impact Score (LIS). Decisions are explicit and logged, never silent.
- **Refusal safety** – when LIS drops below the hard floor (default `-0.6`), the mission halts with `refused_by_learning`. No worker actions execute; traces capture the justification and conflict snapshot.
- **Scoring + persistence** – LIS combines historical success/failure, confidence, and recency into a bounded `[-1.0, 1.0]` score and persists to `learning_index.json` for inspection.
- **Trace accountability** – every override emits `learning_event` entries with the original step, decision type, learning score, confidence, and evidence. `learning_diff.json` and `learning_summary.txt` remain alongside the execution trace.
- **Decision explanations** – whenever learning changes execution (override, refusal, or bias-only), the executor writes `learning_decision_explanation.json` next to `mission_result.json` and links it inside `trace_summary.txt` for one-click review.
- **Override engine** – conflicts are detected when planner steps score below the override threshold or ignore high-confidence skills. Decisions span `ACCEPT`, `REORDER`, `REPLACE_WITH_SKILL`, `SKIP`, or `REFUSE`, always with reasons and evidence.
- **Why safer** – the planner is prevented from repeating failing patterns, low-confidence paths are short-circuited, and reviewers get deterministic artifacts describing exactly why learning intervened.

### v3.0 — Capability Registry (Read-Only)

TuskAct3 now maintains an explicit registry of system capabilities, separate from skills and execution logic. Capabilities describe what the agent can do; skills describe how it is done. This phase records capability usage for transparency and future planning without altering runtime behavior. The registry is static, registry-driven, and does not change planner, execution, or learning decisions.

Example explanation artifact:

```json
{
	"mission_id": "mission_20260113_120000",
	"decision_type": "override",
	"learning_impact_score": -0.42,
	"confidence_score": 0.78,
	"triggering_signals": [{"skill": "login_form_skill", "success_rate": 0.42, "failures": 3, "evidence": "learning_override"}],
	"planner_conflict": true,
	"final_resolution": "override_applied",
	"summary": "Learning override applied (learning_override); impact_score=-0.42."
}
```

## Phase 4: Autonomy Guardrails & Budgets

- **Mission budgets** – every run now hydrates an `AutonomyBudget` (defaults: 30 steps / 3 retries / 120s / 0.4 risk). The executor tracks steps, retries, elapsed time, and derived risk per attempt. Exceeding any limit halts the mission with a `HALTED` termination block plus a fully serialized budget snapshot inside both the summary and `mission_result.json`.
- **Safety contracts** – pass `--safety-contract '{"blocked_actions": ["download_file"]}'` or embed the same structure in `config/canonical_missions.json`. The executor inspects every recorded action and will either halt or escalate to `ASK_HUMAN` whenever a blocked action, out-of-policy action, or confirmation-only operation (e.g., `submit_form`, `execute_script`) is attempted.
- **Low-confidence handoffs** – `--ask-on-uncertainty` (or `"ask_on_uncertainty": true` in canonical entries) watches rolling page-intent confidence and the computed risk score. Runs that stay below 0.4 confidence or approach the configured risk cap flip to the new `ask_human` mission status so a reviewer can intervene with full trace evidence.
- **CLI guardrails** – combine granular knobs: `--budget-max-steps 12 --budget-max-duration 45 --budget-max-risk 0.25 --autonomy-budget '{"max_retries":1}' --ask-on-uncertainty --safety-contract '{"allowed_actions":["navigate","screenshot"]}'`. Canonical manifests accept the same fields so review builds stay deterministic.
- **Cost + reason summaries** – every summary now emits `autonomy_budget`, `cost_estimate` (USD, based on steps/retries/time/risk), `reason_summary`, and a rich `termination` payload describing why the agent halted, escalated, or completed. These fields propagate into artifacts, traces, and CLI output so dashboards and reviewers can quantify trust decisions instantly.

---

## Architecture snapshot

- **Strategist (`src/strategist/`)** – turns natural language goals into structured plans with guardrails.
- **Worker (`src/worker/`)** – executes browser/API/code steps, logs DOM + screenshots, and emits telemetry.
- **Memory (`src/memory/`)** – persists successes/failures so Strategist can re-rank tools.
- **Discovery + Skills (`src/discovery`, `src/skills`)** – package reusable capabilities that can be swapped in or discovered automatically.
- **Workflow (`src/workflow/`)** – orchestrates Strategist ⇄ Worker ⇄ Memory and handles retries.

Peek at `docs/overview.md` for the full research brief plus the roadmap for adaptive tool discovery.

---

## Documentation map

- `docs/demo_quickstart.md` – step-by-step reproduction guide (Windows + macOS/Linux).
- `docs/artifacts.md` – how to inspect `steps.jsonl`, `trace.jsonl`, screenshots, and summaries.
- `docs/quick_video.md` – narration outline for a ≤2 minute video.
- `docs/reviewer_guide.md` – five-minute checklist for YC/recruiters.
- `docs/yc_pitch.md` – written pitch covering problem/solution/traction/ask.

Everything links back to this README so reviewers never get lost.

---

## Stability Loop

- `StabilityMonitor` records reward drift, confidence deltas, repair counts, duration trends, DOM fingerprint similarity, and repeated failures after every autonomy run.
- Each invocation of `run_autonomy_demo.py` now emits `stability_report.json` and `stability_report.md` next to the summary files, plus appends to `artifacts/stability/history.json` for multi-run trend analysis.
- Strategist V2 feeds these metrics into `AgentMemory`, so future plans inherit selector bias, repair pressure, and success-rate context directly from the stability loop.

---

## Autonomous Rollout Engine: Self-Improvement Across Runs

- `run_rollout.py --n 20` executes N autonomy episodes (dry-run by default), reusing the same Strategist/AgentMemory/StabilityMonitor instances so learning compounds between attempts.
- Each episode stores `reward_trace`, confidence medians, repair events, planner evolution, behavior predictions, DOM fingerprints, and the generated `stability_report.*` under `artifacts/rollouts/run_{i}/`.
- After all runs, `rollout_summary.json` + `.md` capture reward trendlines (linear regression), confidence medians, repair-pressure trends, repeated failure clusters, behavior-model deltas, memory growth, stability drift, and pass/fail classifications per run.
- Use rollout summaries in reviewer packets to prove the agent improves autonomously without manual resets.

---

## Self-Improvement Loop

- `eikon_engine/replay/experience_replay.py` replays historical autonomy runs, captures low-confidence failures, and feeds them into `StrategistV2.learn_from_past` without touching the browser. Alternate subgoals are rescored with the reward model and merged back into AgentMemory.
- `eikon_engine/replay/curriculum_builder.py` clusters runs by difficulty spikes, repeated failure clusters, stability drift, and DOM similarity so the most fragile scenarios get replayed first.
- Run the full offline curriculum plus reporting pipeline with:

```bash
python run_offline_eval.py --artifacts artifacts --output artifacts/replay --save-hints
```

- Need a smaller pass? Limit history and skip persistence:

```bash
python run_offline_eval.py --limit 3
```

- Every replay cycle emits `replay_summary.json`, per-batch reports, and `improvement_report.json/md` that summarize selector bias shifts, subgoal merges, and skill metrics. When `--save-hints` is set the newly learned selectors/subgoals land in `memory_hints.json` for the next live Strategist run.

---

## Showcase Dashboard & Release Bundle

- `python run_demo.py` runs a single autonomy episode, prints a human-readable verdict, spins up the FastAPI dashboard (`dashboard/server.py`), and opens it in your browser. Pass `--no-dashboard` if you only need artifacts.
- Dashboard views: last-run summary, reward/confidence charts, repair timeline, planner evolution log, stability drift history, repeated failure clusters, memory growth, behavior predictions, plus a DOM + screenshot viewer for every step. All data comes from `artifacts/autonomy/run_*` so you can refresh without restarts.
- `python build_release.py --rollout 3` reruns the fast autonomy demo, optionally triggers a short rollout, copies everything into `release_bundle/release_<timestamp>/`, generates `demo.gif`, `charts.html`, dashboard snapshots, and exports `README.md` + `docs/yc_pitch.md` to both Markdown and PDF.
- Ship the `release_bundle/` folder to YC reviewers so they can inspect artifacts, metrics, and videos without reproducing the run locally.

---

## API Usage

- Launch the service with `uvicorn api_server:app --reload`. Endpoints include `/run` (full autonomy demo), `/plan`, `/predict`, `/last_run`, and `/artifacts/{path}`.
- **curl example**:

```bash
curl -X POST http://localhost:8000/run \
	-H "Content-Type: application/json" \
	-d '{"goal":"Generate dashboard summary","execute":false}'
```

- **Python example**:

```python
import requests

resp = requests.post("http://localhost:8000/run", json={"goal": "Demo"})
resp.raise_for_status()
summary = resp.json()["summary"]
print("Status:", summary["reason"])
```

- `/plan` returns the raw planner targets, `/predict` calls the BehaviorLearner for a DOM fingerprint, and `/artifacts/...` streams run artifacts for UI embeddings.

---

## Skill Plugin Overview

- Skills now inherit from the async `Skill` base (`eikon_engine/skills/base.py`) and expose a single `execute(context)` coroutine. Context always includes the Playwright `page` plus mission-provided parameters (like credentials).
- `LoginFormSkill` is the first reusable capability; `FormFillSkill` and `ExtractSkill` remain available for compatibility and return structured status payloads.
- The lightweight `eikon_engine/skills/registry.py` maps names to skill instances and powers both the legacy `SkillRegistry` shim and the new memory-aware planner hints.
- `BrowserWorker.run_skill()` shares its live session so skills can drive the page headlessly or headfully without extra plumbing.

---

## Testing & CI

```bash
pytest
```

GitHub Actions (`.github/workflows/ci.yml`) runs lint + tests on every push and blocks merges if the deterministic demo summary goes stale.

---

## Contributing

1. `scripts/setup_dev.sh` (or the PowerShell commands in `docs/demo_quickstart.md`) bootstraps dependencies.
2. Open an issue describing the improvement.
3. Submit a PR with screenshots of new artifacts + updated summaries.

See `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, and `AUTHORS.md` for the canonical guidelines.

---

## License

Apache-2.0 – see `LICENSE`.
