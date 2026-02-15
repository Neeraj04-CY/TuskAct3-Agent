# Phase 3.1 — Trace Replay Architecture

## Goals
- Deterministic re-execution of a recorded mission trace without planner/LLM involvement.
- Provide a CLI (`python -m eikon_engine.replay ...`) that rebuilds the observed browser/session state, replays actions step-by-step, and emits replay artifacts.
- Detect and report divergences early with actionable diagnostics.

## Key Concepts
- **Trace Source**: `ExecutionTrace` objects persisted via `trace.json` (see `eikon_engine/trace/models.py`). These contain ordered `subgoal_traces`, `actions_taken`, skill usages, artifacts, and page intents.
- **Replay Session**: A reconstructed browser worker (`BrowserWorker`) configured in deterministic mode (no strategist/planner). It receives exact actions taken from the trace and reproduces them.
- **Determinism Guarantees**: No planner invocations, no dynamic intent capture, no retries beyond what the trace already contains. Replay stops immediately on mismatch.

## Module Layout
```
eikon_engine/replay/
  __init__.py
  replay_engine.py      # Core orchestration + deterministic runner
  replay_cli.py         # Arg parsing + CLI entrypoint for `python -m eikon_engine.replay`
```

### `ReplayEngine`
Responsibilities:
1. Load trace via `ExecutionTraceSerializer` (existing `read_trace`).
2. Prepare a deterministic `BrowserWorker` instance:
   - Headless toggle from CLI.
   - Disable planner hooks, logging noise.
   - Ensure `demo_mode=False`, `show_browser` derived from `--headless`.
3. Iterate over `subgoal_traces`:
   - For each `SubgoalTrace`, iterate its `actions_taken` list (ordered by `sequence`).
   - Translate recorded actions into browser worker `action` payloads (navigate, click, type, screenshot, etc.).
   - Invoke a deterministic executor that replays exactly one action and collects output (screenshots, DOM snapshots, etc.).
4. Skills: if the trace recorded a `skill_used` for the subgoal, re-run the same skill with captured metadata (HTML snapshot from stored artifacts when possible). Skills are deterministic (no randomness), so we feed them DOM/URL context from trace artifacts.
5. Artifacts: store any replay outputs under `replay_artifacts/<mission_id>/subgoal_<n>/...` to prevent overwriting the original mission artifacts.
6. Divergence detection:
   - If browser responses deviate (e.g., HTTP error, missing selector), raise `ReplayDivergenceError` with context.
   - Compare crucial signals (final URL, HTTP status, skill outputs) to the trace; mismatch triggers a failure classification later.
7. Summary: produce `replay_summary.txt` describing replay outcome, divergences (if any), and artifact locations.

### Deterministic Action Runner
- Wrap `BrowserWorker.execute()` with a strict mode that:
  - Accepts a single action at a time.
  - Disallows fallback resolver logic (selector repair) so the action either succeeds exactly as recorded or fails.
  - Captures actual vs expected DOM snapshots for debugging.
- Provide instrumentation hooks that emit `[REPLAY]` console logs for every action and skill.

### Error Handling
- Define `ReplayError` base class + specific `ReplayDivergenceError` with metadata: `subgoal_id`, `action_sequence`, `expected`, `observed`.
- Surface these errors in:
  - CLI exit code (non-zero).
  - `replay_summary.txt` (section: `Replay divergence detected`).
  - Structured JSON snippet (optional) for future automation.

## CLI (`replay_cli.py`)
- Arguments:
  - `--trace PATH` (required): path to `trace.json`.
  - `--headless` boolean flag (default `true`).
  - `--artifacts-dir` optional override (default `replay_artifacts/<mission_id>`).
- Flow:
  1. Parse args, load trace.
  2. Instantiate `ReplayEngine` with `headless=not args.headless`.
  3. Run `engine.replay(trace)`; capture success/failure.
  4. Print `[REPLAY]` logs with progress, final summary path.

## Outputs
- Directory: `replay_artifacts/<mission_id>/subgoal_<idx>/` containing screenshots, DOM snapshots, skill outputs, and `replay_summary.txt` at the root.
- Summary file content:
  - Mission ID, start/end timestamps, headless mode.
  - Action counts, divergence status, reason if failed.
  - Pointer to canonical artifacts.

## Testing Strategy
1. **Unit tests** for `ReplayEngine` using synthetic traces (small fixtures) to ensure:
   - Actions are loaded/executed in order.
   - Divergence detection triggers on mismatch.
2. **Integration tests** using a recorded trace from tests (mock BrowserWorker) verifying CLI flow.
3. **Failure replay demo**: store a known failing trace under `tests/data/traces/...` and assert CLI exit != 0 with the correct `REPLAY_DIVERGENCE` flag.

## Open Questions / Assumptions
- Browser actions in trace must be serializable to deterministic `BrowserWorker` commands; we will extend the trace schema if missing action metadata.
- Skills may require DOM snapshots; we’ll reuse the original artifact DOM, falling back to live page content only if necessary.
- Replay currently assumes a stable external website; to make tests deterministic, we will also support “mock playback” where DOM snapshots feed headless browser responses (future enhancement if needed).
