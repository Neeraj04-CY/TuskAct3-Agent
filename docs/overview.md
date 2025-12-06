# Overview

EIKON Engine packages a Strategist, Planner, and Browser Worker into a reproducible autonomy stack that runs entirely on your laptop. Strategist V2 converts natural language goals into guarded browser plans, the Browser Worker executes those plans deterministically (dry-run friendly), and Agent Memory feeds future plans with selector and behavior hints. The entire stack is optimized for internships/YC diligence: everything runs offline against a canned HTML app so reviewers can verify logs and artifacts without waiting on flaky endpoints.

Three design choices make the system portfolio-ready:

1. **Deterministic runs** — Plans always target the local `examples/demo_local_testsite/login.html` page and use a fallback screenshot renderer, so CI and demo scripts never leave your machine.
2. **Transparent learning loops** — Strategist telemetry captures reward traces, behavior predictions, and adaptive planner events. Each run folder contains markdown/PDF summaries plus JSONL traces for auditors.
3. **Memory + behavior fusion** — The new `BehaviorLearner` merges reward histories, repair counts, and selector bias into the existing AgentMemory hints so replans inherit both structural and experiential context.

### Stability Loop

Each run now flows through `StabilityMonitor`, which compares reward baselines, confidence, repairs, duration, DOM fingerprints, and repeated failure signatures against the historical record. The monitor emits `stability_report.json` and `stability_report.md` beside the normal summaries and stores long-term aggregates in `artifacts/stability/history.json`. AgentMemory ingests those metrics so the Strategist can bias selector choice and retry budgets based on the stability trend instead of treating every login as a fresh start.

These decisions keep the codebase close to the current frontier of autonomous agents while still being easy to reason about during interviews or accelerator screenings.
