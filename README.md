# EIKON ENGINE

> “An Autonomous Self-Growing AI System with Tool Discovery, Skill Learning, and Recursive Improvement.”

EIKON ENGINE is a modular AI framework where the system behaves like a junior engineer that:

- Plans tasks
- Executes tasks
- Learns new tools
- Upgrades its intelligence
- Stores memory
- Fixes its own mistakes
- Expands itself over time

The long-term goal is a self-growing AI platform capable of becoming a full agent ecosystem.

---

## Vision

EIKON has 7 core pillars:

1. **Strategist (Planner)** — Turns natural language into a multi-step structured plan.
2. **Worker (Executor)** — Executes plans: browser automation, APIs, code, shell, artifacts.
3. **Tool Discovery Engine** — Autonomously discovers, validates, and packages tools into skills.
4. **Skill Library** — Stores reusable tools (skills) as versioned modules.
5. **Memory Engine** — Long-term memory: successes, failures, tool performance, preferences.
6. **Workflow Engine** — Orchestrates Strategist, Worker, Memory, and Skills for each task.
7. **Error Recovery + Self-Debugger** — Detects failures, retries, rewrites, and improves.

---

## Roadmap

EIKON v1 — **Basic Running Agent**

- CLI
- Strategist v1
- Worker v1
- Basic logging
- No memory (or minimal)

EIKON v2 — **Add Memory + Skills**

- Memory engine
- Skill loading
- Basic Tool Discovery

EIKON v3 — **Browser Automation**

- Playwright actions
- DOM parsing
- Screenshot logging

EIKON v4 — **Intelligent Tool Discovery**

- API scanning
- Learning new tools
- Versioning skills

EIKON v5 — **Multi-Agent Behavior**

- Strategist
- Worker
- Debugger

EIKON v6 — **Self-Debugging**

- Rewrite failing code
- Run static analysis
- Fix broken workflows

EIKON v7 — **Recursive Improvement**

- Worker creates new functions
- Strategist stores improvements
- Engine becomes self-growing

EIKON v8 — **Extension System**

- User-defined plugins
- New agent types
- Skill marketplace

EIKON v9 — **Automated DevOps**

- Code generation
- Testing
- Deployment automation

EIKON v10 — **Full AI Operating System**

- Multi-agent OS
- Background tasks
- Real-time monitoring
- Autonomous learning

---

## Engineering Standards

When generating code, always follow this standard:

- modular  
- type-annotated  
- documented  
- small classes  
- small functions  
- decoupled components  
- no monolithic files  
- use dependency injection  
- add tests  
- create interfaces  
- predictable naming  
- follow SOLID principles  

These principles are enforced across:

- Strategist
- Worker
- Discovery
- Skills
- Memory
- Workflow
- Debugger

---

## Repository Structure

```text
eikon-engine/
│
├── src/
│   ├── strategist/
│   ├── worker/
│   ├── discovery/
│   ├── skills/
│   ├── memory/
│   ├── workflow/
│   └── cli/
│
├── configs/
├── tests/
└── README.md
```

See inline module docstrings for details.

---

## Quick Start (Planned)

```bash
# Install dependencies
pip install -r requirements.txt

# Run the demo goal
python scripts/run_demo_goal.py "Scrape the latest AI news and summarize it."
```

This script will:

1. Load Planner v3 and Strategist defaults from `config/settings.yaml`.
2. Generate a BrowserWorker plan for your natural language task.
3. Execute it end-to-end with `BrowserWorkerV1` (Playwright when enabled).
4. Persist DOM snapshots, screenshots, and a `run_summary.json` inside `runs/<timestamp>/`.

---

## BrowserWorker Demo

Use the BrowserWorker demo script to validate dry-run safeguards and live Playwright automation:

```bash
# 1) Install project deps (one time)
pip install -r requirements.txt

# 2) Install Playwright browsers (one time)
python -m playwright install

# 3) Allow browser access if you want to hit external URLs
set EIKON_ALLOW_EXTERNAL=1
set EIKON_ALLOW_SENSITIVE=1

# 4) Optional: bypass dry-run safeguards for real browser control
set PLAYWRIGHT_BYPASS_DRY_RUN=1

# 5) Run the demo (prints standalone worker + Strategist/TaskOrchestrator runs)
python scripts/run_browser_demo.py
```

- Without `PLAYWRIGHT_BYPASS_DRY_RUN=1`, the script shows only the dry-run payload.
- With the env var set, the worker launches Playwright headlessly, captures screenshots, and extracts the DOM from `examples/demo_local_testsite/login.html`.
- The Strategist demo issues natural-language instructions that the BrowserWorker converts into navigate/fill/click operations.

The script prints the direct worker output first and then the TaskOrchestrator transcript so you can inspect both flows side by side.

---

## Live Demo

GitHub Pages can publish the static `docs/` site directly from this repo. Once enabled, share:

```
https://<github-username>.github.io/<repo-name>/
```

GitHub automatically serves the `docs/index.html` landing page plus all artifacts under `docs/artifacts/`.

## How to Reproduce the Demo Locally

Use a single shell session and run the exact commands below:

```bash
# Windows PowerShell
python -m venv .venv && .venv\Scripts\activate

# macOS/Linux
python -m venv .venv && source .venv/bin/activate

pip install -r requirements.txt
playwright install
python scripts/run_demo_goal.py "Log in to https://the-internet.herokuapp.com/login with tomsmith / SuperSecretPassword! and capture the Secure Area screenshot."
```

Artifacts will be written to `runs/<timestamp>/`. Copy the desired run into `docs/artifacts/heroku_sample/` (or any new slug) before pushing so GitHub Pages publishes the same evidence anyone can reproduce locally.

## What to Show to a Recruiter / YC Interview

- Screenshot from `docs/artifacts/heroku_sample/heroku_login.png` displayed on the live site.
- Plan or run trace JSON (`docs/artifacts/heroku_sample/result.json`).
- `runs/<timestamp>/run_summary.json` or CLI log snippet proving end-to-end autonomy.
- `pytest -q` output demonstrating the safety net before the demo.

---

## License

TBD.
