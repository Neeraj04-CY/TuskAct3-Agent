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

# Run the CLI
python -m src.cli.main "Scrape the latest AI news and summarize it."
```

The CLI will:

1. Call the **Workflow Engine** with your natural language task.
2. Internally call the **Strategist** to create a `WorkflowObject`.
3. Execute via the **Worker** (and in later versions, use Memory + Skills + Discovery).
4. Stream logs and produce a final result.

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

## License

TBD.
