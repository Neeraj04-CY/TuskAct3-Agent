# YC Pitch (2-Minute Read)

## 1. Problem

Modern autonomy teams spend more time proving reliability than building features. Tool-chaining LLM agents across browsers, APIs, and code execution requires:
- Deterministic replays for investors and security teams.
- Audit-ready artifacts that survive without access to private sandboxes.
- Fast iteration on new skills without babysitting infrastructure.

## 2. Solution

EIKON Engine behaves like a junior engineer that can plan, execute, and learn new skills. Key differentiators:
- **Deterministic demo harness**: `run_quick_demo.(sh|ps1)` reproduces the Heroku login run, refreshes docs assets, and emits a summarized brief.
- **Skill + memory fusion**: Strategist ⇄ Worker loops automatically write wins/losses into long-term memory for tool ranking.
- **Evidence-first UX**: Every run ships screenshots, DOM dumps, JSON traces, and a markdown summary that plug directly into GitHub Pages.

## 3. Traction / Proof Today

- Renders a complete login workflow with screenshots + DOM snapshots using Playwright dry runs (safe by default).
- Browser worker, Strategist planner, and memory driver already wired with tests and CLI entry points.
- Docs site plus reviewer guide walks investors through proof in five minutes.

## 4. Why Now

- YC, seed funds, and enterprise buyers require verifiable autonomy before pilots.
- Deterministic offline demos lower friction for regulated customers.
- Tool marketplaces and API ecosystems are exploding; agents that can self-discover and grade tools will dominate.

## 5. Ask / Next Steps

- **Funding**: $1.5M pre-seed to scale skill discovery and run agents against real partner workflows.
- **Intros**: Security-conscious design partners in fintech, devtools, and customer support.
- **Hiring**: Founding engineer with deep browser automation experience.

_See `docs/quick_video.md` for the companion narrated walkthrough and `README.md` for setup._
