# Quick Video Script (≤2 minutes)

1. **Opening (15 seconds)**
   - Show the hero GIF (`docs/assets/demo.gif`) inside the README.
   - Narration: “This is EIKON Engine — a strategist + browser worker that plans, executes, and learns entirely offline.”

2. **Terminal proof (35 seconds)**
   - Run `./run_quick_demo.sh` (or the PowerShell equivalent) and keep the terminal visible until it copies the run into `docs/artifacts/heroku_sample`.
   - Call out the deterministic goal (`examples/demo_local_testsite/login.html`) and that the worker stayed in dry-run mode (no external calls, no API keys).

3. **Artifact walkthrough (45 seconds)**
   - Open `docs/artifacts/heroku_sample/run_summary.md` in VS Code preview.
   - Scroll through the sections highlighting: completion status, planner highlights, repairs, and the list of evidence files.
   - Switch to the JSON viewer (`docs/json_viewer.html`) and show `result.json` for reviewers who prefer raw traces.

4. **Learning hook (25 seconds)**
   - Point at the “Behavior prediction” line in the summary or `run_summary.json`.
   - Explain how repeated runs update selector bias / suggested subgoals, and how that feeds back into Strategist V2.

5. **Closing (10 seconds)**
   - Mention that everything (tests, docs, demo artifacts) refreshes via CI + Pages, so the reviewer can clone the repo and rerun the exact demo.
