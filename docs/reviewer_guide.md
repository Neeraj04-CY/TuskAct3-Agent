# Reviewer Guide

1. **Start here** – open `README.md` for the elevator pitch and badges, then scroll down to the “What to look at” section.
2. **Artifacts** – visit `docs/artifacts/heroku_sample/run_summary.md` (already committed) for a human-readable brief. The PDF lives next to it if you prefer attachments.
3. **Evidence chain** – inspect:
   - `docs/artifacts/heroku_sample/step_001/step.json`
   - `docs/artifacts/heroku_sample/step_001/screenshot.png` (or the first screenshot in that folder)
   - `docs/artifacts/heroku_sample/trace.jsonl`
   - `docs/artifacts/heroku_sample/steps.jsonl`
   - `docs/artifacts/heroku_sample/stability_report.md`
4. **JSON trace** – open `docs/json_viewer.html` in a browser; it automatically loads `heroku_sample/result.json`.
5. **Live site** – run `python -m http.server 9000 --directory docs` and browse `http://localhost:9000` to view the static experience exactly as GitHub Pages renders it.
6. **Tests** – execute `pytest` (or `pytest tests/integration/test_quick_demo.py` for the smoke) to prove the CLI still generates screenshots.

Follow those steps and you can assess autonomy, reliability, and documentation quality in under five minutes.
