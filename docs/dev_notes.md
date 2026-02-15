# Dev Notes

## Python Runtime Lock (January 2026)

- **Python 3.10.x is mandatory.** Playwright + Chromium crash and silently break selector resolution on CPython 3.11 and especially 3.13. BrowserWorker now asserts `sys.version_info < (3, 11)` so missions refuse to launch on unsupported runtimes.
- **Installation guidance.** Use `pyenv install 3.10.7 && pyenv local 3.10.7` or the repo's `.venv` helpers before running `pip install -r requirements.txt && python -m playwright install`.
- **Diagnostics baked in.** Every BrowserWorker session startup logs Python executable, interpreter version, Playwright version, Chromium executable path, headless flag, and OS. If a mission fails with `BrowserStartupError`, inspect the mission logs for the exact startup payload.
- **Failure contract.** `BrowserStartupError` is raised whenever Playwright is disabled, not installed, or Chromium/Context/Page creation fails. Missions halt immediately so Strategist/skills never run without a real browser session.

Stick to the pinned runtime so v4 learning/skill scoring work stays deterministic.
