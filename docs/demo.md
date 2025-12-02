# EIKON Engine Demo Guide

This page explains how to walk through the static demo hosted from the `docs/` folder.

## Cards Overview

1. **Open Login** – Shows the navigation-only step captured from `examples/heroku_login/run_sample/result.json`. The screenshot lives beside it in `docs/artifacts/heroku_sample/heroku_login.png`.
2. **Perform Login** – Highlights the credential fill + submit actions. The DOM snapshot (`dom.html`) is linked so viewers can confirm the secure banner.
3. **Extract Content** – Demonstrates the final evidence capture: screenshot + DOM extraction listed in `result.json` `steps[5:]`.

Each card links to the raw artifact sitting in `docs/artifacts/heroku_sample/`, which GitHub Pages serves directly. Updating the demo is as simple as copying a new `runs/<timestamp>` folder via the helper script (see `scripts/generate_demo_assets.py`).

### Visual placeholders

![Hero Preview](artifacts/heroku_sample/heroku_login.png)

> Use the same screenshot as a stand-in for the hero section when taking mockups. Replace it with a freshly generated frame before a live presentation.

![Cards Preview](artifacts/heroku_sample/heroku_login.png)

> This placeholder highlights the card grid spacing; feel free to swap with a tailored collage.

## Refreshing Artifacts

1. Run the CLI goal locally (see README instructions).
2. Point `scripts/generate_demo_assets.py` at the new run folder to populate `docs/artifacts/<slug>/`.
3. Commit the updated artifacts; CI will publish them to GitHub Pages automatically.

## What to show in an interview

- Hero screenshot of the premium landing page (use the placeholders above if nothing newer is available).
- JSON viewer loaded with `artifacts/heroku_sample/result.json` to prove transparency.
- Artifacts viewer showing both the DOM and screenshot evidence.
- Terminal shot of `pytest -q` passing plus the command used to generate the demo run.
