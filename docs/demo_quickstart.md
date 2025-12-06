# Demo Quickstart

Follow these steps to reproduce the Heroku login demo and publish the exact artifacts that ship with the repo.

## Fast path — single command

### Windows PowerShell

```powershell
Set-ExecutionPolicy -Scope Process RemoteSigned
./run_quick_demo.ps1
```

### macOS / Linux (bash / zsh)

```bash
chmod +x run_quick_demo.sh
./run_quick_demo.sh
```

Both scripts will create/refresh `.venv`, install requirements, execute the deterministic autonomy demo, copy the freshest `artifacts/autonomy_demo_*` folder into `docs/artifacts/heroku_sample/`, rebuild `docs/assets/demo.gif`, and regenerate `docs/artifacts/heroku_sample/run_summary.(md|pdf)`.

## Manual path — step-by-step

### Windows PowerShell

```powershell
# 1) Create virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 2) Install dependencies and browsers
pip install -r requirements.txt
python -m playwright install chromium

# 3) Run the safe quick demo (dry-run)
set EIKON_ALLOW_EXTERNAL=0
set EIKON_ALLOW_SENSITIVE=0
set PLAYWRIGHT_BYPASS_DRY_RUN=0
python run_autonomy_demo.py --summary tmp_run\quick_autonomy.json

# 4) Copy outputs into docs
python scripts\generate_demo_assets.py --run artifacts\autonomy_demo_* --slug heroku_sample --overwrite
python scripts\make_demo_gif.py --run docs\artifacts\heroku_sample --output docs\assets\demo.gif --fps 2
python scripts\generate_run_summary.py --run docs\artifacts\heroku_sample --title "Autonomy Demo (dry-run)"

# 5) Preview docs locally
python -m http.server 9000 --directory docs
Start-Process http://localhost:9000
```

### macOS / Linux (bash / zsh)

```bash
# 1) Create virtual environment
python -m venv .venv
source .venv/bin/activate

# 2) Install dependencies and browsers
pip install -r requirements.txt
python -m playwright install chromium

# 3) Run the safe quick demo (dry-run)
export EIKON_ALLOW_EXTERNAL=0
export EIKON_ALLOW_SENSITIVE=0
export PLAYWRIGHT_BYPASS_DRY_RUN=0
python run_autonomy_demo.py --summary tmp_run/quick_autonomy.json

# 4) Copy outputs into docs
python scripts/generate_demo_assets.py --run artifacts/autonomy_demo_* --slug heroku_sample --overwrite
python scripts/make_demo_gif.py --run docs/artifacts/heroku_sample --output docs/assets/demo.gif --fps 2
python scripts/generate_run_summary.py --run docs/artifacts/heroku_sample --title "Autonomy Demo (dry-run)"

# 5) Preview docs locally
python -m http.server 9000 --directory docs
open http://localhost:9000
```

> **Note:** Replace `artifacts/autonomy_demo_*` with the actual folder printed by `run_autonomy_demo.py`. The helper scripts accept absolute paths too.
