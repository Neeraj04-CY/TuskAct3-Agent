#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
VENV="$ROOT/.venv"
PYTHON=${PYTHON:-python}

if [ ! -d "$VENV" ]; then
  echo "[quick-demo] Creating virtualenv"
  "$PYTHON" -m venv "$VENV"
fi
# shellcheck disable=SC1090
source "$VENV/bin/activate"
python -m pip install --upgrade pip >/dev/null
pip install -r "$ROOT/requirements.txt" >/dev/null
python -m playwright install >/dev/null

export EIKON_ALLOW_EXTERNAL="0"
export EIKON_ALLOW_SENSITIVE="0"
export PLAYWRIGHT_BYPASS_DRY_RUN="0"
mkdir -p "$ROOT/tmp_run"
python "$ROOT/run_autonomy_demo.py" --summary "$ROOT/tmp_run/quick_autonomy.json"

LATEST_RUN=$(python - <<PY
from pathlib import Path
root = Path(r"$ROOT") / "artifacts"
folders = sorted([p for p in root.glob('autonomy_demo_*') if p.is_dir()], key=lambda p: p.stat().st_mtime)
print(folders[-1] if folders else "")
PY
)
if [ -z "$LATEST_RUN" ]; then
  echo "[quick-demo] Unable to locate run folder under artifacts/autonomy_demo_*"
  exit 1
fi
TARGET="$ROOT/docs/heroku_sample"
rm -rf "$TARGET"
python - <<PY
from pathlib import Path
import shutil
root = Path(r"$ROOT")
latest = Path(r"$LATEST_RUN")
target = root / "docs" / "heroku_sample"
target.parent.mkdir(parents=True, exist_ok=True)
shutil.copytree(latest, target)
PY

python "$ROOT/scripts/make_demo_gif.py" --run "$TARGET" --output "$ROOT/docs/assets/demo.gif" --fps 2
python "$ROOT/scripts/generate_run_summary.py" --run "$TARGET" --title "Autonomy Demo (dry-run)"

echo "[quick-demo] Refreshed docs/heroku_sample and docs/assets/demo.gif"
