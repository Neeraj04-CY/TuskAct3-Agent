#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
VENV="$ROOT/.venv"
PYTHON=${PYTHON:-python}

if [ ! -d "$VENV" ]; then
  echo "[setup] Creating virtual environment at $VENV"
  "$PYTHON" -m venv "$VENV"
fi

# shellcheck disable=SC1090
source "$VENV/bin/activate"
python -m pip install --upgrade pip
pip install -r "$ROOT/requirements.txt"
python -m playwright install

echo "[setup] Environment ready. Activate with: source $VENV/bin/activate"
