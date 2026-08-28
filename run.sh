#!/usr/bin/env bash
# One-command local bring-up: venv -> deps -> seed -> traffic -> API + console.
set -euo pipefail
cd "$(dirname "$0")"

PY=${PYTHON:-python3}

# Virtualenv directory. Override to reuse an existing one, e.g.
#   VENV=.respo ./run.sh
VENV=${VENV:-.venv}

if [ ! -d "$VENV" ]; then
  echo "==> creating virtualenv at $VENV"
  "$PY" -m venv "$VENV"
else
  echo "==> reusing existing virtualenv at $VENV"
fi

if [ ! -f "$VENV/bin/activate" ]; then
  # Windows layout (Git Bash / MSYS)
  if [ -f "$VENV/Scripts/activate" ]; then
    # shellcheck disable=SC1091
    source "$VENV/Scripts/activate"
  else
    echo "error: no activate script found in $VENV" >&2
    exit 1
  fi
else
  # shellcheck disable=SC1091
  source "$VENV/bin/activate"
fi

echo "==> installing dependencies"
pip install -q --upgrade pip
pip install -q -r requirements.txt

[ -f .env ] || { cp .env.example .env; echo "==> wrote .env (mock provider)"; }

echo "==> warm-starting the router from offline preferences"
python -m scripts.seed

echo "==> generating simulated traffic with ground-truth labels"
python -m scripts.simulate --n 200 --label

echo "==> starting gateway on :8000 and console on :8501"
uvicorn controlplane.app:app --host 0.0.0.0 --port 8000 &
API_PID=$!
# Stop the background API if the console exits or we get Ctrl-C.
trap 'kill $API_PID 2>/dev/null || true' EXIT INT TERM

streamlit run dashboard/app.py --server.port 8501 --server.address 0.0.0.0
