#!/usr/bin/env bash
# Create .venv, install abhack, and run the example experiments.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-python3}"
if ! command -v "$PYTHON" >/dev/null 2>&1; then
  PYTHON=python
fi

if [ ! -d .venv ]; then
  echo "【setup】 create .venv"
  "$PYTHON" -m venv .venv
fi

# shellcheck disable=SC1091
source .venv/bin/activate

echo "【setup】 install abhack into .venv"
python -m pip install -U pip
python -m pip install -e .

RUNS=(demo scores_mix hack sticky_small sticky_full redraw ar1 click)
if [ "$#" -gt 0 ]; then
  RUNS=("$@")
fi

for name in "${RUNS[@]}"; do
  conf="exp/runs/${name}.toml"
  if [ ! -f "$conf" ]; then
    echo "missing $conf" >&2
    exit 1
  fi
  case "$name" in
    demo|scores_mix)
      echo
      echo "【run】 $conf"
      python exp/run.py "$conf"
      ;;
    *)
      echo
      echo "【study】 $conf"
      python exp/study.py "$conf"
      ;;
  esac
done

echo
echo "【done】 outputs are under exp/out/<name>/"
echo "         activate later with:  source .venv/bin/activate"
