#!/usr/bin/env bash
# End-to-end pipeline: raw CSVs -> QC -> DAG -> reproduce -> causal -> bias -> synthesis.md
# Usage: ./run.sh          (creates .venv and installs deps on first run)
set -euo pipefail
cd "$(dirname "$0")"

PY=./.venv/bin/python
if [ ! -x "$PY" ]; then
  echo ">> creating virtual environment (.venv)"
  if command -v uv >/dev/null 2>&1; then
    uv venv --python 3.13 .venv
    uv pip install --python "$PY" -r requirements.txt
  else
    python3 -m venv .venv
    "$PY" -m pip install --upgrade pip
    "$PY" -m pip install -r requirements.txt
  fi
fi

echo ">> [1/6] transcription QC"      && "$PY" 01_digitize_qc.py
echo ">> [2/6] DAG / identifiability" && "$PY" dag.py
echo ">> [3/6] reproduce published"   && "$PY" 02_reproduce.py
echo ">> [4/6] causal models"         && "$PY" 03_causal_models.py
echo ">> [5/6] bias analysis"         && "$PY" 04_bias_analysis.py
echo ">> [6/6] synthesis"             && "$PY" 05_synthesis.py
echo
echo ">> done. See synthesis.md, dag.dot, and results/*.json"
