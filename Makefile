PY := ./.venv/bin/python

.PHONY: all qc dag reproduce causal bias synthesis clean

all:
	./run.sh

qc:        ; $(PY) 01_digitize_qc.py
dag:       ; $(PY) dag.py
reproduce: ; $(PY) 02_reproduce.py
causal:    ; $(PY) 03_causal_models.py
bias:      ; $(PY) 04_bias_analysis.py
synthesis: ; $(PY) 05_synthesis.py

clean:
	rm -rf results synthesis.md dag.dot __pycache__
