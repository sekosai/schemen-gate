.PHONY: check examples modal-canary paper proofs release-check setup test

setup:
	./scripts/setup.sh test

test:
	.venv/bin/python -m pytest -q

examples:
	.venv/bin/python examples/gate_numpy.py

release-check:
	python3 scripts/release_check.py

proofs:
	lake build
	./scripts/check_lean_claims.sh

paper:
	cd paper && latexmk -pdf -interaction=nonstopmode -halt-on-error cdp.tex
	mkdir -p output/pdf
	cd paper/split && latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=../../output/pdf binary-activation-core.tex
	cd paper/split && latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=../../output/pdf binary-activation-transformers.tex

modal-canary:
	../../scripts/modal.sh canary

check: release-check test
