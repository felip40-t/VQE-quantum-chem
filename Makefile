SHOTS ?= 500
CSV   ?=

.PHONY: help setup hamiltonian run run-shots run-noise run-fake plot notebooks

help:
	@echo "Usage:"
	@echo "  make setup             Create venv and install dependencies"
	@echo "  make hamiltonian       Sanity-check Hamiltonian pipeline"
	@echo "  make run-all           Run all VQE modes"
	@echo "  make run-shots         VQE with shot noise  (SHOTS=500)"
	@echo "  make run-noise         VQE with depolarising noise  (SHOTS=500)"
	@echo "  make run-fake          VQE with IBM Vigo fake backend  (SHOTS=500)"
	@echo "  make plot CSV=<file>   Plot dissociation curve from a saved CSV"
	@echo "  make notebooks         Launch Jupyter Lab"
	@echo ""
	@echo "Override shot count:  make run-shots SHOTS=1000"

setup:
	python -m venv .venv
	.venv/bin/pip install -r requirements.txt

hamiltonian:
	python src/hamiltonian.py

run-all:
	python src/main.py --shots $(SHOTS) --all

run-shots:
	python src/main.py --shots $(SHOTS)

run-noise:
	python src/main.py --noise --shots $(SHOTS)

run-fake:
	python src/main.py --fake-backend --shots $(SHOTS)

plot:
ifndef CSV
	$(error CSV is not set. Usage: make plot CSV=<filename>)
endif
	python src/plot_dissociation.py --csv $(CSV)

notebooks:
	jupyter lab
