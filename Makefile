## Top-level Makefile for grn_confound_audit.

PYTHON ?= python3
PIP    ?= pip

.PHONY: help install sims pert synth sweeps runtime figures consistency all clean

help:  ## Show available targets
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  %-14s %s\n", $$1, $$2}'

install:  ## Install the package in editable mode
	$(PIP) install -e .

sims:  ## Run the simulation tool-validation benchmark
	PYTHONPATH=. $(PYTHON) -u scripts/run_simulation_benchmarks.py

pert:  ## Re-run the perturbation validation (Section A.11)
	PYTHONPATH=. $(PYTHON) -u scripts/run_perturbation_validation.py

synth:  ## Re-run the cross-class synthesis on real artefacts
	PYTHONPATH=. $(PYTHON) -u scripts/run_cross_class_synthesis_real.py

sweeps:  ## Run hyperparameter sensitivity sweeps
	PYTHONPATH=. $(PYTHON) -u scripts/run_sensitivity_sweeps.py

runtime:  ## Run wall-clock / RSS benchmark
	PYTHONPATH=. $(PYTHON) -u scripts/run_runtime_benchmarks.py

PAPER_FIG_DIR ?= ../subproject_merged_B_confound_bias_audit/paper/figures

figures:  ## Regenerate figures 1-6 from CSV/JSON outputs AND sync to paper dir
	PYTHONPATH=. $(PYTHON) -u scripts/generate_figures.py
	@if [ -d "$(PAPER_FIG_DIR)" ]; then \
		cp figures/*.pdf "$(PAPER_FIG_DIR)/" && \
		echo "Synced figures to $(PAPER_FIG_DIR)/"; \
	else \
		echo "(skip sync: $(PAPER_FIG_DIR) not present)"; \
	fi

consistency:  ## Run paper <-> code consistency check
	PYTHONPATH=. $(PYTHON) -u scripts/check_paper_code_consistency.py

all: sims pert synth sweeps runtime figures consistency  ## Run everything

clean:  ## Remove generated benchmark/figure outputs
	rm -rf data/benchmarks/*.csv data/benchmarks/*.json figures/*.pdf
