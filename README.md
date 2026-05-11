# grn_confound_audit

**Unified three-class confound audit for single-cell gene regulatory networks.**

This is the v0.2 release for the BMC Bioinformatics revision. Relative to v0.1
it adds (i) a per-edge degree-preserving topological FDR with optional GPD
tail extension, (ii) a real cross-class synthesis function (pairwise observed
vs.\ expected-under-independence agreement, $\phi$, $\chi^2$, bootstrap CI on
three-way joint retention), (iii) end-to-end CLI wiring for `--metadata`,
`--counts`, `--edge-features`, `--n-null-replicates`, `--null-tail-method`,
`--proximity-null-families`, and (iv) a ground-truth simulator for tool
validation.

## What it audits

| Class | What it tests | Per-edge output | Per-network output |
|-------|---------------|-----------------|--------------------|
| **1 -- Technical** | Donor / batch / assay-method leakage in derived edge scores via the **Artefact Sensitivity Index** (ASI) and per-cell leakage classifiers | ASI, blacklist flag | leakage AUC per covariate |
| **2 -- Genomic proximity** | Inflation of top-k by genomically proximate TF/target pairs, under **three null families** (source-preserving, same-chromosome-conditional, degree-preserving) plus **hub-degree-stratified** breakdown | distance, proximity flag | enrichment + p/q values per null family |
| **3 -- Topological** | Degree-preserving null with **per-edge BH FDR** at $B \geq 2{,}000$ replicates + optional **GPD tail extension** | empirical & combined per-edge p-value, q-value | global $z$, swap diagnostics, valid-block flag |
| **Cross-class** | Pairwise observed vs.\ expected-under-independence agreement; $\phi$ + $\chi^2$; bootstrap 95% CI on three-way joint retention | joint pass flag | full synthesis CSV |

## Install

```bash
pip install -e .
# or
conda env create -f environment.yml && conda activate grn-confound-audit
```

## End-to-end CLI

```bash
grn-confound-audit run \
  --edges data/edges.csv \
  --gene-coords data/coords.tsv \
  --metadata data/cells.csv \
  --counts data/counts.parquet \
  --scores-balanced data/balanced.csv \
  --n-null-replicates 2000 \
  --null-tail-method gpd \
  --proximity-null-families source,same_chr,degree \
  --asi-threshold 0.5 \
  --fdr-q 0.10 \
  --output report/
```

Inputs:

  * `edges.csv` -- TF, target, score (one row per edge).
  * `coords.tsv` -- gene, chr, tss.
  * `cells.csv` -- cell_id plus any of {donor, batch, method}.
  * `counts.parquet` (or .csv) -- cell x gene log-normalised matrix.
  * `--edge-features` (optional) -- a pre-computed cell x edge matrix
    (overrides on-the-fly construction from counts).
  * `--scores-balanced` (optional) -- balanced-donor edge scores for ASI.

Outputs (in `report/`):

  * `audit_results.json` -- machine-readable full audit results.
  * `edge_quality_indices.csv` -- per-edge ASI, distance, q-value, pass flags.
  * `cross_class_synthesis.csv` -- pairwise $\phi$/$\chi^2$ + joint retention.
  * `audit_summary.txt` -- human-readable summary.

## Reproduce the BMC paper

```bash
make all          # sims + pert + synth + sweeps + runtime + figures + consistency
make sims         # tool validation on simulated data (Fig 5)
make pert         # perturbation validation (Sec A.11, Fig 1B)
make synth        # cross-class synthesis (Fig 4)
make sweeps       # hyperparameter sensitivity sweeps (Supplement)
make runtime      # wall-clock and peak RSS (BENCHMARKS.md, Sec 4.10)
make figures      # regenerate all 6 figures from CSV/JSON outputs
make consistency  # paper <-> code numeric consistency check
```

See `docs/method_variants.md` for the full enumeration of the 12 inference
method variants audited in the paper, and `BENCHMARKS.md` for the runtime
benchmark machine spec and per-scale timing.

## Repository layout

```
grn-confound-audit/
  grn_confound_audit/
    __init__.py
    technical.py          # Class 1: ASI + leakage classifiers
    proximity.py          # Class 2: 3 nulls + hub-degree stratification
    topological.py        # Class 3: per-edge degree-preserving BH + GPD tail
    pipeline.py           # Orchestrator + cross-class synthesis
    simulate.py           # Ground-truth confound simulator
    cli.py                # End-to-end command-line interface
  scripts/
    run_simulation_benchmarks.py
    run_perturbation_validation.py
    run_cross_class_synthesis_real.py
    run_sensitivity_sweeps.py
    run_runtime_benchmarks.py
    generate_figures.py
    check_paper_code_consistency.py
  data/
    class1_technical/     # legacy aggregate results
    class2_proximity/     # legacy proximity curves
    class3_topological/   # legacy topological calibration summary
    benchmarks/           # outputs from the scripts above
  docs/
    method_variants.md    # full enumeration of the 12 variants
  figures/                # PDF outputs of scripts/generate_figures.py
  BENCHMARKS.md           # runtime / RSS benchmark machine + numbers
  Makefile
  pyproject.toml
  requirements.txt
  environment.yml
  CITATION.cff
  LICENSE                  # MIT
```

## License

MIT (see `LICENSE`).
