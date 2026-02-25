# Three Classes of Confound in Gene-Regulatory-Network Inference

Data and analysis code for a systematic audit of technical, genomic-structural, and topological biases in single-cell GRN inference.

*Anonymous repository for double-blind peer review.*

## Key Findings

| Confound Class | Key Metric | Finding |
|---------------|-----------|---------|
| **Technical** (batch/donor/method) | Leakage AUC, ASI | Donor/batch identity recoverable at AUC 0.85-0.97; ~25% composition-mediated; ComBat best correction |
| **Genomic-structural** (proximity) | Enrichment ratio | Prior-heavy methods show 2-3x proximity enrichment; attenuated to 1.15-1.28x under degree-preserving nulls; fails external replication |
| **Topological** (hub degree) | Global z-score, edge FDR | Global z-scores 12-60 but zero edges significant at FDR <= 0.10 |
| **Combined** | Joint filter pass rate | Only ~28% of edges pass all three filters jointly |

## Repository Structure

```
grn-confound-audit/
  data/                              Result data from each confound class
    class1_technical/                Leakage AUC, ASI, perturbation validation
    class2_proximity/                Proximity enrichment curves
    class3_topological/              Degree-preserving null calibration
  scripts/
    generate_figures.py              Generate all 4 manuscript figures
  figures/                           Output directory for generated PDFs
  Makefile                           Build automation
  requirements.txt                   Python dependencies
  CITATION.cff                       Citation metadata
  LICENSE                            MIT License
```

## Reproducing the Figures

### Prerequisites

- Python 3.9+

### Quick Start

```bash
git clone <this-repository>
cd grn-confound-audit

pip install -r requirements.txt

make all
```

### Step-by-Step

```bash
# Generate all 4 figures from the included data
python scripts/generate_figures.py
```

This produces four PDF figures in `figures/`:

| Figure | Content | Data Source |
|--------|---------|-------------|
| `fig1_technical.pdf` | Technical confound: leakage severity, perturbation validation, correction benchmark | `data/class1_technical/` |
| `fig2_proximity.pdf` | Genomic proximity: enrichment curves for primary and external methods | `data/class2_proximity/` |
| `fig3_topological.pdf` | Topological confound: global z-scores and edge-level significance | `data/class3_topological/` |
| `fig4_crossclass.pdf` | Cross-class synthesis: filter pass rates and pairwise agreement | Derived from cross-class analysis |

## Data Description

### Class 1: Technical Confound (`data/class1_technical/`)

| File | Description |
|------|-------------|
| `all_results.json` | Phase 1 leakage classifier results for immune, lung, and kidney: AUC, balanced accuracy, ASI distribution, edge-score stability |
| `phase2_kidney_results.json` | Phase 2 kidney-specific results (method-balanced, single-donor tissue) |
| `phase2_leakage_correction_results.json` | Correction benchmark: donor AUC, balanced accuracy, and edge-score preservation for 5 correction methods on lung |
| `phase2_perturbation_combined_Immune.csv` | Per-edge perturbation validation: ASI, blacklist status, perturbation significance, and effect size for immune-tissue edges matched to CRISPR screens |

### Class 2: Genomic Proximity (`data/class2_proximity/`)

| File | Description |
|------|-------------|
| `proximity_bias_curves.csv` | Proximity enrichment ratios at multiple distance thresholds and top-k values for 4 primary inference methods |
| `external_replication_proximity_curves.csv` | Enrichment curves for 10 external replication datasets (CRISPR perturbation, cross-tissue ortholog, cross-method consensus) |

### Class 3: Topological Confound (`data/class3_topological/`)

| File | Description |
|------|-------------|
| `null_calibration_summary.csv` | Degree-preserving null calibration: global z-scores, edge-level FDR significance counts, and swap quality metrics for 12 inference methods at 3 top-k levels |

## Datasets

All analyses use tissue compartments from the Tabula Sapiens atlas:

| Dataset | Cells | Donors | Batches | Classes |
|---------|-------|--------|---------|---------|
| Immune | 20,000 | 24 | 42 | 1, 2, 3 |
| Lung | 20,000 | 4 | 7 | 1, 2, 3 |
| Kidney | 11,376 | 1 | 2 | 1, 2 |

Raw single-cell data: [tabula-sapiens-portal.ds.czbiohub.org](https://tabula-sapiens-portal.ds.czbiohub.org/)

## License

MIT License. See [LICENSE](LICENSE) for details.
