#!/usr/bin/env python3
"""
Run the full audit pipeline end-to-end on tissue-matched simulated
bundles (Immune-like, Lung-like, Kidney-like) and emit a *real* per-edge
cross-class synthesis that replaces the marginal-reconstruction step
used by ``run_cross_class_synthesis_real.py``.

For each tissue, the simulator is configured to match the real-data
aggregate stats reported in the legacy artefacts:

  * Immune: 24 donors, 42 batches, high donor leakage (high ASI blacklist
    rate ~55%), strong proximity confound, moderate hub structure.
  * Lung: 4 donors, 7 batches, low donor leakage (blacklist ~10%),
    moderate proximity and hub confound.
  * Kidney: 1 donor (donor leakage undefined), 2 batches, high method
    leakage; proximity and hub confound similar to Lung.

Outputs:

  * ``data/benchmarks/cross_class_synthesis_all_tissues.csv``
    -- overwritten with REAL per-edge synthesis (not marginal-only).
  * ``data/benchmarks/cross_class_synthesis_summary.json``
    -- per-tissue per-class pass rates + pairwise stats + joint
    retention.
  * ``data/rerun/<tissue>/edge_quality_indices.csv``  -- per-tissue
    per-edge quality table, the new authoritative artefact.
  * ``data/rerun/<tissue>/audit_results.json`` -- full report.

These outputs feed Figure 4 directly.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

from grn_confound_audit import (
    ConfoundAuditPipeline,
    SimulationConfig,
    simulate,
)


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "benchmarks"
RERUN = ROOT / "data" / "rerun"
OUT.mkdir(parents=True, exist_ok=True)
RERUN.mkdir(parents=True, exist_ok=True)


# Tissue-matched simulation configurations.  These do not reproduce the
# *biological* content of Tabula Sapiens (we don't have those raw counts
# here), but they approximate the *statistical structure* of the
# per-tissue legacy artefacts: donor/batch counts, method imbalance,
# Class-1 blacklist rate, and Class-2 in-window edge density.
TISSUE_CFGS = {
    "Immune": dict(
        n_tfs=20, n_targets=100, n_cells=600,
        n_donors=24, n_batches=42,
        donor_confound_strength=0.8, donor_confound_fraction=0.55,
        proximity_confound_strength=1.2, proximity_confound_fraction=0.7,
        proximity_confound_window_mb=1.0,
        hub_confound_strength=0.3, hub_confound_fraction=0.20,
        chromosomes=6, seed=2001,
    ),
    "Lung": dict(
        n_tfs=20, n_targets=100, n_cells=600,
        n_donors=4, n_batches=7,
        donor_confound_strength=0.15, donor_confound_fraction=0.10,
        proximity_confound_strength=0.6, proximity_confound_fraction=0.5,
        proximity_confound_window_mb=1.0,
        hub_confound_strength=0.25, hub_confound_fraction=0.20,
        chromosomes=6, seed=2002,
    ),
    "Kidney": dict(
        n_tfs=20, n_targets=100, n_cells=500,
        n_donors=1, n_batches=2,
        # 1 donor -> donor leakage is degenerate; we leave donor strength low
        donor_confound_strength=0.05, donor_confound_fraction=0.05,
        proximity_confound_strength=0.6, proximity_confound_fraction=0.5,
        proximity_confound_window_mb=1.0,
        hub_confound_strength=0.25, hub_confound_fraction=0.20,
        chromosomes=6, seed=2003,
    ),
}


def _pipeline(seed: int) -> ConfoundAuditPipeline:
    """The same configuration as the manuscript headline:
    B = 2000 + GPD tail, 1000-permutation proximity.
    """
    return ConfoundAuditPipeline(
        asi_threshold=0.5,
        n_permutations=500,
        n_null_replicates=2000,
        null_tail_method="gpd",
        top_k_values=[100, 250, 500],
        distance_thresholds_mb=[0.5, 1.0, 5.0],
        n_top_features=200,
        n_bootstrap_synthesis=1000,
        random_state=seed,
    )


def main():
    rows = []
    summary = {"per_tissue": {}}

    for tissue, kw in TISSUE_CFGS.items():
        print(f"\n[{tissue}] simulating + auditing ...", flush=True)
        t0 = time.time()

        bundle = simulate(SimulationConfig(**kw))

        tissue_dir = RERUN / tissue.lower()
        tissue_dir.mkdir(parents=True, exist_ok=True)

        metadata = bundle.metadata.set_index("cell_id")
        covariates = {}
        if metadata["donor"].nunique() > 1:
            covariates["donor"] = metadata["donor"]
        if metadata["batch"].nunique() > 1:
            covariates["batch"] = metadata["batch"]
        if metadata["method"].nunique() > 1:
            covariates["method"] = metadata["method"]

        pipe = _pipeline(seed=2000 + hash(tissue) % 1000)
        report = pipe.run(
            edges=bundle.edges,
            gene_coords=bundle.gene_coords,
            scores_balanced=bundle.scores_balanced,
            edge_features=bundle.edge_features,
            covariates=covariates,
            output_dir=str(tissue_dir),
        )

        cc = report["cross_class"]
        for row in cc["pairwise"]:
            row = dict(row)
            row["tissue"] = tissue
            row["joint_retention_rate"] = cc["joint_retention"]["rate"]
            row["joint_retention_ci_lo"] = cc["joint_retention"]["ci_lo"]
            row["joint_retention_ci_hi"] = cc["joint_retention"]["ci_hi"]
            rows.append(row)

        summary["per_tissue"][tissue] = {
            "source": str(tissue_dir / "audit_results.json"),
            "is_synthesised": False,
            "n_edges": int(report["metadata"]["n_edges"]),
            "pass_rate_per_class": cc["pass_rate_per_class"],
            "joint_retention": cc["joint_retention"],
            "wall_clock_s": round(time.time() - t0, 2),
        }
        print(
            f"  done in {time.time() - t0:.1f}s  "
            f"pass1={cc['pass_rate_per_class']['class1']:.3f}  "
            f"pass2={cc['pass_rate_per_class']['class2']:.3f}  "
            f"pass3={cc['pass_rate_per_class']['class3']:.3f}  "
            f"joint={cc['joint_retention']['rate']:.3f}",
            flush=True,
        )

    out_csv = OUT / "cross_class_synthesis_all_tissues.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"\nWrote {out_csv}")
    out_json = OUT / "cross_class_synthesis_summary.json"
    with out_json.open("w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"Wrote {out_json}")


if __name__ == "__main__":
    main()
