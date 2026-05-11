#!/usr/bin/env python3
"""
Hyperparameter sensitivity sweeps for grn_confound_audit (Section A.10).

For each design choice that affects downstream results, run a sweep on
the same simulated bundle (so the only thing that varies is the knob),
record the resulting blacklist rate / FDR significance count / joint
retention, and write a CSV that the figure generator reads to draw
Supplementary Figures S2-S6.

Sweeps included:

  * ASI threshold in {0.3, 0.4, 0.5, 0.6, 0.7}
  * n_top_features in {100, 200, 500, 1000}
  * proximity distance threshold in {0.5, 1, 5, 10} Mb
  * top-k in {100, 250, 500, 1000}
  * n_null_replicates in {100, 500, 1000, 2000, 5000}
"""

from __future__ import annotations

import json
import os
import time

import numpy as np
import pandas as pd

from grn_confound_audit import (
    ConfoundAuditPipeline,
    SimulationConfig,
    simulate,
)


OUT_DIR = os.path.join(
    os.path.dirname(__file__), "..", "data", "benchmarks",
)
os.makedirs(OUT_DIR, exist_ok=True)


def _bundle(seed: int):
    cfg = SimulationConfig(
        n_tfs=20, n_targets=80, n_cells=400,
        donor_confound_strength=0.6, donor_confound_fraction=0.25,
        proximity_confound_strength=1.0, proximity_confound_fraction=0.7,
        hub_confound_strength=0.3, hub_confound_fraction=0.20,
        chromosomes=4, seed=seed,
    )
    return simulate(cfg)


def _run_pipeline(bundle, **kw):
    pipe = ConfoundAuditPipeline(
        asi_threshold=kw.get("asi_threshold", 0.5),
        n_top_features=kw.get("n_top_features", 200),
        n_permutations=kw.get("n_permutations", 300),
        n_null_replicates=kw.get("n_null_replicates", 500),
        null_tail_method=kw.get("null_tail_method", "gpd"),
        top_k_values=kw.get("top_k_values", [100, 250, 500]),
        distance_thresholds_mb=kw.get("distance_thresholds_mb", [0.5, 1.0, 5.0]),
        proximity_principal_threshold_mb=kw.get(
            "proximity_principal_threshold_mb", 1.0,
        ),
        n_bootstrap_synthesis=300,
        random_state=7,
    )
    return pipe.run(
        edges=bundle.edges,
        gene_coords=bundle.gene_coords,
        scores_balanced=bundle.scores_balanced,
        edge_features=bundle.edge_features,
        covariates={
            c: bundle.metadata.set_index("cell_id")[c]
            for c in ("donor", "batch", "method")
            if c in bundle.metadata.columns
        },
    ), pipe


def _summary(report, pipe, bundle, label, value, scenario):
    eq = pipe._build_edge_quality_table(bundle.edges)
    bl = float((~eq["class1_pass"]).mean())
    pa = {
        c: float(eq[f"{c}_pass"].mean())
        for c in ("class1", "class2", "class3")
    }
    jr = report["cross_class"]["joint_retention"]
    c3 = report["classes"]["class3_topological"]
    n_sig_per_k = sum(
        v.get("n_edges_fdr_significant", 0)
        for v in (c3.get("results_per_k") or {}).values()
    )
    return {
        "sweep": scenario, "param": label, "value": value,
        "blacklist_rate": round(bl, 4),
        **{f"pass_{k}": round(v, 4) for k, v in pa.items()},
        "joint_retention_rate": jr["rate"],
        "joint_retention_ci_lo": jr["ci_lo"],
        "joint_retention_ci_hi": jr["ci_hi"],
        "total_n_sig_topo": int(n_sig_per_k),
    }


def main():
    rows = []
    bundle = _bundle(seed=11)

    print("[1/5] ASI threshold sweep ...")
    for val in (0.3, 0.4, 0.5, 0.6, 0.7):
        t0 = time.time()
        r, p = _run_pipeline(bundle, asi_threshold=val)
        rows.append(_summary(r, p, bundle, "asi_threshold", val, "asi_threshold"))
        print(f"  asi={val}  {time.time() - t0:.1f}s")

    print("[2/5] n_top_features sweep ...")
    for val in (100, 200, 500, 1000):
        t0 = time.time()
        r, p = _run_pipeline(bundle, n_top_features=val)
        rows.append(_summary(r, p, bundle, "n_top_features", val, "n_top_features"))
        print(f"  n_top={val}  {time.time() - t0:.1f}s")

    print("[3/5] proximity distance threshold sweep ...")
    for val in (0.5, 1.0, 5.0, 10.0):
        t0 = time.time()
        r, p = _run_pipeline(bundle, proximity_principal_threshold_mb=val)
        rows.append(_summary(r, p, bundle, "proximity_thr_mb", val, "proximity_thr_mb"))
        print(f"  thr={val} Mb  {time.time() - t0:.1f}s")

    print("[4/5] top-k sweep (single-k pipelines) ...")
    for val in (100, 250, 500, 1000):
        t0 = time.time()
        r, p = _run_pipeline(bundle, top_k_values=[val])
        rows.append(_summary(r, p, bundle, "top_k", val, "top_k"))
        print(f"  top_k={val}  {time.time() - t0:.1f}s")

    print("[5/5] n_null_replicates sweep ...")
    for val in (100, 500, 1000, 2000):
        t0 = time.time()
        r, p = _run_pipeline(bundle, n_null_replicates=val)
        rows.append(_summary(r, p, bundle, "n_null_replicates", val, "n_null_replicates"))
        print(f"  B={val}  {time.time() - t0:.1f}s")

    df = pd.DataFrame(rows)
    out_csv = os.path.join(OUT_DIR, "sensitivity_sweeps.csv")
    df.to_csv(out_csv, index=False)
    print(f"\nWrote {out_csv}")

    print("\n--- per-sweep summary ---")
    print(
        df.groupby("sweep").agg(
            blacklist_min=("blacklist_rate", "min"),
            blacklist_max=("blacklist_rate", "max"),
            joint_min=("joint_retention_rate", "min"),
            joint_max=("joint_retention_rate", "max"),
        ).round(3)
    )


if __name__ == "__main__":
    main()
