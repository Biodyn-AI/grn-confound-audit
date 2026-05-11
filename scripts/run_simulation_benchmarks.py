#!/usr/bin/env python3
"""
Tool-validation benchmark for grn_confound_audit (Section A.5).

For each of the three confound classes we (i) inject a controllable
amount of that confound into a simulated GRN, (ii) run the matching
audit module, and (iii) compute PR / AUROC / FPR of the audit's
per-edge flag against the ground-truth injection mask.  We also run a
"no confound" scenario to check the false-positive rate at the
recommended thresholds.

Outputs are written to ``data/benchmarks/`` and are consumed by
``scripts/generate_figures.py`` to draw Figure 5.
"""

from __future__ import annotations

import json
import os
import time
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
)

from grn_confound_audit import (
    ConfoundAuditPipeline,
    SimulationConfig,
    simulate,
)


OUT_DIR = os.path.join(
    os.path.dirname(__file__), "..", "data", "benchmarks",
)
os.makedirs(OUT_DIR, exist_ok=True)


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _aligned_quality(rep_quality: pd.DataFrame, truth: pd.DataFrame) -> pd.DataFrame:
    """Merge the audit's per-edge quality with the ground-truth flags."""
    key_q = rep_quality["tf"] + "->" + rep_quality["target"]
    key_t = truth["tf"] + "->" + truth["target"]
    rep_quality = rep_quality.copy()
    truth = truth.copy()
    rep_quality["edge_id"] = key_q.values
    truth["edge_id"] = key_t.values
    return rep_quality.merge(
        truth[
            ["edge_id", "donor_injected", "proximity_injected", "topology_injected"]
        ],
        on="edge_id", how="left",
    )


def _classification_metrics(y_true: np.ndarray, score: np.ndarray) -> dict:
    if np.unique(y_true).size < 2:
        return {
            "auroc": float("nan"),
            "auprc": float("nan"),
            "prevalence": float(np.mean(y_true)),
        }
    return {
        "auroc": float(roc_auc_score(y_true, score)),
        "auprc": float(average_precision_score(y_true, score)),
        "prevalence": float(np.mean(y_true)),
    }


# ----------------------------------------------------------------------
# Scenarios
# ----------------------------------------------------------------------


def _run_one(
    label: str, cfg: SimulationConfig, pipe: ConfoundAuditPipeline,
) -> dict:
    bundle = simulate(cfg)
    metadata = bundle.metadata.set_index("cell_id")
    report = pipe.run(
        edges=bundle.edges,
        gene_coords=bundle.gene_coords,
        scores_balanced=bundle.scores_balanced,
        edge_features=bundle.edge_features,
        covariates={
            c: metadata[c] for c in ("donor", "batch", "method")
            if c in metadata.columns
        },
    )

    eq = pipe._build_edge_quality_table(bundle.edges)
    merged = _aligned_quality(eq, bundle.truth)

    # Class 1 score = ASI (higher = more confounded); flag = blacklisted
    asi = merged["asi"].fillna(0.0).values
    donor_truth = merged["donor_injected"].fillna(False).values.astype(int)
    c1 = _classification_metrics(donor_truth, asi)

    # Class 2 score = proximity indicator (proximate at principal threshold)
    proximate = (
        (~merged["class2_pass"].astype(bool)).astype(int).values
    )
    prox_truth = merged["proximity_injected"].fillna(False).values.astype(int)
    c2 = {
        "prevalence_truth": float(np.mean(prox_truth)),
        "prevalence_flag": float(np.mean(proximate)),
    }
    # Confusion at the principal threshold
    if prox_truth.sum() > 0:
        tp = int(np.sum(proximate & prox_truth))
        fp = int(np.sum(proximate & ~prox_truth.astype(bool)))
        fn = int(np.sum(~proximate.astype(bool) & prox_truth.astype(bool)))
        tn = int(np.sum(~proximate.astype(bool) & ~prox_truth.astype(bool)))
        c2.update({
            "precision": tp / max(tp + fp, 1),
            "recall": tp / max(tp + fn, 1),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        })

    # Class 3 score = 1 - q_bh (higher = more confidently non-null)
    topo_score = 1.0 - merged["topo_q"].fillna(1.0).values
    topo_truth = merged["topology_injected"].fillna(False).values.astype(int)
    c3 = _classification_metrics(topo_truth, topo_score)

    # Cross-class
    cc = report.get("cross_class", {})

    return {
        "label": label,
        "config": vars(cfg),
        "n_edges": int(len(bundle.edges)),
        "class1_donor_metrics": c1,
        "class1_blacklist_rate": float(merged["class1_pass"].mean()),
        "class1_leakage_auc": (
            report["classes"]["class1_technical"]
            .get("leakage", {})
            .get("donor", {})
            .get("auc_best")
        ),
        "class2_metrics": c2,
        "class3_topology_metrics": c3,
        "class3_block_summary": {
            k: {
                "z_score": v.get("z_score"),
                "n_sig": v.get("n_edges_fdr_significant"),
                "valid_block": v.get("valid_block"),
            }
            for k, v in (
                report["classes"]["class3_topological"].get("results_per_k", {}).items()
            )
        },
        "joint_retention": cc.get("joint_retention"),
        "pass_rate_per_class": cc.get("pass_rate_per_class"),
    }


def _pipeline(seed: int) -> ConfoundAuditPipeline:
    # B = 500 for the validation benchmark.  This is enough to estimate
    # AUROC of audit flags vs. ground-truth injections to within a few
    # hundredths; for the real-data analyses the manuscript uses
    # B = 2000 (with optional GPD tail) so that per-edge BH FDR is also
    # meaningfully calibrated.
    return ConfoundAuditPipeline(
        asi_threshold=0.5,
        n_permutations=300,
        n_null_replicates=500,
        null_tail_method="gpd",
        top_k_values=[100, 250, 500],
        distance_thresholds_mb=[0.5, 1.0, 5.0],
        n_top_features=200,
        n_bootstrap_synthesis=300,
        random_state=seed,
    )


SIM_KW = dict(
    n_tfs=15, n_targets=60, n_cells=300, n_donors=4, n_batches=6,
)


def benchmark_null_scenario() -> List[dict]:
    print("[1/5] Null scenario (no injected confounds) ...", flush=True)
    rows = []
    for seed in range(2):
        cfg = SimulationConfig(
            **SIM_KW,
            donor_confound_strength=0.0,
            proximity_confound_strength=0.0,
            hub_confound_strength=0.0,
            seed=100 + seed,
        )
        t0 = time.time()
        rows.append(_run_one(f"null_seed{seed}", cfg, _pipeline(seed)))
        print(f"  seed={seed}  {time.time() - t0:.1f}s", flush=True)
    return rows


def benchmark_single_class(class_name: str) -> List[dict]:
    print(f"[*] Single-class scenario: {class_name} ...", flush=True)
    rows = []
    for strength in (0.3, 1.0):
        for seed in range(2):
            kw = dict(SIM_KW)
            kw["seed"] = 200 + seed
            if class_name == "donor":
                kw["donor_confound_strength"] = strength
                kw["donor_confound_fraction"] = 0.25
            elif class_name == "proximity":
                kw["proximity_confound_strength"] = strength
                kw["proximity_confound_fraction"] = 0.7
                kw["proximity_confound_window_mb"] = 1.0
                kw["chromosomes"] = 4
            elif class_name == "topology":
                kw["hub_confound_strength"] = strength
                kw["hub_confound_fraction"] = 0.20
            cfg = SimulationConfig(**kw)
            t0 = time.time()
            rows.append(_run_one(
                f"{class_name}_s{strength}_seed{seed}", cfg, _pipeline(seed),
            ))
            print(
                f"  {class_name} s={strength} seed={seed}  "
                f"{time.time() - t0:.1f}s", flush=True,
            )
    return rows


def benchmark_combined() -> List[dict]:
    print("[5/5] Combined-confound scenario ...", flush=True)
    rows = []
    for seed in range(2):
        cfg = SimulationConfig(
            **SIM_KW,
            donor_confound_strength=0.8, donor_confound_fraction=0.20,
            proximity_confound_strength=1.0, proximity_confound_fraction=0.6,
            proximity_confound_window_mb=1.0,
            hub_confound_strength=0.4, hub_confound_fraction=0.15,
            chromosomes=4,
            seed=400 + seed,
        )
        t0 = time.time()
        rows.append(_run_one(f"combined_seed{seed}", cfg, _pipeline(seed)))
        print(f"  combined seed={seed}  {time.time() - t0:.1f}s", flush=True)
    return rows


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------


def main():
    all_results = {
        "null": benchmark_null_scenario(),
        "donor_only": benchmark_single_class("donor"),
        "proximity_only": benchmark_single_class("proximity"),
        "topology_only": benchmark_single_class("topology"),
        "combined": benchmark_combined(),
    }

    out_json = os.path.join(OUT_DIR, "simulation_benchmark_results.json")
    with open(out_json, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nWrote {out_json}")

    # Flattened CSV
    rows = []
    for scenario, runs in all_results.items():
        for run in runs:
            rows.append({
                "scenario": scenario,
                "label": run["label"],
                "n_edges": run["n_edges"],
                "donor_strength": run["config"].get("donor_confound_strength"),
                "prox_strength": run["config"].get("proximity_confound_strength"),
                "hub_strength": run["config"].get("hub_confound_strength"),
                "class1_donor_AUROC": run["class1_donor_metrics"]["auroc"],
                "class1_donor_AUPRC": run["class1_donor_metrics"]["auprc"],
                "class1_leakage_AUC_donor": run.get("class1_leakage_auc"),
                "class2_proximity_precision": run["class2_metrics"].get("precision"),
                "class2_proximity_recall": run["class2_metrics"].get("recall"),
                "class3_topology_AUROC": run["class3_topology_metrics"]["auroc"],
                "class3_topology_AUPRC": run["class3_topology_metrics"]["auprc"],
                "joint_retention_rate": (
                    (run.get("joint_retention") or {}).get("rate")
                ),
                "joint_retention_ci_lo": (
                    (run.get("joint_retention") or {}).get("ci_lo")
                ),
                "joint_retention_ci_hi": (
                    (run.get("joint_retention") or {}).get("ci_hi")
                ),
            })
    df = pd.DataFrame(rows)
    out_csv = os.path.join(OUT_DIR, "simulation_benchmark_results.csv")
    df.to_csv(out_csv, index=False)
    print(f"Wrote {out_csv}")

    print("\n--- summary ---")
    print(df.groupby("scenario")[
        [
            "class1_donor_AUROC",
            "class1_leakage_AUC_donor",
            "class3_topology_AUROC",
            "joint_retention_rate",
        ]
    ].mean(numeric_only=True).round(3))


if __name__ == "__main__":
    main()
