#!/usr/bin/env python3
"""
Hardware / runtime benchmarking for grn_confound_audit (Section A.13).

Measures wall-clock and peak resident-set size for each audit module on
simulated bundles of varying sizes.  Reports CPU model and total RAM.
Output: ``data/benchmarks/runtime_benchmarks.csv`` and ``BENCHMARKS.md``.
"""

from __future__ import annotations

import json
import os
import platform
import resource
import time
from pathlib import Path

import pandas as pd

from grn_confound_audit import (
    ConfoundAuditPipeline,
    SimulationConfig,
    simulate,
)


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "benchmarks"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _machine_info() -> dict:
    out = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or "unknown",
        "python": platform.python_version(),
    }
    # macOS extras
    try:
        import subprocess

        cpu = subprocess.check_output(
            ["sysctl", "-n", "machdep.cpu.brand_string"], text=True,
        ).strip()
        out["cpu_brand"] = cpu
        ram_bytes = int(subprocess.check_output(
            ["sysctl", "-n", "hw.memsize"], text=True,
        ).strip())
        out["ram_gb"] = round(ram_bytes / 1024**3, 1)
        cores = int(subprocess.check_output(
            ["sysctl", "-n", "hw.ncpu"], text=True,
        ).strip())
        out["n_cores"] = cores
    except Exception:
        pass
    return out


def _peak_rss_mb() -> float:
    """Peak resident-set size in MB (Linux/macOS unit handling)."""
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # Darwin returns bytes; Linux returns KB.
    if platform.system() == "Darwin":
        return rss / (1024 * 1024)
    return rss / 1024


def _time_one(scale_label: str, n_tfs: int, n_targets: int, n_cells: int) -> dict:
    rss0 = _peak_rss_mb()
    cfg = SimulationConfig(
        n_tfs=n_tfs, n_targets=n_targets, n_cells=n_cells,
        donor_confound_strength=0.6, donor_confound_fraction=0.25,
        proximity_confound_strength=1.0, proximity_confound_fraction=0.7,
        hub_confound_strength=0.3, hub_confound_fraction=0.20,
        chromosomes=4, seed=11,
    )
    t_sim = time.time()
    bundle = simulate(cfg)
    t_sim = time.time() - t_sim

    pipe = ConfoundAuditPipeline(
        asi_threshold=0.5,
        n_top_features=200,
        n_permutations=300,
        n_null_replicates=500,
        null_tail_method="gpd",
        top_k_values=[100, 250, 500],
        distance_thresholds_mb=[0.5, 1.0, 5.0],
        n_bootstrap_synthesis=300,
        random_state=11,
    )

    # Class 1
    t1 = time.time()
    c1 = pipe.technical.run(
        scores_full=bundle.edges.set_index(
            bundle.edges["tf"] + "->" + bundle.edges["target"]
        )["score"],
        scores_balanced=bundle.scores_balanced,
        edge_features=bundle.edge_features,
        covariates={
            c: bundle.metadata.set_index("cell_id")[c]
            for c in ("donor", "batch", "method")
        },
    )
    t1 = time.time() - t1

    # Class 2
    t2 = time.time()
    c2 = pipe.proximity.run(bundle.edges, bundle.gene_coords)
    t2 = time.time() - t2

    # Class 3
    t3 = time.time()
    c3 = pipe.topological.run(bundle.edges)
    t3 = time.time() - t3

    rss1 = _peak_rss_mb()
    return {
        "scale": scale_label,
        "n_edges": int(len(bundle.edges)),
        "n_cells": int(bundle.metadata.shape[0]),
        "sim_time_s": round(t_sim, 2),
        "class1_time_s": round(t1, 2),
        "class2_time_s": round(t2, 2),
        "class3_time_s": round(t3, 2),
        "total_audit_time_s": round(t1 + t2 + t3, 2),
        "peak_rss_mb": round(rss1, 1),
        "rss_delta_mb": round(rss1 - rss0, 1),
    }


def main():
    info = _machine_info()
    print("Machine:", json.dumps(info, indent=2))

    rows = []
    for label, n_tfs, n_targets, n_cells in (
        ("small", 15, 50, 200),
        ("medium", 25, 120, 500),
        ("large", 40, 200, 1000),
    ):
        print(f"[{label}] running ...")
        rows.append(_time_one(label, n_tfs, n_targets, n_cells))
        print(f"  {rows[-1]}")

    df = pd.DataFrame(rows)
    out_csv = OUT_DIR / "runtime_benchmarks.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nWrote {out_csv}")

    md_path = ROOT / "BENCHMARKS.md"
    with md_path.open("w") as f:
        f.write("# Runtime and Memory Benchmarks\n\n")
        f.write("## Machine\n\n")
        for k, v in info.items():
            f.write(f"- **{k}**: {v}\n")
        f.write("\n## Per-scale wall-clock and peak RSS\n\n")
        # Simple ASCII table fallback if `tabulate` is not installed
        headers = list(df.columns)
        rows = df.values.tolist()
        f.write("| " + " | ".join(headers) + " |\n")
        f.write("|" + "|".join(["---"] * len(headers)) + "|\n")
        for r in rows:
            f.write("| " + " | ".join(str(x) for x in r) + " |\n")
        f.write("\n")
        f.write(
            "**Notes.** Times above use ``n_null_replicates = 500`` and "
            "``n_permutations = 300``, which are sufficient for the "
            "tool-validation analyses in this paper.  For real-data "
            "publication use we recommend ``n_null_replicates >= 2000`` "
            "with ``null_tail_method='gpd'`` (B = 2000 multiplies the "
            "Class 3 time by roughly 4x, but only Class 3 scales with B). "
            "All other modules are linear in n_edges and approximately "
            "constant in n_cells past a few thousand.\n"
        )
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
