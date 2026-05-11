"""
End-to-end CLI smoke test.

Creates a tiny set of CSV/TSV inputs in a temp directory, invokes the
``grn-confound-audit run`` CLI via ``subprocess``, and confirms that the
expected output files appear and contain non-empty content.
"""

import os
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd

from grn_confound_audit import simulate, SimulationConfig


def _write_toy_inputs(tmp: str):
    cfg = SimulationConfig(
        n_tfs=6, n_targets=20, n_cells=150,
        donor_confound_strength=0.3,
        proximity_confound_strength=0.5,
        hub_confound_strength=0.1,
        chromosomes=2,
        seed=51,
    )
    b = simulate(cfg)

    edges_path = os.path.join(tmp, "edges.csv")
    b.edges.to_csv(edges_path, index=False)

    coords_path = os.path.join(tmp, "coords.tsv")
    b.gene_coords.to_csv(coords_path, sep="\t", index=False)

    cells_path = os.path.join(tmp, "cells.csv")
    b.metadata.to_csv(cells_path, index=False)

    counts_path = os.path.join(tmp, "counts.csv")
    counts = b.counts.reset_index()
    counts.rename(columns={counts.columns[0]: "cell_id"}, inplace=True)
    counts.to_csv(counts_path, index=False)

    bal_path = os.path.join(tmp, "balanced.csv")
    pd.DataFrame({
        "edge_id": b.scores_balanced.index,
        "score": b.scores_balanced.values,
    }).to_csv(bal_path, index=False)

    return edges_path, coords_path, cells_path, counts_path, bal_path


def test_cli_run_end_to_end():
    with tempfile.TemporaryDirectory() as tmp:
        edges, coords, cells, counts, bal = _write_toy_inputs(tmp)
        out = os.path.join(tmp, "report")

        repo_root = os.path.abspath(
            os.path.join(os.path.dirname(__file__), os.pardir)
        )
        env = os.environ.copy()
        env["PYTHONPATH"] = repo_root + os.pathsep + env.get("PYTHONPATH", "")

        cmd = [
            sys.executable, "-u", "-m", "grn_confound_audit.cli", "run",
            "--edges", edges,
            "--gene-coords", coords,
            "--metadata", cells,
            "--counts", counts,
            "--scores-balanced", bal,
            "--n-null-replicates", "60",
            "--null-tail-method", "empirical",
            "--proximity-null-families", "source,degree",
            "--n-permutations", "80",
            "--asi-threshold", "0.5",
            "--fdr-q", "0.10",
            "--output", out,
        ]
        result = subprocess.run(
            cmd, env=env, capture_output=True, text=True, timeout=300,
        )
        assert result.returncode == 0, (
            f"CLI exited with {result.returncode}\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )

        for fn in (
            "audit_results.json",
            "edge_quality_indices.csv",
            "cross_class_synthesis.csv",
            "audit_summary.txt",
        ):
            path = os.path.join(out, fn)
            assert os.path.exists(path), f"missing output {fn}"
            assert os.path.getsize(path) > 0, f"empty output {fn}"
