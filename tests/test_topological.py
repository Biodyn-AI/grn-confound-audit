"""
Tests for the Class 3 topological audit.

Covers (i) Benjamini-Hochberg correctness on a small array,
(ii) per-edge p-value shape and bounds, (iii) the validity-gate
diagnostics on a toy small-network example, and (iv) that a constant
score column yields a degenerate but non-crashing report.
"""

import numpy as np
import pandas as pd
import pytest

from grn_confound_audit import TopologicalAudit, benjamini_hochberg


def test_bh_simple_case():
    p = np.array([0.001, 0.04, 0.10, 0.50, 0.90])
    q = benjamini_hochberg(p)
    # Monotone non-decreasing after re-ordering by p
    order = np.argsort(p)
    assert np.all(np.diff(q[order]) >= -1e-12)
    # First p (0.001) -> q must be very small
    assert q[0] < 0.01
    # All values in [0, 1]
    assert q.min() >= 0
    assert q.max() <= 1


def test_bh_handles_ties():
    p = np.array([0.05, 0.05, 0.05, 0.5])
    q = benjamini_hochberg(p)
    assert q.shape == p.shape
    assert np.all(q <= 1)


def test_topological_audit_returns_per_edge_table():
    rng = np.random.default_rng(7)
    n_tf, n_tg = 10, 20
    rows = []
    for i in range(n_tf):
        targets = rng.choice(n_tg, size=10, replace=False)
        for t in targets:
            rows.append({
                "tf": f"TF{i}",
                "target": f"TG{t}",
                "score": rng.random(),
            })
    edges = pd.DataFrame(rows)

    audit = TopologicalAudit(
        top_k_values=[20, 50], n_null_replicates=50,
        random_state=11, null_tail_method="empirical",
    )
    res = audit.run(edges)
    assert "per_edge" in res
    per_edge = res["per_edge"]
    assert isinstance(per_edge, pd.DataFrame)
    assert len(per_edge) == len(edges)
    for col in ("p_emp", "p_combined", "q_bh"):
        assert per_edge[col].between(0, 1).all()
    # n_edges_fdr_significant_total is honestly derived from per-edge q
    n_real = int((per_edge["q_bh"] <= 0.10).sum())
    assert res["n_edges_fdr_significant_total"] == n_real


def test_topological_audit_swap_diagnostics_reported():
    rng = np.random.default_rng(13)
    edges = pd.DataFrame({
        "tf": [f"TF{i % 5}" for i in range(40)],
        "target": [f"TG{(i * 7) % 17}" for i in range(40)],
        "score": rng.random(40),
    }).drop_duplicates(subset=["tf", "target"]).reset_index(drop=True)
    audit = TopologicalAudit(
        top_k_values=[10], n_null_replicates=30, random_state=9,
    )
    res = audit.run(edges)
    diag = res["swap_diagnostics_summary"]
    assert 0.0 <= diag["mean_swap_success"] <= 1.0
    assert 0.0 <= diag["mean_edge_turnover"] <= 1.0


def test_topological_audit_constant_scores_doesnt_crash():
    edges = pd.DataFrame({
        "tf": ["TF0", "TF0", "TF1", "TF1"],
        "target": ["TG0", "TG1", "TG0", "TG1"],
        "score": [0.5, 0.5, 0.5, 0.5],
    })
    audit = TopologicalAudit(
        top_k_values=[2], n_null_replicates=10, random_state=1,
    )
    res = audit.run(edges)
    # No exception is the test; result either has results_per_k or a warning
    assert "warning" in res or "results_per_k" in res
