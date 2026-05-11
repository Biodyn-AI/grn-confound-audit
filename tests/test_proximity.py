"""
Tests for the Class 2 proximity audit.

Covers (i) genomic-distance computation, (ii) the three null families
all run end-to-end without exception, (iii) hub-degree stratification
returns at most 4 rows, and (iv) the report shape under a minimal toy
universe.
"""

import numpy as np
import pandas as pd

from grn_confound_audit import ProximityAudit


def _toy_inputs(seed=0):
    rng = np.random.default_rng(seed)
    tfs = [f"TF{i}" for i in range(5)]
    tgs = [f"TG{i}" for i in range(15)]
    rows = []
    for tf in tfs:
        for tg in tgs:
            rows.append({"tf": tf, "target": tg, "score": rng.random()})
    edges = pd.DataFrame(rows)
    coord_rows = []
    for g in tfs + tgs:
        coord_rows.append({
            "gene": g,
            "chr": f"chr{rng.integers(1, 5)}",
            "tss": int(rng.uniform(0, 100e6)),
        })
    coords = pd.DataFrame(coord_rows)
    return edges, coords


def test_distance_computation_returns_aligned_series():
    edges, coords = _toy_inputs(seed=1)
    audit = ProximityAudit()
    dist = audit.compute_genomic_distance(edges, coords)
    assert len(dist) == len(edges)
    assert dist.name == "distance_mb"
    assert (dist[dist.notna()] >= 0).all()


def test_three_null_families_run():
    edges, coords = _toy_inputs(seed=2)
    audit = ProximityAudit(
        distance_thresholds_mb=[1.0, 5.0],
        top_k_values=[20, 50],
        n_permutations=50,
        null_families=("source", "same_chr", "degree"),
        random_state=3,
    )
    res = audit.run(edges, coords)
    grid = res["enrichment_grid"]
    assert len(grid) > 0
    for row in grid:
        nulls = row["nulls"]
        assert set(nulls.keys()) == {"source", "same_chr", "degree"}
        for blk in nulls.values():
            assert 0.0 <= blk["p_value"] <= 1.0


def test_hub_stratified_returns_at_most_four_strata():
    edges, coords = _toy_inputs(seed=4)
    audit = ProximityAudit(
        distance_thresholds_mb=[1.0], top_k_values=[20],
        n_permutations=20, random_state=5,
    )
    res = audit.run(edges, coords)
    strata = res["hub_stratified_principal"]
    assert isinstance(strata, list)
    assert len(strata) <= 4
    if strata:
        keys = {"tf_degree_quartile", "fraction_topk_proximate", "enrichment_ratio"}
        assert all(keys.issubset(r) for r in strata)
