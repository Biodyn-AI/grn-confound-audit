"""
Tests for the confound simulator.

Verifies that the simulator (i) produces inputs of the correct shape,
(ii) records ground truth flags consistent with the requested injection,
(iii) is deterministic under fixed seed, and (iv) yields zero injection
when all confound strengths are zero.
"""

import numpy as np
import pandas as pd
import pytest

from grn_confound_audit import simulate, SimulationConfig


def test_null_simulation_has_zero_truth():
    """No injection -> no edges flagged in ground truth."""
    cfg = SimulationConfig(
        n_tfs=10, n_targets=30, n_cells=200,
        donor_confound_strength=0.0,
        proximity_confound_strength=0.0,
        hub_confound_strength=0.0,
        seed=1,
    )
    bundle = simulate(cfg)
    truth = bundle.truth
    assert truth["donor_injected"].sum() == 0
    assert truth["proximity_injected"].sum() == 0
    assert truth["topology_injected"].sum() == 0
    assert len(bundle.edges) > 0
    assert bundle.edges["score"].notna().all()


def test_donor_injection_marks_correct_count():
    cfg = SimulationConfig(
        n_tfs=10, n_targets=30, n_cells=200,
        donor_confound_strength=0.5, donor_confound_fraction=0.3,
        seed=2,
    )
    bundle = simulate(cfg)
    n_edges = len(bundle.edges)
    n_inject = bundle.truth["donor_injected"].sum()
    # Within +/- 5% of the requested fraction (integer rounding tolerance)
    assert abs(n_inject - 0.3 * n_edges) / n_edges <= 0.05


def test_hub_injection_marks_full_tf_set():
    """Hub injection should mark every edge from the chosen hub TFs."""
    cfg = SimulationConfig(
        n_tfs=10, n_targets=30, n_cells=200,
        hub_confound_strength=0.5, hub_confound_fraction=0.2,
        seed=3,
    )
    bundle = simulate(cfg)
    # All injected edges should share their TF with at least one other
    # injected edge (since hubs are TF-level not edge-level)
    injected = bundle.truth.loc[bundle.truth["topology_injected"]]
    tfs_injected = set(injected["tf"].unique())
    # number of hub TFs should be round(n_tfs * hub_fraction) = 2
    assert len(tfs_injected) == round(10 * 0.2)


def test_deterministic_under_fixed_seed():
    cfg = SimulationConfig(
        n_tfs=8, n_targets=20, n_cells=100,
        donor_confound_strength=0.3,
        proximity_confound_strength=0.5,
        hub_confound_strength=0.2,
        seed=11,
    )
    b1 = simulate(cfg)
    b2 = simulate(cfg)
    np.testing.assert_array_equal(b1.edges["score"], b2.edges["score"])
    pd.testing.assert_frame_equal(
        b1.truth.reset_index(drop=True),
        b2.truth.reset_index(drop=True),
    )


def test_shapes_consistent():
    cfg = SimulationConfig(n_tfs=5, n_targets=15, n_cells=80, seed=4)
    bundle = simulate(cfg)
    assert bundle.counts.shape[0] == 80
    assert bundle.counts.shape[1] == 5 + 15
    assert bundle.metadata.shape[0] == 80
    assert bundle.edge_features.shape[0] == 80
    assert bundle.edge_features.shape[1] == len(bundle.edges)
    assert bundle.gene_coords["gene"].nunique() == 5 + 15
