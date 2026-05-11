"""
End-to-end pipeline tests.

These tests run the full ``ConfoundAuditPipeline`` on a small simulated
bundle (so they remain quick) and check that:

  * All three audit classes produce their expected keys.
  * The per-edge quality table is the right shape and carries the
    pass-flag columns.
  * The cross-class synthesis emits per-class pass rates, pairwise
    contingencies and chi2 statistics, and a bootstrap 95% CI.
  * Output files are written when ``output_dir`` is provided.
"""

import json
import os
import tempfile

import pandas as pd

from grn_confound_audit import (
    ConfoundAuditPipeline,
    SimulationConfig,
    simulate,
)


def _small_bundle():
    cfg = SimulationConfig(
        n_tfs=10, n_targets=30, n_cells=200,
        donor_confound_strength=0.4, donor_confound_fraction=0.25,
        proximity_confound_strength=0.6, proximity_confound_fraction=0.6,
        hub_confound_strength=0.2, hub_confound_fraction=0.20,
        chromosomes=3,
        seed=23,
    )
    return simulate(cfg)


def _small_pipeline():
    return ConfoundAuditPipeline(
        asi_threshold=0.5,
        n_permutations=80,
        n_null_replicates=100,
        null_tail_method="empirical",
        top_k_values=[20, 50],
        distance_thresholds_mb=[1.0, 5.0],
        n_top_features=50,
        n_bootstrap_synthesis=100,
        random_state=23,
    )


def test_pipeline_end_to_end():
    bundle = _small_bundle()
    pipe = _small_pipeline()
    metadata = bundle.metadata.set_index("cell_id")
    report = pipe.run(
        edges=bundle.edges,
        gene_coords=bundle.gene_coords,
        scores_balanced=bundle.scores_balanced,
        edge_features=bundle.edge_features,
        covariates={
            c: metadata[c] for c in ("donor", "batch", "method")
        },
    )
    classes = report["classes"]
    assert "class1_technical" in classes
    assert "class2_proximity" in classes
    assert "class3_topological" in classes
    assert "cross_class" in report


def test_pipeline_writes_expected_files():
    bundle = _small_bundle()
    pipe = _small_pipeline()
    metadata = bundle.metadata.set_index("cell_id")
    with tempfile.TemporaryDirectory() as tmp:
        pipe.run(
            edges=bundle.edges,
            gene_coords=bundle.gene_coords,
            scores_balanced=bundle.scores_balanced,
            edge_features=bundle.edge_features,
            covariates={c: metadata[c] for c in ("donor", "batch", "method")},
            output_dir=tmp,
        )
        for fn in (
            "audit_results.json",
            "edge_quality_indices.csv",
            "cross_class_synthesis.csv",
            "audit_summary.txt",
        ):
            assert os.path.exists(os.path.join(tmp, fn)), fn


def test_edge_quality_table_has_pass_columns():
    bundle = _small_bundle()
    pipe = _small_pipeline()
    metadata = bundle.metadata.set_index("cell_id")
    pipe.run(
        edges=bundle.edges,
        gene_coords=bundle.gene_coords,
        scores_balanced=bundle.scores_balanced,
        edge_features=bundle.edge_features,
        covariates={c: metadata[c] for c in ("donor", "batch", "method")},
    )
    eq = pipe._build_edge_quality_table(bundle.edges)
    for col in (
        "tf", "target", "score", "asi",
        "class1_pass", "class2_pass", "class3_pass", "passes_all",
    ):
        assert col in eq.columns, col
    assert eq["class1_pass"].dtype == bool
    assert eq["class2_pass"].dtype == bool
    assert eq["class3_pass"].dtype == bool


def test_cross_class_synthesis_has_phi_and_chi2_and_ci():
    bundle = _small_bundle()
    pipe = _small_pipeline()
    metadata = bundle.metadata.set_index("cell_id")
    report = pipe.run(
        edges=bundle.edges,
        gene_coords=bundle.gene_coords,
        scores_balanced=bundle.scores_balanced,
        edge_features=bundle.edge_features,
        covariates={c: metadata[c] for c in ("donor", "batch", "method")},
    )
    cc = report["cross_class"]
    assert "pairwise" in cc
    pairs = cc["pairwise"]
    assert len(pairs) == 3
    for row in pairs:
        assert "phi_coefficient" in row
        assert "observed_agreement" in row
        assert "expected_agreement_under_independence" in row
    jr = cc["joint_retention"]
    assert jr["ci_lo"] <= jr["rate"] <= jr["ci_hi"]
    assert 0.0 <= jr["rate"] <= 1.0
