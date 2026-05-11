"""
Confound simulator for ground-truth validation of grn_confound_audit.

Produces synthetic edge lists, optional cell x gene counts, optional
metadata, and gene coordinates with **known** confound injections that
the audit pipeline should detect.

Three injection modes (any subset, optionally combined):

  * ``donor_confound``    -- a chosen subset of edges is shifted in score
    by an amount correlated with donor identity, so Class 1 ASI and the
    leakage classifiers should pick them up.
  * ``proximity_confound`` -- a chosen subset of edges between
    genomically close TF/target pairs are *up-weighted*, so Class 2
    proximity enrichment should rise.
  * ``topological_confound`` -- a chosen subset of hub TFs has their
    edges up-weighted, inflating their representation in the top-k and
    producing degree-driven enrichment that Class 3 should detect.

The simulator returns the inputs the audit expects plus a *ground truth*
DataFrame marking which edges (and which confound class) were injected.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np
import pandas as pd


# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------


@dataclass
class SimulationConfig:
    n_tfs: int = 30
    n_targets: int = 200
    n_cells: int = 1000
    n_donors: int = 4
    n_batches: int = 6
    chromosomes: int = 10
    chromosome_length_mb: float = 100.0

    # Edge construction
    edge_density: float = 0.5   # fraction of TF x target candidates with non-zero score
    # Confound injections
    donor_confound_strength: float = 0.0       # added score per donor bias; 0 disables
    donor_confound_fraction: float = 0.2       # fraction of edges injected with donor effect
    proximity_confound_strength: float = 0.0   # multiplicative boost on score for proximate pairs
    proximity_confound_window_mb: float = 1.0
    proximity_confound_fraction: float = 0.5   # fraction of within-window edges receiving boost
    hub_confound_strength: float = 0.0         # additive score for hub-TF edges
    hub_confound_fraction: float = 0.10        # fraction of TFs treated as hubs

    # Noise
    base_noise_sd: float = 0.10
    seed: int = 7


@dataclass
class SimulationBundle:
    edges: pd.DataFrame
    gene_coords: pd.DataFrame
    counts: pd.DataFrame
    metadata: pd.DataFrame
    edge_features: pd.DataFrame
    scores_balanced: pd.Series
    truth: pd.DataFrame
    config: SimulationConfig = field(default_factory=SimulationConfig)


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _gene_panel(cfg: SimulationConfig, rng: np.random.Generator) -> pd.DataFrame:
    """Return gene coordinates DataFrame placing TFs and targets on chromosomes."""
    n_genes = cfg.n_tfs + cfg.n_targets
    names = (
        [f"TF{i:03d}" for i in range(cfg.n_tfs)]
        + [f"TG{i:03d}" for i in range(cfg.n_targets)]
    )
    chrom = rng.integers(1, cfg.chromosomes + 1, size=n_genes)
    tss = rng.uniform(0, cfg.chromosome_length_mb * 1e6, size=n_genes).astype(int)
    return pd.DataFrame({
        "gene": names,
        "chr": [f"chr{c}" for c in chrom],
        "tss": tss,
        "is_tf": [True] * cfg.n_tfs + [False] * cfg.n_targets,
    })


def _candidate_edges(
    cfg: SimulationConfig, gene_coords: pd.DataFrame,
    rng: np.random.Generator,
) -> pd.DataFrame:
    tfs = gene_coords.loc[gene_coords["is_tf"], "gene"].tolist()
    tgts = gene_coords.loc[~gene_coords["is_tf"], "gene"].tolist()

    candidate_rows = []
    for tf in tfs:
        # Each TF has a random subset of targets
        keep = rng.random(len(tgts)) < cfg.edge_density
        for tg, k in zip(tgts, keep):
            if k:
                candidate_rows.append((tf, tg))
    return pd.DataFrame(candidate_rows, columns=["tf", "target"])


def _baseline_scores(
    edges: pd.DataFrame, rng: np.random.Generator, noise_sd: float,
) -> np.ndarray:
    # Each TF gets a latent activity; each TF->target edge a latent affinity.
    tfs = edges["tf"].unique()
    tgts = edges["target"].unique()
    tf_act = pd.Series(rng.normal(0, 1, size=tfs.size), index=tfs)
    tg_aff = pd.Series(rng.normal(0, 1, size=tgts.size), index=tgts)
    base = (
        tf_act.reindex(edges["tf"]).values
        + tg_aff.reindex(edges["target"]).values
        + rng.normal(0, noise_sd, size=len(edges))
    )
    # Rescale to [0, 1] so confound additions stay interpretable
    base = (base - base.min()) / (base.max() - base.min() + 1e-9)
    return base


# ----------------------------------------------------------------------
# Top-level
# ----------------------------------------------------------------------


def simulate(cfg: Optional[SimulationConfig] = None) -> SimulationBundle:
    """Generate a simulated GRN scenario with configurable confounds."""
    if cfg is None:
        cfg = SimulationConfig()
    rng = np.random.default_rng(cfg.seed)

    gene_coords = _gene_panel(cfg, rng)
    edges = _candidate_edges(cfg, gene_coords, rng).reset_index(drop=True)
    baseline = _baseline_scores(edges, rng, cfg.base_noise_sd)

    truth = pd.DataFrame({
        "tf": edges["tf"].values,
        "target": edges["target"].values,
        "base_score": baseline,
        "donor_injected": False,
        "proximity_injected": False,
        "topology_injected": False,
    })

    # ------------------------------------------------------------------
    # Donor confound (Class 1)
    # ------------------------------------------------------------------
    n_edges = len(edges)
    donor_idx = pd.Series(
        rng.integers(0, cfg.n_donors, size=cfg.n_cells),
        name="donor",
    )
    batch_idx = pd.Series(
        rng.integers(0, cfg.n_batches, size=cfg.n_cells),
        name="batch",
    )
    method_idx = pd.Series(
        rng.integers(0, 2, size=cfg.n_cells),
        name="method",
    )
    metadata = pd.DataFrame({
        "cell_id": [f"cell_{i:05d}" for i in range(cfg.n_cells)],
        "donor": [f"donor{d}" for d in donor_idx],
        "batch": [f"batch{b}" for b in batch_idx],
        "method": ["10X" if m == 0 else "SS2" for m in method_idx],
    }).set_index("cell_id")

    donor_perturbed_mask = np.zeros(n_edges, dtype=bool)
    donor_perturbation_signal = np.zeros(n_edges, dtype=float)
    if cfg.donor_confound_strength > 0 and cfg.donor_confound_fraction > 0:
        n_perturb = int(round(n_edges * cfg.donor_confound_fraction))
        chosen = rng.choice(n_edges, size=n_perturb, replace=False)
        donor_perturbed_mask[chosen] = True
        # Donor sensitivity per perturbed edge
        donor_sens = rng.normal(0, 1, size=n_perturb)
        donor_perturbation_signal[chosen] = donor_sens
        truth.loc[chosen, "donor_injected"] = True

    # Counts matrix (cells x genes). Latent factor model:
    # each cell has an expression vector from a Gaussian mixture by donor,
    # with TF expression that pushes target expression for donor-perturbed
    # edges only.
    gene_names = gene_coords["gene"].tolist()
    g_idx = {g: i for i, g in enumerate(gene_names)}

    # Per-donor latent shift for each gene
    donor_latents = rng.normal(0, 0.4, size=(cfg.n_donors, len(gene_names)))
    base_expression = (
        donor_latents[donor_idx.values]
        + rng.normal(0, 0.5, size=(cfg.n_cells, len(gene_names)))
    )
    counts = pd.DataFrame(
        np.exp(base_expression),
        index=metadata.index,
        columns=gene_names,
    )

    # Per-cell ASI signal: for each perturbed edge, donor membership shifts
    # the edge's *score* relative to a balanced average.
    #
    # We don't need a fully causal generative model -- we need that:
    #   * full-sample edge scores carry donor-confound,
    #   * balanced-sample edge scores remove most of it.
    # This is exactly the situation ASI is designed to detect.
    cell_edge_features = pd.DataFrame(
        index=metadata.index,
        columns=[f"{tf}->{tg}" for tf, tg in zip(edges["tf"], edges["target"])],
        dtype=float,
    )

    donor_one_hot = pd.get_dummies(donor_idx).values  # (n_cells, n_donors)
    donor_signal_per_edge = rng.normal(
        0, cfg.donor_confound_strength, size=(cfg.n_donors, n_edges)
    )
    # Only perturbed edges carry signal
    donor_signal_per_edge[:, ~donor_perturbed_mask] = 0.0
    edge_base_mat = np.repeat(baseline[None, :], cfg.n_cells, axis=0)
    edge_noise = rng.normal(0, cfg.base_noise_sd, size=(cfg.n_cells, n_edges))
    cell_signal = donor_one_hot @ donor_signal_per_edge  # (n_cells, n_edges)
    cell_edge_features.iloc[:, :] = edge_base_mat + cell_signal + edge_noise

    # Full and balanced edge scores
    scores_full = cell_edge_features.mean(axis=0)
    # Balanced: average within donor first, then across donors
    scores_balanced = (
        cell_edge_features.assign(_donor=metadata["donor"].values)
        .groupby("_donor", observed=True).mean()
        .mean(axis=0)
    )
    scores_full.index = [
        f"{tf}->{tg}" for tf, tg in zip(edges["tf"], edges["target"])
    ]
    scores_balanced.index = scores_full.index

    # ------------------------------------------------------------------
    # Proximity confound (Class 2)
    # ------------------------------------------------------------------
    coord_map = gene_coords.set_index("gene")
    distance_mb = np.full(n_edges, np.nan)
    for i, (tf, tg) in enumerate(zip(edges["tf"], edges["target"])):
        if coord_map.loc[tf, "chr"] != coord_map.loc[tg, "chr"]:
            continue
        distance_mb[i] = (
            abs(int(coord_map.loc[tf, "tss"]) - int(coord_map.loc[tg, "tss"]))
            / 1e6
        )
    truth["distance_mb"] = distance_mb

    proximity_perturbed_mask = np.zeros(n_edges, dtype=bool)
    if cfg.proximity_confound_strength > 0 and cfg.proximity_confound_fraction > 0:
        candidate = np.where(
            (~np.isnan(distance_mb))
            & (distance_mb <= cfg.proximity_confound_window_mb)
        )[0]
        n_take = int(round(candidate.size * cfg.proximity_confound_fraction))
        if n_take > 0:
            chosen = rng.choice(candidate, size=n_take, replace=False)
            proximity_perturbed_mask[chosen] = True
            truth.loc[chosen, "proximity_injected"] = True

    # ------------------------------------------------------------------
    # Topological confound (Class 3)
    # ------------------------------------------------------------------
    tfs_all = edges["tf"].unique()
    n_hubs = max(1, int(round(tfs_all.size * cfg.hub_confound_fraction)))
    hub_tfs = set(
        rng.choice(tfs_all, size=n_hubs, replace=False).tolist()
    )
    topo_perturbed_mask = np.array(
        [tf in hub_tfs for tf in edges["tf"]], dtype=bool,
    )
    if cfg.hub_confound_strength > 0:
        truth.loc[topo_perturbed_mask, "topology_injected"] = True
    else:
        topo_perturbed_mask = np.zeros(n_edges, dtype=bool)

    # ------------------------------------------------------------------
    # Final edge scores: baseline + injected effects
    # ------------------------------------------------------------------
    final_score = baseline.copy()
    if cfg.proximity_confound_strength > 0:
        boost = (
            cfg.proximity_confound_strength * proximity_perturbed_mask
        )
        final_score = final_score * (1.0 + boost)
    if cfg.hub_confound_strength > 0:
        final_score = (
            final_score + cfg.hub_confound_strength * topo_perturbed_mask
        )
    # Donor confound shifts manifest via scores_full vs scores_balanced;
    # we keep edges["score"] = scores_full so the audit operates on the
    # confounded ranking.
    if cfg.donor_confound_strength > 0:
        final_score = scores_full.values.copy()

    edges = edges.copy()
    edges["score"] = final_score
    truth["final_score"] = final_score

    return SimulationBundle(
        edges=edges,
        gene_coords=gene_coords[["gene", "chr", "tss"]].copy(),
        counts=counts,
        metadata=metadata.reset_index(),
        edge_features=cell_edge_features,
        scores_balanced=scores_balanced,
        truth=truth,
        config=cfg,
    )
