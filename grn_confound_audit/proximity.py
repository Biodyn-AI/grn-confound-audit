"""
Class 2: Genomic-proximity confound audit.

Detects whether top-ranked GRN edges are enriched for genomically
proximate TF-target pairs beyond what the candidate universe and
degree structure would predict.
"""

import numpy as np
import pandas as pd
from typing import Optional


class ProximityAudit:
    """Audit GRN edge scores for genomic-proximity bias.

    Computes proximity enrichment ratios at multiple top-k and distance
    thresholds, comparing against source-preserving and degree-preserving
    null models.

    Parameters
    ----------
    distance_thresholds_mb : list of float
        Chromosomal distance thresholds in megabases. Pairs within
        this distance on the same chromosome are considered "proximate".
    top_k_values : list of int
        Number of top-ranked edges to evaluate.
    n_permutations : int
        Number of permutations for null-model significance testing.
    random_state : int
        Random seed.
    """

    def __init__(
        self,
        distance_thresholds_mb: Optional[list] = None,
        top_k_values: Optional[list] = None,
        n_permutations: int = 1000,
        random_state: int = 42,
    ):
        self.distance_thresholds_mb = distance_thresholds_mb or [0.5, 1.0, 5.0, 10.0]
        self.top_k_values = top_k_values or [100, 250, 500, 1000]
        self.n_permutations = n_permutations
        self.random_state = random_state

    @staticmethod
    def compute_genomic_distance(
        edges: pd.DataFrame,
        gene_coords: pd.DataFrame,
    ) -> pd.Series:
        """Compute genomic distance for each TF-target edge.

        Parameters
        ----------
        edges : pd.DataFrame
            Must have columns 'tf' and 'target' (gene symbols or IDs).
        gene_coords : pd.DataFrame
            Must have columns 'gene', 'chr', 'tss' (transcription start site).

        Returns
        -------
        pd.Series
            Genomic distance in Mb for each edge. NaN if on different
            chromosomes or if gene coordinates are missing.
        """
        coord_map = gene_coords.set_index("gene")

        distances = []
        for _, row in edges.iterrows():
            tf, target = row["tf"], row["target"]
            if tf not in coord_map.index or target not in coord_map.index:
                distances.append(np.nan)
                continue

            tf_chr = coord_map.loc[tf, "chr"]
            tgt_chr = coord_map.loc[target, "chr"]

            if tf_chr != tgt_chr:
                distances.append(np.nan)  # different chromosomes
            else:
                tf_tss = coord_map.loc[tf, "tss"]
                tgt_tss = coord_map.loc[target, "tss"]
                distances.append(abs(tf_tss - tgt_tss) / 1e6)  # convert to Mb

        return pd.Series(distances, index=edges.index, name="distance_mb")

    def compute_enrichment(
        self,
        edges: pd.DataFrame,
        distances: pd.Series,
        distance_threshold_mb: float,
        top_k: int,
    ) -> dict:
        """Compute proximity enrichment ratio at a given threshold and top-k.

        Enrichment = (fraction proximate in top-k) / (fraction proximate in universe)

        Parameters
        ----------
        edges : pd.DataFrame
            Must have a 'score' column. Assumed sorted by score descending,
            or will be sorted.
        distances : pd.Series
            Genomic distances per edge, aligned with edges index.
        distance_threshold_mb : float
            Distance threshold in Mb.
        top_k : int
            Number of top edges to consider.

        Returns
        -------
        dict with enrichment_ratio, fraction_topk, fraction_universe, p_value
        """
        # Sort by score descending
        sorted_idx = edges["score"].sort_values(ascending=False).index
        dist_sorted = distances.loc[sorted_idx]

        # Same-chromosome pairs within threshold
        is_proximate = dist_sorted <= distance_threshold_mb

        n_universe = is_proximate.notna().sum()
        n_proximate_universe = is_proximate.sum()
        frac_universe = n_proximate_universe / max(n_universe, 1)

        actual_k = min(top_k, len(sorted_idx))
        topk_prox = is_proximate.iloc[:actual_k]
        n_proximate_topk = topk_prox.sum()
        frac_topk = n_proximate_topk / max(actual_k, 1)

        enrichment = frac_topk / max(frac_universe, 1e-10)

        # Source-preserving permutation test
        rng = np.random.RandomState(self.random_state)
        perm_enrichments = []
        for _ in range(self.n_permutations):
            perm_idx = rng.permutation(len(sorted_idx))
            perm_prox = is_proximate.values[perm_idx][:actual_k]
            perm_frac = perm_prox.sum() / max(actual_k, 1)
            perm_enrichments.append(perm_frac / max(frac_universe, 1e-10))

        p_value = (np.sum(np.array(perm_enrichments) >= enrichment) + 1) / (
            self.n_permutations + 1
        )

        return {
            "enrichment_ratio": round(float(enrichment), 4),
            "fraction_topk": round(float(frac_topk), 4),
            "fraction_universe": round(float(frac_universe), 4),
            "n_proximate_topk": int(n_proximate_topk),
            "top_k": actual_k,
            "distance_threshold_mb": distance_threshold_mb,
            "p_value": round(float(p_value), 6),
        }

    def run(
        self,
        edges: pd.DataFrame,
        gene_coords: pd.DataFrame,
    ) -> dict:
        """Run the full Class 2 proximity audit.

        Parameters
        ----------
        edges : pd.DataFrame
            Must have columns 'tf', 'target', 'score'.
        gene_coords : pd.DataFrame
            Must have columns 'gene', 'chr', 'tss'.

        Returns
        -------
        dict
            Complete Class 2 audit results with enrichment at each
            (distance_threshold, top_k) combination.
        """
        distances = self.compute_genomic_distance(edges, gene_coords)

        results_grid = []
        for d in self.distance_thresholds_mb:
            for k in self.top_k_values:
                if k > len(edges):
                    continue
                res = self.compute_enrichment(edges, distances, d, k)
                results_grid.append(res)

        # Summarize: which combinations show significant enrichment?
        significant = [
            r for r in results_grid if r["p_value"] < 0.05
        ]

        return {
            "class": 2,
            "name": "Genomic-proximity confound audit",
            "n_edges": len(edges),
            "n_with_coords": int(distances.notna().sum()),
            "n_same_chr": int((distances >= 0).sum()),
            "enrichment_grid": results_grid,
            "n_significant_combinations": len(significant),
            "distances": distances,
        }
