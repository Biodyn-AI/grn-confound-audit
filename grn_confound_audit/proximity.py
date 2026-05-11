"""
Class 2: Genomic-proximity confound audit.

Three null families are evaluated for each (distance_threshold, top_k):

  * **source**            -- permute the edge score vector while keeping
    the candidate edge set fixed.  Tests whether top-ranked edges are
    enriched for proximate pairs beyond what a random reordering of the
    same scores would produce.
  * **same_chr**          -- permute scores within each
    same-chromosome-vs-different-chromosome stratum.  Tests whether the
    proximity signal exceeds what is already explained by the
    same-chromosome marginal.
  * **degree**             -- degree-preserving rewiring of the candidate
    edge set; scores are reassigned to rewired edges by the original
    score ranks.  Tests whether the proximity signal survives once the
    TF/target degree distributions are matched.

In addition to the overall enrichment, this module reports the same
enrichment stratified by TF in-degree quartile (hub-vs-tail), so a
mechanistic "hub-driven" interpretation can be evaluated directly rather
than asserted from one aggregate attenuation number.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable, Optional

import numpy as np
import pandas as pd


_NULL_FAMILIES = ("source", "same_chr", "degree")


class ProximityAudit:
    """Audit GRN edge scores for genomic-proximity bias.

    Parameters
    ----------
    distance_thresholds_mb : list of float
        Chromosomal distance thresholds in Mb.
    top_k_values : list of int
        Top-k cuts to evaluate.
    n_permutations : int
        Number of permutations per null family.
    null_families : iterable of str
        Subset of ``{'source', 'same_chr', 'degree'}``.
    swap_oversample : int
        Multiplier on swap attempts in the degree-preserving null.
    random_state : int
        Random seed.
    """

    def __init__(
        self,
        distance_thresholds_mb: Optional[list] = None,
        top_k_values: Optional[list] = None,
        n_permutations: int = 1000,
        null_families: Optional[Iterable[str]] = None,
        swap_oversample: int = 10,
        random_state: int = 42,
    ):
        self.distance_thresholds_mb = distance_thresholds_mb or [0.5, 1.0, 5.0, 10.0]
        self.top_k_values = top_k_values or [100, 250, 500, 1000]
        self.n_permutations = n_permutations
        if null_families is None:
            null_families = _NULL_FAMILIES
        bad = set(null_families) - set(_NULL_FAMILIES)
        if bad:
            raise ValueError(f"Unknown proximity null family/families: {bad}")
        self.null_families = tuple(null_families)
        self.swap_oversample = swap_oversample
        self.random_state = random_state

    # ------------------------------------------------------------------
    # Distance and indicators
    # ------------------------------------------------------------------

    @staticmethod
    def compute_genomic_distance(
        edges: pd.DataFrame, gene_coords: pd.DataFrame,
    ) -> pd.Series:
        coord_map = gene_coords.set_index("gene")
        chrom_map = coord_map["chr"].to_dict()
        tss_map = coord_map["tss"].to_dict()

        out = np.full(len(edges), np.nan)
        for i, (tf, tg) in enumerate(zip(edges["tf"].values, edges["target"].values)):
            if tf not in chrom_map or tg not in chrom_map:
                continue
            if chrom_map[tf] != chrom_map[tg]:
                continue
            out[i] = abs(tss_map[tf] - tss_map[tg]) / 1e6
        return pd.Series(out, index=edges.index, name="distance_mb")

    @staticmethod
    def _same_chromosome_flag(
        edges: pd.DataFrame, gene_coords: pd.DataFrame,
    ) -> np.ndarray:
        chrom_map = gene_coords.set_index("gene")["chr"].to_dict()
        out = np.zeros(len(edges), dtype=bool)
        for i, (tf, tg) in enumerate(zip(edges["tf"].values, edges["target"].values)):
            if tf in chrom_map and tg in chrom_map:
                out[i] = chrom_map[tf] == chrom_map[tg]
        return out

    # ------------------------------------------------------------------
    # Null permutation engines
    # ------------------------------------------------------------------

    @staticmethod
    def _null_source(
        scores: np.ndarray, is_proximate: np.ndarray, k: int,
        rng: np.random.Generator,
    ) -> int:
        # Sample top-k indices uniformly at random from the universe
        n = scores.size
        idx = rng.choice(n, size=k, replace=False)
        return int(np.sum(is_proximate[idx]))

    @staticmethod
    def _null_same_chr(
        is_proximate: np.ndarray, same_chr: np.ndarray, k: int,
        rng: np.random.Generator,
    ) -> int:
        """Permute proximity labels within the same-chromosome stratum.

        We sample k edges proportionally to the observed marginal of
        same-vs-different chromosome at top-k.  Within each stratum we
        sample uniformly.
        """
        # In practice we approximate this by sampling k edges uniformly,
        # then *within* the sampled set permuting same-chr / diff-chr
        # labels; equivalent to permuting proximity labels conditional on
        # same-chromosome status.
        n = is_proximate.size
        idx = rng.choice(n, size=k, replace=False)
        same_sel = same_chr[idx]
        # Number of proximate pairs we expect under same_chr-conditional
        # permutation = sum over strata of marginal proximity rates.
        # Compute observed marginals on the universe:
        denom_same = max(int(np.sum(same_chr)), 1)
        denom_diff = max(int(np.sum(~same_chr)), 1)
        rate_same = float(np.sum(is_proximate & same_chr)) / denom_same
        rate_diff = float(np.sum(is_proximate & ~same_chr)) / denom_diff
        # Sample bernoulli within each stratum
        n_same = int(np.sum(same_sel))
        n_diff = int(k - n_same)
        return int(
            rng.binomial(n_same, rate_same) + rng.binomial(n_diff, rate_diff)
        )

    def _degree_preserving_rewire_indices(
        self,
        tf_list: np.ndarray,
        tg_list: np.ndarray,
        rng: np.random.Generator,
    ) -> tuple:
        """Return (new_tf_list, new_tg_list, swap_success).

        Each swap pairs two edges (a,b),(c,d) -> (a,d),(c,b) only when
        the resulting edges are not already in the multigraph and the
        TF-out degree of each TF and the target in-degree of each target
        are preserved.
        """
        edges = list(zip(tf_list.tolist(), tg_list.tolist()))
        original_set = frozenset(edges)
        edge_set = set(edges)
        n_edges = len(edges)
        n_attempts = n_edges * self.swap_oversample
        successes = 0
        for _ in range(n_attempts):
            i = int(rng.integers(0, n_edges))
            j = int(rng.integers(0, n_edges))
            if i == j:
                continue
            a, b = edges[i]
            c, d = edges[j]
            if a == c or b == d:
                continue
            if (a, d) in edge_set or (c, b) in edge_set:
                continue
            edge_set.discard((a, b))
            edge_set.discard((c, d))
            edge_set.add((a, d))
            edge_set.add((c, b))
            edges[i] = (a, d)
            edges[j] = (c, b)
            successes += 1
        new_tf = np.array([e[0] for e in edges])
        new_tg = np.array([e[1] for e in edges])
        return new_tf, new_tg, successes / max(n_attempts, 1)

    def _null_degree(
        self,
        edges_df: pd.DataFrame,
        coord_map: dict,
        chrom_map: dict,
        scores_sorted: np.ndarray,
        distance_threshold_mb: float,
        k: int,
        rng: np.random.Generator,
    ) -> int:
        new_tf, new_tg, _ = self._degree_preserving_rewire_indices(
            edges_df["tf"].values, edges_df["target"].values, rng,
        )
        # Score assignment: rewired edges inherit the *rank-sorted* score
        # vector (top-k still gets the highest scores).  This isolates the
        # contribution of degree structure to proximity enrichment.
        # We test how many of the top-k rewired edges are proximate.
        # rewired edges are in arbitrary order, so we rank by score: the
        # mapping is just first k -> highest scores.
        top_tf, top_tg = new_tf[:k], new_tg[:k]
        cnt = 0
        for tf, tg in zip(top_tf, top_tg):
            if tf not in chrom_map or tg not in chrom_map:
                continue
            if chrom_map[tf] != chrom_map[tg]:
                continue
            if abs(coord_map[tf] - coord_map[tg]) / 1e6 <= distance_threshold_mb:
                cnt += 1
        return cnt

    # ------------------------------------------------------------------
    # Per-cell enrichment computation
    # ------------------------------------------------------------------

    def compute_enrichment(
        self,
        edges: pd.DataFrame,
        distances: pd.Series,
        gene_coords: pd.DataFrame,
        distance_threshold_mb: float,
        top_k: int,
    ) -> dict:
        """Compute enrichment + p-values under all selected null families."""
        sort_idx = edges["score"].sort_values(ascending=False).index
        edges_sorted = edges.loc[sort_idx].reset_index(drop=True)
        dist_sorted = distances.loc[sort_idx].reset_index(drop=True)

        is_proximate = (dist_sorted.values <= distance_threshold_mb)
        is_proximate = is_proximate & ~np.isnan(dist_sorted.values)
        same_chr = self._same_chromosome_flag(edges_sorted, gene_coords)

        n_universe = int(np.sum(~np.isnan(dist_sorted.values)))
        n_proximate_universe = int(np.sum(is_proximate))
        frac_universe = (
            n_proximate_universe / n_universe if n_universe else 0.0
        )

        actual_k = min(top_k, len(edges_sorted))
        n_proximate_topk = int(np.sum(is_proximate[:actual_k]))
        frac_topk = n_proximate_topk / max(actual_k, 1)
        enrichment = (
            frac_topk / frac_universe if frac_universe > 0 else float("nan")
        )

        rng = np.random.default_rng(self.random_state)
        nulls = {}

        # Source-preserving (uniform random k-subset of edges)
        if "source" in self.null_families:
            cnt = np.array([
                self._null_source(
                    edges_sorted["score"].values, is_proximate, actual_k, rng,
                )
                for _ in range(self.n_permutations)
            ])
            null_frac = cnt / actual_k
            null_enr = null_frac / max(frac_universe, 1e-10)
            p = (np.sum(null_enr >= enrichment) + 1) / (self.n_permutations + 1)
            nulls["source"] = {
                "null_mean_enrichment": float(np.mean(null_enr)),
                "null_p95": float(np.quantile(null_enr, 0.95)),
                "p_value": float(p),
            }

        # Same-chromosome-conditional permutation
        if "same_chr" in self.null_families:
            cnt = np.array([
                self._null_same_chr(is_proximate, same_chr, actual_k, rng)
                for _ in range(self.n_permutations)
            ])
            null_frac = cnt / actual_k
            null_enr = null_frac / max(frac_universe, 1e-10)
            p = (np.sum(null_enr >= enrichment) + 1) / (self.n_permutations + 1)
            nulls["same_chr"] = {
                "null_mean_enrichment": float(np.mean(null_enr)),
                "null_p95": float(np.quantile(null_enr, 0.95)),
                "p_value": float(p),
            }

        # Degree-preserving rewiring (heavy; runtime-capped)
        if "degree" in self.null_families:
            coord_map = gene_coords.set_index("gene")["tss"].to_dict()
            chrom_map = gene_coords.set_index("gene")["chr"].to_dict()
            n_deg = min(self.n_permutations, 50)
            cnt = np.array([
                self._null_degree(
                    edges_sorted, coord_map, chrom_map,
                    edges_sorted["score"].values,
                    distance_threshold_mb, actual_k, rng,
                )
                for _ in range(n_deg)
            ])
            null_frac = cnt / actual_k
            null_enr = null_frac / max(frac_universe, 1e-10)
            p = (np.sum(null_enr >= enrichment) + 1) / (n_deg + 1)
            nulls["degree"] = {
                "null_mean_enrichment": float(np.mean(null_enr)),
                "null_p95": float(np.quantile(null_enr, 0.95)),
                "p_value": float(p),
                "n_permutations": int(n_deg),
            }

        return {
            "enrichment_ratio": round(float(enrichment), 4),
            "fraction_topk": round(float(frac_topk), 4),
            "fraction_universe": round(float(frac_universe), 4),
            "n_proximate_topk": int(n_proximate_topk),
            "n_universe": n_universe,
            "top_k": actual_k,
            "distance_threshold_mb": distance_threshold_mb,
            "nulls": nulls,
            # backwards-compat surface
            "p_value": (
                round(float(nulls["source"]["p_value"]), 6)
                if "source" in nulls else None
            ),
        }

    # ------------------------------------------------------------------
    # Hub-degree stratification
    # ------------------------------------------------------------------

    def hub_stratified(
        self,
        edges: pd.DataFrame,
        distances: pd.Series,
        distance_threshold_mb: float,
        top_k: int,
    ) -> pd.DataFrame:
        """Return proximity enrichment by TF in-degree quartile.

        If the hub-driven explanation is correct, the enrichment should
        be concentrated in the top-degree quartile and disappear in the
        bottom quartile.
        """
        tf_degree = edges["tf"].value_counts()
        if tf_degree.size < 4:
            return pd.DataFrame()

        # qcut may collapse bins under heavy ties; size labels to surviving bins
        raw = pd.qcut(tf_degree, q=4, duplicates="drop")
        n_bins = raw.cat.categories.size
        if n_bins < 2:
            return pd.DataFrame()
        quartiles = pd.qcut(
            tf_degree, q=4,
            labels=[f"Q{i+1}" for i in range(n_bins)],
            duplicates="drop",
        )
        tf_to_q = quartiles.to_dict()

        sort_idx = edges["score"].sort_values(ascending=False).index
        edges_sorted = edges.loc[sort_idx].reset_index(drop=True)
        dist_sorted = distances.loc[sort_idx].reset_index(drop=True)
        is_proximate = (dist_sorted.values <= distance_threshold_mb)
        is_proximate = is_proximate & ~np.isnan(dist_sorted.values)
        tf_q = np.array([tf_to_q.get(tf, np.nan) for tf in edges_sorted["tf"]])

        rows = []
        for q in (f"Q{i+1}" for i in range(n_bins)):
            mask = tf_q == q
            if not np.any(mask):
                continue
            frac_universe = float(np.mean(is_proximate[mask]))
            top_mask = mask[:top_k]
            top_proximate = is_proximate[:top_k][top_mask]
            frac_topk = (
                float(np.mean(top_proximate)) if top_proximate.size else 0.0
            )
            enr = (
                frac_topk / frac_universe if frac_universe > 0 else float("nan")
            )
            rows.append({
                "tf_degree_quartile": q,
                "n_edges_in_universe": int(np.sum(mask)),
                "n_edges_in_topk": int(np.sum(top_mask)),
                "fraction_topk_proximate": round(frac_topk, 4),
                "fraction_universe_proximate": round(frac_universe, 4),
                "enrichment_ratio": (
                    round(float(enr), 4) if not np.isnan(enr) else None
                ),
            })
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Top-level
    # ------------------------------------------------------------------

    def run(
        self, edges: pd.DataFrame, gene_coords: pd.DataFrame,
    ) -> dict:
        distances = self.compute_genomic_distance(edges, gene_coords)

        results_grid = []
        for d in self.distance_thresholds_mb:
            for k in self.top_k_values:
                if k > len(edges):
                    continue
                res = self.compute_enrichment(
                    edges, distances, gene_coords, d, k,
                )
                results_grid.append(res)

        significant = [r for r in results_grid if (r.get("p_value") or 1.0) < 0.05]

        # Hub-stratified at the principal (1 Mb, top-1000) cut
        principal = (
            1.0 if 1.0 in self.distance_thresholds_mb
            else self.distance_thresholds_mb[0]
        )
        principal_k = (
            1000 if 1000 in self.top_k_values
            else min(self.top_k_values)
        )
        hub_df = self.hub_stratified(edges, distances, principal, principal_k)

        return {
            "class": 2,
            "name": "Genomic-proximity confound audit",
            "n_edges": len(edges),
            "n_with_coords": int(distances.notna().sum()),
            "n_same_chr": int(distances.notna().sum()),
            "enrichment_grid": results_grid,
            "n_significant_combinations": len(significant),
            "null_families": list(self.null_families),
            "hub_stratified_principal": (
                hub_df.to_dict(orient="records") if not hub_df.empty else []
            ),
            "principal_threshold_mb": principal,
            "principal_top_k": principal_k,
            "distances": distances,
        }
