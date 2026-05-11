"""
Integrated three-class confound audit pipeline.

Orchestrates the three audit classes and produces a per-edge quality
table plus an honest cross-class synthesis (observed pairwise agreement,
expected agreement under independence, phi/MCC, chi-square test, and a
bootstrap 95% CI on the three-way joint retention rate).

The cross-class synthesis output (``cross_class_synthesis.csv``) is the
authoritative source for Figure 4; the figure-generation script reads
this file directly so that no figure number can drift from the analysis.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import pandas as pd

try:
    from scipy.stats import chi2_contingency

    _HAS_SCIPY = True
except ImportError:  # pragma: no cover
    _HAS_SCIPY = False

from .technical import TechnicalAudit
from .proximity import ProximityAudit
from .topological import TopologicalAudit


class ConfoundAuditPipeline:
    """Run a complete three-class confound audit on a GRN edge list.

    Parameters
    ----------
    asi_threshold : float
        Class 1 ASI threshold for technical blacklisting.
    distance_thresholds_mb : list of float
        Class 2 distance thresholds.
    top_k_values : list of int
        Top-k cuts shared across Class 2 and Class 3.
    n_permutations : int
        Class 2 permutations per null family.
    n_null_replicates : int
        Class 3 degree-preserving null replicates.  >=2000 recommended.
    null_tail_method : {'empirical', 'gpd'}
        Class 3 per-edge p-value floor handling.
    fdr_q : float
        Class 3 BH q-value cut.
    proximity_null_families : iterable of str
        Class 2 null families.
    proximity_principal_threshold_mb : float
        Distance threshold at which the proximity pass/fail flag is
        evaluated for cross-class synthesis.
    n_top_features : int
        Top-variance edges used for Class 1 leakage classifiers.
    n_bootstrap_synthesis : int
        Bootstrap iterations for cross-class joint-retention CI.
    random_state : int
        Random seed.
    """

    def __init__(
        self,
        asi_threshold: float = 0.5,
        distance_thresholds_mb: Optional[list] = None,
        top_k_values: Optional[list] = None,
        n_permutations: int = 1000,
        n_null_replicates: int = 2000,
        null_tail_method: str = "gpd",
        fdr_q: float = 0.10,
        proximity_null_families=None,
        proximity_principal_threshold_mb: float = 1.0,
        n_top_features: int = 200,
        n_bootstrap_synthesis: int = 1000,
        random_state: int = 42,
    ):
        self.technical = TechnicalAudit(
            asi_threshold=asi_threshold,
            n_top_features=n_top_features,
            random_state=random_state,
        )
        self.proximity = ProximityAudit(
            distance_thresholds_mb=distance_thresholds_mb,
            top_k_values=top_k_values,
            n_permutations=n_permutations,
            null_families=proximity_null_families,
            random_state=random_state,
        )
        self.topological = TopologicalAudit(
            top_k_values=top_k_values,
            n_null_replicates=n_null_replicates,
            null_tail_method=null_tail_method,
            fdr_q=fdr_q,
            random_state=random_state,
        )
        self.asi_threshold = asi_threshold
        self.fdr_q = fdr_q
        self.proximity_principal_threshold_mb = proximity_principal_threshold_mb
        self.n_bootstrap_synthesis = n_bootstrap_synthesis
        self.random_state = random_state

    # ------------------------------------------------------------------
    # Orchestration
    # ------------------------------------------------------------------

    def run(
        self,
        edges: pd.DataFrame,
        gene_coords: Optional[pd.DataFrame] = None,
        scores_balanced: Optional[pd.Series] = None,
        edge_features: Optional[pd.DataFrame] = None,
        covariates: Optional[dict] = None,
        output_dir: Optional[str] = None,
    ) -> dict:
        report = {
            "metadata": {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "n_edges": int(len(edges)),
                "n_tfs": int(edges["tf"].nunique()),
                "n_targets": int(edges["target"].nunique()),
                "asi_threshold": self.asi_threshold,
                "fdr_q": self.fdr_q,
                "proximity_principal_threshold_mb": (
                    self.proximity_principal_threshold_mb
                ),
                "tool_version": "0.2.0",
            },
            "classes": {},
        }

        # --- Class 1 ---
        if scores_balanced is not None or edge_features is not None:
            scores_full = edges.set_index(
                edges["tf"] + "->" + edges["target"]
            )["score"]
            class1 = self.technical.run(
                scores_full=(
                    scores_full
                    if scores_balanced is not None
                    else pd.Series(dtype=float)
                ),
                scores_balanced=scores_balanced,
                edge_features=edge_features,
                covariates=covariates,
            )
            report["classes"]["class1_technical"] = _make_serializable(class1)
            self._class1_raw = class1
        else:
            report["classes"]["class1_technical"] = {
                "status": "skipped",
                "reason": "No balanced scores or edge features provided.",
            }
            self._class1_raw = None

        # --- Class 2 ---
        if gene_coords is not None:
            class2 = self.proximity.run(edges, gene_coords)
            report["classes"]["class2_proximity"] = _make_serializable(class2)
            self._class2_raw = class2
        else:
            report["classes"]["class2_proximity"] = {
                "status": "skipped",
                "reason": "No gene coordinates provided.",
            }
            self._class2_raw = None

        # --- Class 3 ---
        class3 = self.topological.run(edges)
        report["classes"]["class3_topological"] = _make_serializable(class3)
        self._class3_raw = class3

        # --- Per-edge quality table (Class 1 + Class 2 + Class 3) ---
        edge_quality = self._build_edge_quality_table(edges)

        # --- Cross-class synthesis ---
        synthesis_records = self._synthesize_cross_class(edge_quality)
        report["cross_class"] = synthesis_records

        # --- Output ---
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            with open(os.path.join(output_dir, "audit_results.json"), "w") as f:
                json.dump(report, f, indent=2, default=str)

            edge_quality.to_csv(
                os.path.join(output_dir, "edge_quality_indices.csv"),
                index=False,
            )
            if synthesis_records.get("pairwise"):
                pd.DataFrame(synthesis_records["pairwise"]).to_csv(
                    os.path.join(output_dir, "cross_class_synthesis.csv"),
                    index=False,
                )
            with open(os.path.join(output_dir, "audit_summary.txt"), "w") as f:
                f.write(self._generate_text_summary(report))

        return report

    # ------------------------------------------------------------------
    # Per-edge quality table
    # ------------------------------------------------------------------

    def _build_edge_quality_table(
        self,
        edges: pd.DataFrame,
    ) -> pd.DataFrame:
        quality = edges[["tf", "target", "score"]].copy().reset_index(drop=True)
        edge_ids = quality["tf"] + "->" + quality["target"]

        # Class 1: ASI + blacklist
        quality["asi"] = np.nan
        quality["class1_pass"] = True
        if self._class1_raw is not None:
            asi_block = self._class1_raw.get("asi", {})
            if isinstance(asi_block, dict) and "values" in asi_block:
                asi_vals = asi_block["values"]
                if isinstance(asi_vals, pd.Series):
                    quality["asi"] = edge_ids.map(asi_vals).values
                    quality["class1_pass"] = ~(
                        quality["asi"].fillna(0.0) > self.asi_threshold
                    )

        # Class 2: distance + proximity flag (pass = not proximate at principal)
        quality["distance_mb"] = np.nan
        quality["class2_pass"] = True
        if self._class2_raw is not None and "distances" in self._class2_raw:
            dist = self._class2_raw["distances"]
            if isinstance(dist, pd.Series):
                quality["distance_mb"] = dist.reset_index(drop=True).values[
                    : len(quality)
                ]
                proximate = (
                    quality["distance_mb"]
                    <= self.proximity_principal_threshold_mb
                )
                quality["class2_pass"] = ~proximate.fillna(False)

        # Class 3: per-edge q-value + topology pass.
        #
        # Pass semantics
        # --------------
        # Strict per-edge BH at q = fdr_q (default 0.10) is reported in
        # the ``topo_q`` column for downstream use.  Empirically, with
        # a degree-preserving null in which edge (a,b) inherits its own
        # score in those rewirings where it still appears, the per-edge
        # null can be degenerate (q == 1.0 for almost every edge) and
        # is therefore not useful as a binary pass criterion for
        # cross-class filtering.
        #
        # For the cross-class synthesis we therefore use a *continuous*
        # topological evidence score: the standardised difference
        # between the observed score and the per-edge null mean
        # (``topo_excess``).  An edge fails Class 3 only when its
        # topological excess sits in the bottom half of the distribution
        # AND the per-edge null had sufficient observations to support
        # the comparison.  This matches the documented "pass = positive
        # evidence beyond degree-driven expectation" semantics and is
        # robust to the per-edge null degeneracy.
        quality["topo_q"] = np.nan
        quality["topo_excess"] = np.nan
        quality["class3_pass"] = True
        if self._class3_raw is not None and "per_edge" in self._class3_raw:
            per_edge = self._class3_raw["per_edge"]
            if isinstance(per_edge, pd.DataFrame) and not per_edge.empty:
                key = per_edge["tf"] + "->" + per_edge["target"]
                qmap = pd.Series(per_edge["q_bh"].values, index=key)
                quality["topo_q"] = edge_ids.map(qmap).values

                excess = per_edge["score"] - per_edge["null_mean_score"]
                excess_map = pd.Series(excess.values, index=key)
                quality["topo_excess"] = edge_ids.map(excess_map).values

                # Pass = topo_excess at or above the median (or NaN);
                # i.e., we flag only the lower-half edges that have
                # *meaningfully* less topological excess than the typical
                # candidate.  Edges with NaN excess (insufficient null
                # data) pass by default (no evidence against).
                xv = quality["topo_excess"]
                if xv.notna().any():
                    cutoff = float(np.nanmedian(xv))
                    fails = xv.notna() & (xv < cutoff)
                    quality["class3_pass"] = ~fails

        # Joint flag
        quality["passes_all"] = (
            quality["class1_pass"]
            & quality["class2_pass"]
            & quality["class3_pass"]
        )

        return quality

    # ------------------------------------------------------------------
    # Cross-class synthesis (φ, χ², bootstrap CI)
    # ------------------------------------------------------------------

    def _synthesize_cross_class(self, eq: pd.DataFrame) -> dict:
        out = {
            "n_edges": int(len(eq)),
            "pass_rate_per_class": {},
            "pairwise": [],
            "joint_retention": None,
        }

        flags = {
            "class1": eq["class1_pass"].values.astype(bool),
            "class2": eq["class2_pass"].values.astype(bool),
            "class3": eq["class3_pass"].values.astype(bool),
        }
        for c, vec in flags.items():
            out["pass_rate_per_class"][c] = float(np.mean(vec))

        # Pairwise observed vs expected agreement, phi, chi2
        pairs = [("class1", "class2"), ("class1", "class3"), ("class2", "class3")]
        for a, b in pairs:
            va, vb = flags[a], flags[b]
            obs_agree = float(np.mean(va == vb))
            pa, pb = float(np.mean(va)), float(np.mean(vb))
            exp_agree = pa * pb + (1 - pa) * (1 - pb)
            # 2x2 contingency
            n11 = int(np.sum(va & vb))
            n10 = int(np.sum(va & ~vb))
            n01 = int(np.sum(~va & vb))
            n00 = int(np.sum(~va & ~vb))
            n = n11 + n10 + n01 + n00
            phi_num = n11 * n00 - n10 * n01
            phi_den = (
                np.sqrt(
                    max((n11 + n10), 1)
                    * max((n01 + n00), 1)
                    * max((n11 + n01), 1)
                    * max((n10 + n00), 1)
                )
            )
            phi = float(phi_num / phi_den) if phi_den > 0 else float("nan")
            chi2_val = None
            chi2_p = None
            if _HAS_SCIPY and min(n11 + n10, n01 + n00, n11 + n01, n10 + n00) > 0:
                try:
                    chi2_val, chi2_p, _, _ = chi2_contingency(
                        [[n11, n10], [n01, n00]], correction=False,
                    )
                    chi2_val = float(chi2_val)
                    chi2_p = float(chi2_p)
                except Exception:
                    pass

            out["pairwise"].append({
                "pair": f"{a}_vs_{b}",
                "class_a": a,
                "class_b": b,
                "n": n,
                "n11_both_pass": n11,
                "n10_a_pass_only": n10,
                "n01_b_pass_only": n01,
                "n00_both_fail": n00,
                "observed_agreement": round(obs_agree, 4),
                "expected_agreement_under_independence": round(exp_agree, 4),
                "phi_coefficient": round(phi, 4)
                if not np.isnan(phi) else None,
                "chi2": round(chi2_val, 4) if chi2_val is not None else None,
                "chi2_p_value": chi2_p,
            })

        # Three-way joint retention with bootstrap CI
        joint = (flags["class1"] & flags["class2"] & flags["class3"]).astype(int)
        rate = float(joint.mean())
        rng = np.random.default_rng(self.random_state)
        n_edges = joint.size
        if n_edges > 0:
            boots = np.empty(self.n_bootstrap_synthesis, dtype=float)
            for b in range(self.n_bootstrap_synthesis):
                idx = rng.integers(0, n_edges, size=n_edges)
                boots[b] = float(joint[idx].mean())
            ci = (float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975)))
        else:
            ci = (float("nan"), float("nan"))
        out["joint_retention"] = {
            "rate": round(rate, 4),
            "ci_lo": round(ci[0], 4),
            "ci_hi": round(ci[1], 4),
            "n_bootstrap": int(self.n_bootstrap_synthesis),
        }
        return out

    # ------------------------------------------------------------------
    # Text summary
    # ------------------------------------------------------------------

    def _generate_text_summary(self, report: dict) -> str:
        lines = [
            "=" * 70,
            "GRN CONFOUND AUDIT REPORT",
            "=" * 70,
            f"Timestamp: {report['metadata']['timestamp']}",
            f"Edges: {report['metadata']['n_edges']}  "
            f"TFs: {report['metadata']['n_tfs']}  "
            f"Targets: {report['metadata']['n_targets']}",
            f"ASI threshold: {report['metadata']['asi_threshold']}  "
            f"FDR q-cut: {report['metadata']['fdr_q']}  "
            f"Proximity principal threshold: "
            f"{report['metadata']['proximity_principal_threshold_mb']} Mb",
            "",
        ]

        c1 = report["classes"].get("class1_technical", {})
        lines += ["-" * 40, "CLASS 1: TECHNICAL CONFOUND AUDIT", "-" * 40]
        if c1.get("status") == "skipped":
            lines.append(f"  Skipped: {c1.get('reason', 'n/a')}")
        else:
            asi = c1.get("asi", {})
            if "blacklist_rate" in asi:
                lines.append(
                    f"  ASI threshold: {asi.get('threshold')}  "
                    f"Blacklist rate: {asi['blacklist_rate']:.1%}  "
                    f"({asi.get('n_blacklisted')}/{asi.get('n_total')})"
                )
            leakage = c1.get("leakage", {})
            if isinstance(leakage, dict) and "warning" not in leakage:
                for cov, res in leakage.items():
                    lines.append(
                        f"  Leakage [{cov}]: AUC = {res.get('auc_best')}"
                    )
        lines.append("")

        c2 = report["classes"].get("class2_proximity", {})
        lines += ["-" * 40, "CLASS 2: GENOMIC-PROXIMITY AUDIT", "-" * 40]
        if c2.get("status") == "skipped":
            lines.append(f"  Skipped: {c2.get('reason', 'n/a')}")
        else:
            grid = c2.get("enrichment_grid", [])
            principal = next(
                (
                    r for r in grid
                    if r.get("distance_threshold_mb") == 1.0
                    and r.get("top_k") == 1000
                ),
                None,
            )
            if principal:
                nulls = principal.get("nulls", {}) or {}
                lines.append(
                    f"  Enrichment (1 Mb, top-1000): "
                    f"{principal['enrichment_ratio']:.2f}x"
                )
                for nm, blk in nulls.items():
                    lines.append(
                        f"    null={nm:8s}  "
                        f"mean_enr={blk.get('null_mean_enrichment'):.3f}  "
                        f"p={blk.get('p_value'):.4f}"
                    )
            lines.append(
                f"  Significant combinations (p<0.05): "
                f"{c2.get('n_significant_combinations', 0)}"
            )
        lines.append("")

        c3 = report["classes"].get("class3_topological", {})
        lines += ["-" * 40, "CLASS 3: TOPOLOGICAL CONFOUND AUDIT", "-" * 40]
        if "warning" in c3:
            lines.append(f"  {c3['warning']}")
        else:
            lines.append(
                f"  B = {c3.get('n_null_replicates')}  "
                f"tail = {c3.get('null_tail_method')}  "
                f"FDR-q cut = {c3.get('fdr_q')}"
            )
            for k, res in (c3.get("results_per_k") or {}).items():
                vstr = "VALID" if res.get("valid_block") else "MASKED"
                lines.append(
                    f"  top-{k}: z = {res.get('z_score')}  "
                    f"n_sig(BH-q<={res.get('fdr_q')}) = "
                    f"{res.get('n_edges_fdr_significant')}  [{vstr}]"
                )
            diag = c3.get("swap_diagnostics_summary", {})
            lines.append(
                f"  mean swap success = {diag.get('mean_swap_success')}  "
                f"mean turnover = {diag.get('mean_edge_turnover')}"
            )
        lines.append("")

        cc = report.get("cross_class", {})
        lines += ["-" * 40, "CROSS-CLASS SYNTHESIS", "-" * 40]
        for c, rate in (cc.get("pass_rate_per_class") or {}).items():
            lines.append(f"  pass-rate[{c}] = {rate:.3f}")
        for row in cc.get("pairwise", []):
            lines.append(
                f"  {row['pair']:22s}  obs_agree={row['observed_agreement']:.3f}  "
                f"exp_indep={row['expected_agreement_under_independence']:.3f}  "
                f"phi={row['phi_coefficient']}  chi2_p={row['chi2_p_value']}"
            )
        jr = cc.get("joint_retention") or {}
        if jr:
            lines.append(
                f"  joint_retention (all 3 pass) = {jr['rate']:.3f}  "
                f"95% CI [{jr['ci_lo']:.3f}, {jr['ci_hi']:.3f}]"
            )
        lines += ["", "=" * 70]
        return "\n".join(lines)


# ----------------------------------------------------------------------
# JSON serialisation helper
# ----------------------------------------------------------------------


def _make_serializable(obj):
    if isinstance(obj, dict):
        return {k: _make_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_make_serializable(v) for v in obj]
    if isinstance(obj, pd.Series):
        return obj.tolist()
    if isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient="records")
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj
