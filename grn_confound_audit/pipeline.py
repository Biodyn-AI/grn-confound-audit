"""
Integrated three-class confound audit pipeline.

Orchestrates all three audit classes and produces combined results
with cross-class synthesis.
"""

import json
import os
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd

from .technical import TechnicalAudit
from .proximity import ProximityAudit
from .topological import TopologicalAudit


class ConfoundAuditPipeline:
    """Run a complete three-class confound audit on a GRN edge list.

    This is the main entry point for the audit. It accepts a scored edge
    list and optional metadata, runs all applicable audit classes, and
    produces a combined report.

    Parameters
    ----------
    asi_threshold : float
        ASI threshold for technical blacklisting (Class 1).
    distance_thresholds_mb : list of float
        Genomic distance thresholds for proximity audit (Class 2).
    top_k_values : list of int
        Top-k levels for proximity and topological audits.
    n_permutations : int
        Permutations for proximity null models.
    n_null_replicates : int
        Null replicates for topological audit.
    random_state : int
        Random seed.
    """

    def __init__(
        self,
        asi_threshold: float = 0.5,
        distance_thresholds_mb: Optional[list] = None,
        top_k_values: Optional[list] = None,
        n_permutations: int = 1000,
        n_null_replicates: int = 48,
        random_state: int = 42,
    ):
        self.technical = TechnicalAudit(
            asi_threshold=asi_threshold,
            random_state=random_state,
        )
        self.proximity = ProximityAudit(
            distance_thresholds_mb=distance_thresholds_mb,
            top_k_values=top_k_values,
            n_permutations=n_permutations,
            random_state=random_state,
        )
        self.topological = TopologicalAudit(
            top_k_values=top_k_values,
            n_null_replicates=n_null_replicates,
            random_state=random_state,
        )
        self.random_state = random_state

    def run(
        self,
        edges: pd.DataFrame,
        gene_coords: Optional[pd.DataFrame] = None,
        scores_balanced: Optional[pd.Series] = None,
        edge_features: Optional[pd.DataFrame] = None,
        covariates: Optional[dict] = None,
        output_dir: Optional[str] = None,
    ) -> dict:
        """Run the full three-class confound audit.

        Parameters
        ----------
        edges : pd.DataFrame
            Must have columns 'tf', 'target', 'score'.
        gene_coords : pd.DataFrame, optional
            Gene coordinates with columns 'gene', 'chr', 'tss'.
            Required for Class 2 (proximity audit).
        scores_balanced : pd.Series, optional
            Balanced edge scores for ASI computation (Class 1).
            Index should match edge identifiers.
        edge_features : pd.DataFrame, optional
            Cell-by-edge feature matrix for leakage classification (Class 1).
        covariates : dict, optional
            Technical covariate labels for leakage classification (Class 1).
        output_dir : str, optional
            Directory to write results. If None, results are only returned.

        Returns
        -------
        dict
            Combined audit results with per-class summaries and cross-class
            synthesis.
        """
        report = {
            "metadata": {
                "timestamp": datetime.utcnow().isoformat(),
                "n_edges": len(edges),
                "n_tfs": edges["tf"].nunique(),
                "n_targets": edges["target"].nunique(),
                "tool_version": "0.1.0",
            },
            "classes": {},
        }

        # --- Class 1: Technical audit ---
        if scores_balanced is not None or edge_features is not None:
            class1 = self.technical.run(
                scores_full=edges.set_index(
                    edges["tf"] + "->" + edges["target"]
                )["score"] if scores_balanced is not None else pd.Series(dtype=float),
                scores_balanced=scores_balanced,
                edge_features=edge_features,
                covariates=covariates,
            )
            report["classes"]["class1_technical"] = _make_serializable(class1)
        else:
            report["classes"]["class1_technical"] = {
                "status": "skipped",
                "reason": "No balanced scores or edge features provided.",
            }

        # --- Class 2: Proximity audit ---
        if gene_coords is not None:
            class2 = self.proximity.run(edges, gene_coords)
            report["classes"]["class2_proximity"] = _make_serializable(class2)
        else:
            report["classes"]["class2_proximity"] = {
                "status": "skipped",
                "reason": "No gene coordinates provided.",
            }

        # --- Class 3: Topological audit ---
        class3 = self.topological.run(edges)
        report["classes"]["class3_topological"] = _make_serializable(class3)

        # --- Cross-class synthesis ---
        report["cross_class"] = self._synthesize(report)

        # --- Write output ---
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

            # JSON results
            with open(os.path.join(output_dir, "audit_results.json"), "w") as f:
                json.dump(report, f, indent=2, default=str)

            # Per-edge CSV with quality indices
            edge_quality = self._build_edge_quality_table(edges, report)
            edge_quality.to_csv(
                os.path.join(output_dir, "edge_quality_indices.csv"),
                index=False,
            )

            # Human-readable summary
            summary = self._generate_text_summary(report)
            with open(os.path.join(output_dir, "audit_summary.txt"), "w") as f:
                f.write(summary)

        return report

    def _synthesize(self, report: dict) -> dict:
        """Cross-class synthesis: independence and joint filter rates."""
        synthesis = {}

        # Count how many classes were actually run
        classes_run = [
            k for k, v in report["classes"].items()
            if v.get("status") != "skipped"
        ]
        synthesis["classes_run"] = classes_run
        synthesis["n_classes_run"] = len(classes_run)

        return synthesis

    def _build_edge_quality_table(
        self,
        edges: pd.DataFrame,
        report: dict,
    ) -> pd.DataFrame:
        """Build per-edge quality index table combining all classes."""
        quality = edges[["tf", "target", "score"]].copy()

        # Class 1: ASI
        class1 = report["classes"].get("class1_technical", {})
        if "asi" in class1 and "values" in class1["asi"]:
            edge_ids = edges["tf"] + "->" + edges["target"]
            asi_vals = class1["asi"]["values"]
            if isinstance(asi_vals, pd.Series):
                quality["asi"] = edge_ids.map(asi_vals).values
                quality["blacklisted"] = quality["asi"] > self.technical.asi_threshold
            else:
                quality["asi"] = np.nan
                quality["blacklisted"] = False
        else:
            quality["asi"] = np.nan
            quality["blacklisted"] = False

        # Class 2: proximity flag
        class2 = report["classes"].get("class2_proximity", {})
        if "distances" in class2:
            dist = class2["distances"]
            if isinstance(dist, pd.Series):
                quality["distance_mb"] = dist.values[:len(quality)]
                quality["proximate_1mb"] = quality["distance_mb"] <= 1.0
            else:
                quality["distance_mb"] = np.nan
                quality["proximate_1mb"] = False
        else:
            quality["distance_mb"] = np.nan
            quality["proximate_1mb"] = False

        # Joint quality flag: passes all available filters
        quality["passes_all"] = ~quality["blacklisted"] & ~quality["proximate_1mb"]

        return quality

    def _generate_text_summary(self, report: dict) -> str:
        """Generate a human-readable audit summary."""
        lines = [
            "=" * 70,
            "GRN CONFOUND AUDIT REPORT",
            "=" * 70,
            f"Timestamp: {report['metadata']['timestamp']}",
            f"Edges: {report['metadata']['n_edges']}",
            f"TFs: {report['metadata']['n_tfs']}",
            f"Targets: {report['metadata']['n_targets']}",
            "",
        ]

        # Class 1
        c1 = report["classes"].get("class1_technical", {})
        lines.append("-" * 40)
        lines.append("CLASS 1: TECHNICAL CONFOUND AUDIT")
        lines.append("-" * 40)
        if c1.get("status") == "skipped":
            lines.append(f"  Skipped: {c1.get('reason', 'N/A')}")
        else:
            asi = c1.get("asi", {})
            if "blacklist_rate" in asi:
                lines.append(f"  ASI threshold: {asi.get('threshold', 0.5)}")
                lines.append(f"  Blacklist rate: {asi['blacklist_rate']:.1%}")
                lines.append(f"  Blacklisted edges: {asi.get('n_blacklisted', 'N/A')}/{asi.get('n_total', 'N/A')}")
            leakage = c1.get("leakage", {})
            if isinstance(leakage, dict) and "warning" not in leakage:
                for cov, res in leakage.items():
                    lines.append(f"  Leakage ({cov}): AUC = {res.get('auc_best', 'N/A')}")
        lines.append("")

        # Class 2
        c2 = report["classes"].get("class2_proximity", {})
        lines.append("-" * 40)
        lines.append("CLASS 2: GENOMIC-PROXIMITY AUDIT")
        lines.append("-" * 40)
        if c2.get("status") == "skipped":
            lines.append(f"  Skipped: {c2.get('reason', 'N/A')}")
        else:
            lines.append(f"  Edges with coordinates: {c2.get('n_with_coords', 'N/A')}")
            grid = c2.get("enrichment_grid", [])
            if grid:
                # Show 1 Mb / top-1000 if available
                target = [r for r in grid if r.get("distance_threshold_mb") == 1.0 and r.get("top_k") == 1000]
                if target:
                    r = target[0]
                    lines.append(f"  Enrichment (1 Mb, top-1000): {r['enrichment_ratio']:.2f}x (p = {r['p_value']:.4f})")
                lines.append(f"  Significant combinations (p < 0.05): {c2.get('n_significant_combinations', 0)}")
        lines.append("")

        # Class 3
        c3 = report["classes"].get("class3_topological", {})
        lines.append("-" * 40)
        lines.append("CLASS 3: TOPOLOGICAL CONFOUND AUDIT")
        lines.append("-" * 40)
        if c3.get("status") == "skipped" or "warning" in c3:
            lines.append(f"  {c3.get('warning', c3.get('reason', 'N/A'))}")
        else:
            rpk = c3.get("results_per_k", {})
            for k, res in rpk.items():
                z = res.get("z_score", "N/A")
                valid = "VALID" if res.get("valid_block") else "MASKED"
                lines.append(f"  top-{k}: z = {z}, [{valid}]")
            diag = c3.get("swap_diagnostics_summary", {})
            lines.append(f"  Mean swap success: {diag.get('mean_swap_success', 'N/A')}")
            lines.append(f"  Mean edge turnover: {diag.get('mean_edge_turnover', 'N/A')}")
        lines.append("")

        # Cross-class
        lines.append("-" * 40)
        lines.append("CROSS-CLASS SYNTHESIS")
        lines.append("-" * 40)
        cc = report.get("cross_class", {})
        lines.append(f"  Classes run: {cc.get('n_classes_run', 0)}/3")
        lines.append("")
        lines.append("=" * 70)

        return "\n".join(lines)


def _make_serializable(obj):
    """Recursively convert numpy/pandas types to JSON-serializable Python types."""
    if isinstance(obj, dict):
        return {k: _make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_make_serializable(v) for v in obj]
    elif isinstance(obj, pd.Series):
        return obj.tolist()
    elif isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient="records")
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    return obj
