#!/usr/bin/env python3
"""
Figure-generation script for grn_confound_audit.

Every panel is read from a canonical CSV/JSON produced by the audit
pipeline or by ``scripts/run_simulation_benchmarks.py`` /
``scripts/run_perturbation_validation.py``.  No numeric value is
hard-coded.  If a source file is missing the corresponding panel is
left blank with a printed warning so that consistency drift between the
manuscript and the released code becomes immediately visible.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
BENCH = DATA / "benchmarks"
FIG_DIR = ROOT / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def _safe_read_csv(path: Path):
    if not path.exists():
        print(f"  [WARN] missing {path}")
        return None
    return pd.read_csv(path)


def _safe_read_json(path: Path):
    if not path.exists():
        print(f"  [WARN] missing {path}")
        return None
    with path.open() as f:
        return json.load(f)


# ----------------------------------------------------------------------
# Figure 1 -- Technical confound audit
# ----------------------------------------------------------------------


def figure_1_technical():
    out_pdf = FIG_DIR / "fig1_technical.pdf"
    pert = _safe_read_json(BENCH / "perturbation_validation_summary.json")
    leakage = _safe_read_json(DATA / "class1_technical" / "all_results.json") or {}
    corr = _safe_read_json(
        DATA / "class1_technical" / "phase2_leakage_correction_results.json"
    )

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.0))

    # --- Panel A: per-tissue leakage AUC + blacklist rate
    ax = axes[0]
    tissues = ["Immune", "Lung", "Kidney"]
    method_auc, blacklist = [], []
    for t in tissues:
        block = leakage.get(t, {}) or {}
        method_block = block.get("method_leakage", {}) or {}
        aucs = [v.get("auc") for v in method_block.values() if isinstance(v, dict)]
        method_auc.append(float(np.nanmax(aucs)) if aucs else np.nan)
        bl = block.get("asi_blacklist", {})
        blacklist.append(float(bl.get("blacklist_rate", np.nan)))
    x = np.arange(len(tissues))
    w = 0.35
    ax.bar(x - w / 2, method_auc, w, label="method leakage AUC",
           color="#4c72b0")
    ax.bar(x + w / 2, blacklist, w, label="blacklist rate (ASI > 0.5)",
           color="#dd8452", hatch="//")
    ax.axhline(0.5, color="red", linestyle="--", linewidth=1, alpha=0.7,
               label="chance")
    ax.set_xticks(x); ax.set_xticklabels(tissues)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("AUC  /  blacklist fraction")
    ax.set_title("A. Tissue-level leakage")
    ax.legend(loc="lower left", fontsize=8, frameon=False)

    # --- Panel B: perturbation validation with bootstrap CI
    ax = axes[1]
    if pert is not None:
        rates = pert["rates"]
        labels = ["Clean\n(ASI <= 0.5)", "Blacklisted\n(ASI > 0.5)"]
        vals = [rates["rate_pert_sig_clean"], rates["rate_pert_sig_blacklisted"]]
        bars = ax.bar(labels, vals, color=["#55a868", "#c44e52"])
        for bar, v in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.003,
                f"{v:.1%}", ha="center", fontsize=9,
            )
        ax.set_ylim(0, max(vals) * 1.7 if max(vals) > 0 else 1)
        ax.set_ylabel("perturbation-significant rate")
        rr = rates["rate_ratio"]
        rr_lo, rr_hi = rates["rate_ratio_bootstrap_95CI"]
        ax.set_title(
            "B. Perturbation validation\n"
            f"rate ratio = {rr:.2f}  (95% CI {rr_lo:.2f}-{rr_hi:.2f})",
            fontsize=10,
        )
        fp = (pert.get("fisher_test") or {}).get("p_value")
        if fp is not None:
            ax.text(
                0.5, 0.95, f"Fisher one-sided p = {fp:.3f}",
                transform=ax.transAxes, ha="center", fontsize=8,
            )
    else:
        ax.set_title("B. Perturbation validation (missing)")
        ax.set_axis_off()

    # --- Panel C: leakage-correction benchmark (lung)
    ax = axes[2]
    if corr is not None:
        try:
            methods = list(corr.keys())
            after = [
                float(corr[m].get("donor_auc_after",
                                   corr[m].get("auc_after", np.nan)))
                for m in methods
            ]
            ax.barh(methods, after, color="#8172b3")
            ax.axvline(0.5, color="red", linestyle="--", alpha=0.7)
            ax.set_xlabel("Donor classification AUC (after correction)")
            ax.set_title("C. Leakage correction (lung - balanced donors)")
            ax.set_xlim(0, 1.0)
        except Exception as e:
            ax.text(0.05, 0.5, f"correction parse error: {e}",
                    transform=ax.transAxes, fontsize=8)
            ax.set_axis_off()
    else:
        ax.set_axis_off()

    plt.tight_layout()
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig1] wrote {out_pdf}")


# ----------------------------------------------------------------------
# Figure 2 -- Proximity audit
# ----------------------------------------------------------------------


def figure_2_proximity():
    out_pdf = FIG_DIR / "fig2_proximity.pdf"
    curves = _safe_read_csv(DATA / "class2_proximity" / "proximity_bias_curves.csv")
    ext = _safe_read_csv(
        DATA / "class2_proximity" / "external_replication_proximity_curves.csv"
    )

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))

    ax = axes[0]
    if curves is not None:
        sub = curves[(curves["threshold_bp"] == 1_000_000) & (curves["k"] == 1000)]
        sub = sub.head(10).copy()
        methods = sub["method"].tolist()
        src = sub["enrichment_ratio"].astype(float).fillna(0).values
        deg = sub.get("degree_enrichment_ratio",
                      pd.Series([np.nan] * len(sub))).astype(float).fillna(0).values
        x = np.arange(len(methods))
        w = 0.4
        ax.bar(x - w / 2, src, w, label="source-preserving",
               color="#4c72b0")
        ax.bar(x + w / 2, deg, w, label="degree-preserving",
               color="#dd8452")
        ax.axhline(1.0, color="black", linestyle=":", linewidth=0.8)
        ax.set_xticks(x); ax.set_xticklabels(methods, rotation=45, ha="right",
                                              fontsize=8)
        ax.set_ylabel("Proximity enrichment ratio")
        ax.set_title("A. Source- vs. degree-preserving null (top-1000, 1 Mb)")
        ax.legend(loc="upper right", fontsize=8, frameon=False)
    else:
        ax.set_axis_off()

    ax = axes[1]
    if ext is not None:
        groups = ext["external_group"].unique()[:3]
        colors = ["#55a868", "#c44e52", "#8172b3"]
        for g, c in zip(groups, colors):
            sub = ext[(ext["external_group"] == g) & (ext["threshold_bp"] == 500_000)]
            ax.scatter(
                sub["k"], sub["enrichment_ratio"].astype(float),
                label=g, color=c, alpha=0.7, s=18,
            )
        ax.axhline(1.0, color="black", linestyle=":", linewidth=0.8)
        ax.set_xlabel("top-k")
        ax.set_ylabel("Proximity enrichment ratio")
        ax.set_title("B. External replication panels (BH-q within group)")
        ax.legend(loc="upper right", fontsize=8, frameon=False)
    else:
        ax.set_axis_off()

    plt.tight_layout()
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig2] wrote {out_pdf}")


# ----------------------------------------------------------------------
# Figure 3 -- Topological audit (all 12 methods)
# ----------------------------------------------------------------------


def figure_3_topological():
    out_pdf = FIG_DIR / "fig3_topological.pdf"
    summary = _safe_read_csv(
        DATA / "class3_topological" / "null_calibration_summary.csv"
    )
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))

    ax = axes[0]
    if summary is not None:
        sub = summary.copy().sort_values(["method", "top_k"])
        methods = sub["method"].unique()
        cmap = plt.cm.tab20
        for i, m in enumerate(methods):
            ss = sub[sub["method"] == m]
            ax.plot(
                ss["top_k"], ss["z_mean_score"], marker="o", linewidth=1,
                color=cmap(i % 20), label=m, alpha=0.8,
            )
        ax.set_xscale("log")
        ax.set_xlabel("top-k")
        ax.set_ylabel("Global z-score (observed vs. degree-preserving null)")
        ax.set_title("A. Global separation across all methods")
        ax.axhline(0, color="grey", linewidth=0.6, linestyle=":")
        ax.legend(loc="upper right", fontsize=6, ncol=2, frameon=False)
    else:
        ax.set_axis_off()

    ax = axes[1]
    if summary is not None:
        try:
            piv = summary.pivot_table(
                index="method", columns="top_k",
                values="edge_sig_count_q010", aggfunc="mean",
            )
            im = ax.imshow(piv.values, aspect="auto", cmap="Blues")
            ax.set_xticks(np.arange(piv.columns.size))
            ax.set_xticklabels(piv.columns)
            ax.set_yticks(np.arange(piv.index.size))
            ax.set_yticklabels(piv.index, fontsize=7)
            ax.set_xlabel("top-k")
            ax.set_title("B. Per-edge BH(q <= 0.10) significant count")
            vmax = float(np.nanmax(piv.values)) if piv.size else 1.0
            for (i, j), v in np.ndenumerate(piv.values):
                if np.isnan(v):
                    continue
                ax.text(
                    j, i, f"{int(v)}", ha="center", va="center", fontsize=7,
                    color="black" if v < vmax / 2 else "white",
                )
            plt.colorbar(im, ax=ax, shrink=0.7)
        except Exception as e:
            ax.text(0.05, 0.5, f"heatmap error: {e}",
                    transform=ax.transAxes, fontsize=8)
            ax.set_axis_off()
    else:
        ax.set_axis_off()

    plt.tight_layout()
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig3] wrote {out_pdf}")


# ----------------------------------------------------------------------
# Figure 4 -- Cross-class synthesis (read from CSV)
# ----------------------------------------------------------------------


def figure_4_crossclass():
    out_pdf = FIG_DIR / "fig4_crossclass.pdf"
    cc = _safe_read_csv(BENCH / "cross_class_synthesis_all_tissues.csv")
    if cc is None:
        cc = _safe_read_csv(BENCH / "cross_class_synthesis.csv")
    if cc is None:
        print("[fig4] no cross-class synthesis CSV; skipping")
        return

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.0))

    pairs = cc["pair"].unique()
    obs = cc.groupby("pair")["observed_agreement"].mean().reindex(pairs).values
    exp = (
        cc.groupby("pair")["expected_agreement_under_independence"]
        .mean().reindex(pairs).values
    )

    ax = axes[0]
    x = np.arange(len(pairs))
    w = 0.4
    ax.bar(x - w / 2, obs, w, label="observed", color="#4c72b0")
    ax.bar(x + w / 2, exp, w, label="expected (indep.)", color="#dd8452")
    ax.set_xticks(x); ax.set_xticklabels(pairs, rotation=30, ha="right", fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Pairwise agreement")
    ax.set_title("A. Observed vs. expected agreement")
    ax.legend(loc="lower right", fontsize=8, frameon=False)

    ax = axes[1]
    phi = cc.groupby("pair")["phi_coefficient"].mean().reindex(pairs).values
    chi2_p = cc.groupby("pair")["chi2_p_value"].mean().reindex(pairs).values
    bars = ax.bar(pairs, phi, color="#55a868")
    for bar, p in zip(bars, chi2_p):
        if p is None or (isinstance(p, float) and np.isnan(p)):
            label = ""
        else:
            label = f"chi2 p={p:.2g}"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            label, ha="center", fontsize=8,
        )
    ax.axhline(0, color="black", linewidth=0.7)
    ax.tick_params(axis="x", rotation=30)
    ax.set_ylabel("phi coefficient")
    ax.set_title("B. Pairwise phi (with chi2 p)")

    ax = axes[2]
    if "tissue" in cc.columns:
        tissues = cc["tissue"].unique()
        rate = cc.groupby("tissue")["joint_retention_rate"].first().reindex(tissues).values
        ci_lo = cc.groupby("tissue")["joint_retention_ci_lo"].first().reindex(tissues).values
        ci_hi = cc.groupby("tissue")["joint_retention_ci_hi"].first().reindex(tissues).values
        x = np.arange(len(tissues))
        ax.bar(x, rate, color="#8172b3")
        ax.errorbar(
            x, rate, yerr=[rate - ci_lo, ci_hi - rate],
            fmt="none", color="black",
        )
        ax.set_xticks(x); ax.set_xticklabels(tissues)
    else:
        ax.text(0.05, 0.5,
                "Joint retention summary unavailable across tissues",
                transform=ax.transAxes, fontsize=8)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Three-way joint retention rate")
    ax.set_title("C. Joint retention (95% bootstrap CI)")

    plt.tight_layout()
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig4] wrote {out_pdf}")


# ----------------------------------------------------------------------
# Figure 5 -- Tool validation on simulated data (NEW)
# ----------------------------------------------------------------------


def figure_5_simulation():
    out_pdf = FIG_DIR / "fig5_simulation_validation.pdf"
    df = _safe_read_csv(BENCH / "simulation_benchmark_results.csv")
    if df is None:
        print("[fig5] simulation benchmark CSV missing; skipping")
        return

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.0))

    ax = axes[0]
    sub = df[df["scenario"] == "donor_only"]
    if len(sub):
        agg = sub.groupby("donor_strength").agg(
            mean_auroc=("class1_donor_AUROC", "mean"),
            sd_auroc=("class1_donor_AUROC", "std"),
        ).reset_index()
        ax.errorbar(
            agg["donor_strength"], agg["mean_auroc"],
            yerr=agg["sd_auroc"], fmt="o-", color="#c44e52", capsize=4,
        )
        ax.axhline(0.5, color="grey", linestyle=":", linewidth=0.8)
        ax.set_xlabel("Injected donor-confound strength")
        ax.set_ylabel("Class 1 ASI -> truth AUROC")
        ax.set_ylim(0, 1.05)
        ax.set_title("A. Class 1 sensitivity (donor)")

    ax = axes[1]
    sub = df[df["scenario"] == "topology_only"]
    if len(sub):
        agg = sub.groupby("hub_strength").agg(
            mean_auroc=("class3_topology_AUROC", "mean"),
            sd_auroc=("class3_topology_AUROC", "std"),
        ).reset_index()
        ax.errorbar(
            agg["hub_strength"], agg["mean_auroc"],
            yerr=agg["sd_auroc"], fmt="s-", color="#4c72b0", capsize=4,
        )
        ax.axhline(0.5, color="grey", linestyle=":", linewidth=0.8)
        ax.set_xlabel("Injected hub-confound strength")
        ax.set_ylabel("Class 3 (1 - q) -> truth AUROC")
        ax.set_ylim(0, 1.05)
        ax.set_title("B. Class 3 sensitivity (topology)")

    ax = axes[2]
    sub = df.groupby("scenario").agg(
        n=("joint_retention_rate", "count"),
        mean_rate=("joint_retention_rate", "mean"),
        sd_rate=("joint_retention_rate", "std"),
    ).reset_index()
    sub = sub[sub["mean_rate"].notna()]
    if len(sub):
        ax.bar(
            sub["scenario"], sub["mean_rate"], color="#55a868",
            yerr=sub["sd_rate"], capsize=4,
        )
        ax.set_ylim(0, 1.0)
        ax.set_ylabel("Joint retention rate")
        ax.set_title("C. Joint retention by scenario")
        ax.tick_params(axis="x", rotation=20)

    plt.tight_layout()
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig5] wrote {out_pdf}")


# ----------------------------------------------------------------------
# Figure 6 -- Pipeline schematic (NEW)
# ----------------------------------------------------------------------


def figure_6_pipeline_schematic():
    out_pdf = FIG_DIR / "fig6_pipeline.pdf"
    fig, ax = plt.subplots(figsize=(11, 5.4))
    ax.set_axis_off()

    nodes = [
        (0.04, 0.65, 0.18, 0.22, "Raw counts\n+ metadata", "#cfe2f3"),
        (0.04, 0.10, 0.18, 0.22, "TF / target /\ncoord resources", "#cfe2f3"),
        (0.27, 0.40, 0.18, 0.22, "Inference\nmethods (x12)", "#fce5cd"),
        (0.50, 0.72, 0.18, 0.16,
         "Class 1 audit\nper-cell features,\nASI, leakage AUC", "#d9ead3"),
        (0.50, 0.42, 0.18, 0.16,
         "Class 2 audit\nproximity, 3 null\nfamilies, hub strata", "#d9ead3"),
        (0.50, 0.12, 0.18, 0.16,
         "Class 3 audit\ndegree null,\nper-edge BH (GPD tail)", "#d9ead3"),
        (0.74, 0.42, 0.20, 0.40,
         "Cross-class synthesis\nphi, chi2 indep. test,\nbootstrap CI on\n3-way joint retention",
         "#ead1dc"),
        (0.74, 0.10, 0.20, 0.22,
         "Outputs\n- audit_results.json\n- edge_quality_indices.csv\n- cross_class_synthesis.csv\n- audit_summary.txt",
         "#fff2cc"),
    ]
    for x, y, w, h, label, color in nodes:
        ax.add_patch(plt.Rectangle((x, y), w, h, facecolor=color,
                                    edgecolor="black", linewidth=1))
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center",
                fontsize=8.5)

    def arrow(x0, y0, x1, y1):
        ax.annotate(
            "", xy=(x1, y1), xytext=(x0, y0),
            arrowprops=dict(arrowstyle="->", lw=1.0),
        )
    arrow(0.22, 0.76, 0.27, 0.55)
    arrow(0.22, 0.21, 0.27, 0.50)
    arrow(0.45, 0.55, 0.50, 0.80)
    arrow(0.45, 0.50, 0.50, 0.50)
    arrow(0.45, 0.45, 0.50, 0.20)
    arrow(0.68, 0.80, 0.74, 0.70)
    arrow(0.68, 0.50, 0.74, 0.55)
    arrow(0.68, 0.20, 0.74, 0.45)
    arrow(0.84, 0.42, 0.84, 0.32)

    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig6] wrote {out_pdf}")


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------


def main():
    figure_1_technical()
    figure_2_proximity()
    figure_3_topological()
    figure_4_crossclass()
    figure_5_simulation()
    figure_6_pipeline_schematic()
    print("\nAll figures written to", FIG_DIR)


if __name__ == "__main__":
    main()
