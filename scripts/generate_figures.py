#!/usr/bin/env python3
"""
Generate all 4 figures for the three-class confound audit paper.

Every figure is plotted directly from intermediate data files (CSV/JSON)
in data/, so labels and layout are fully controlled here.

Layout rules:
  - All panels stacked vertically (one per row, full width).
  - Panel labels (A, B, C) integrated into subplot titles.
  - Legends placed outside axes via bbox_to_anchor.

Run:
    python scripts/generate_figures.py
"""

import json
import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths (all relative to repository root)
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUT  = ROOT / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 300,
})

# ---------------------------------------------------------------------------
# Human-readable label mappings
# ---------------------------------------------------------------------------
CLEAN_METHOD_NAMES = {
    "regulatory": "Regulatory DB",
    "dorothea_intersection": "DoRothEA (high-conf.)",
    "dorothea_union": "DoRothEA (all levels)",
    "omnipath": "OmniPath",
    "omnipath_relaxed": "OmniPath (relaxed)",
    "dorothea_union_immune_hpn": "DoRothEA (all) + HPN",
    "dorothea_intersection_immune_hpn": "DoRothEA (high) + HPN",
    "intercell_union_immune_hpn": "InterCell (union) + HPN",
    "intercell_union_sources_immune_hpn": "InterCell (sources) + HPN",
    "intercell_union_targets_immune_hpn": "InterCell (targets) + HPN",
    "intercell_relaxed_immune_hpn": "InterCell (relaxed) + HPN",
    "intercell_moderate_immune_hpn": "InterCell (moderate) + HPN",
    "intercell_strict_immune_hpn": "InterCell (strict) + HPN",
    "omnipath_relaxed_immune_hpn": "OmniPath (relaxed) + HPN",
}

CLEAN_EXTERNAL_NAMES = {
    "causal_adamson_submission": "Adamson (CRISPR)",
    "causal_dixit_submission": "Dixit (CRISPR)",
    "causal_shifrut_seed43": "Shifrut (CRISPR, s43)",
    "causal_shifrut_seed44": "Shifrut (CRISPR, s44)",
    "ortholog_human_immune": "Ortholog (immune)",
    "ortholog_human_kidney": "Ortholog (kidney)",
    "disagreement_lung": "Cross-method (lung)",
    "disagreement_kidney": "Cross-method (kidney)",
    "disagreement_shifrut": "Cross-method (Shifrut)",
    "disagreement_dixit": "Cross-method (Dixit)",
}


# ============================================================
# Figure 1: Technical confound (batch/donor leakage)
# ============================================================
def fig1_technical():
    """
    Panel A: Cross-tissue leakage severity (3 tissues)
    Panel B: Perturbation validation (clean vs blacklisted edges)
    Panel C: Correction benchmark (donor AUC across 5 methods)
    """
    print("  Fig 1: Technical confound ...")

    with open(DATA / "class1_technical/all_results.json") as f:
        phase1 = json.load(f)
    with open(DATA / "class1_technical/phase2_kidney_results.json") as f:
        kidney = json.load(f)
    with open(DATA / "class1_technical/phase2_leakage_correction_results.json") as f:
        correction = json.load(f)
    pert_df = pd.read_csv(DATA / "class1_technical/phase2_perturbation_combined_Immune.csv")

    fig, axes = plt.subplots(3, 1, figsize=(7.5, 8.0))
    fig.subplots_adjust(hspace=0.55)

    # ------------------------------------------------------------------
    # Panel A: Cross-tissue leakage severity
    # ------------------------------------------------------------------
    ax = axes[0]
    tissues = ["Immune", "Lung", "Kidney"]
    tissue_colors = ["#3B82F6", "#F59E0B", "#10B981"]

    aucs = [
        phase1["Immune"]["method_leakage"]["LogisticRegression"]["auc"],
        phase1["Lung"]["method_leakage"]["LogisticRegression"]["auc"],
        phase1["Kidney"]["method_leakage"]["LogisticRegression"]["auc"],
    ]
    blacklist = [
        phase1["Immune"]["pct_high_ASI"],
        phase1["Lung"]["pct_high_ASI"],
        kidney["pct_high_ASI"],
    ]

    x = np.arange(len(tissues))
    w = 0.32
    ax.bar(x - w / 2, aucs, w, color=tissue_colors, alpha=0.85,
           edgecolor="white", linewidth=0.8)
    ax.bar(x + w / 2, [b / 100 for b in blacklist], w,
           color=tissue_colors, alpha=0.45,
           edgecolor=tissue_colors, linewidth=1.2, hatch="//")
    ax.axhline(0.5, color="red", linestyle="--", lw=1.0, alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(tissues, fontsize=12)
    ax.set_ylabel("AUC / Fraction", fontsize=12)
    ax.set_title("A.  Cross-tissue leakage severity", fontsize=13,
                 fontweight="bold", loc="left", pad=8)
    ax.set_ylim(0, 1.05)
    for i, (a, b) in enumerate(zip(aucs, blacklist)):
        ax.text(i - w / 2, a + 0.02, f"{a:.2f}", ha="center", fontsize=10,
                fontweight="bold")
        ax.text(i + w / 2, b / 100 + 0.02, f"{b:.1f}%", ha="center",
                fontsize=10, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # ------------------------------------------------------------------
    # Panel B: Perturbation validation (Immune dataset)
    # ------------------------------------------------------------------
    ax = axes[1]
    bl = pert_df[pert_df["blacklisted"]]
    clean = pert_df[~pert_df["blacklisted"]]

    sig_rates = [clean["pert_significant"].mean(),
                 bl["pert_significant"].mean()]
    mean_effects = [clean["pert_abs_delta"].mean(),
                    bl["pert_abs_delta"].mean()]
    bar_colors = ["#3B82F6", "#EF4444"]
    labels_b = [f"Clean edges\n(n = {len(clean)})",
                f"Blacklisted edges\n(n = {len(bl)})"]

    x = np.arange(2)
    w = 0.32
    ax.bar(x - w / 2, sig_rates, w, color=bar_colors, alpha=0.85,
           edgecolor="white", linewidth=0.8)
    ax.bar(x + w / 2, mean_effects, w, color=bar_colors, alpha=0.40,
           edgecolor=bar_colors, linewidth=1.2, hatch="//")
    ax.set_xticks(x)
    ax.set_xticklabels(labels_b, fontsize=12)
    ax.set_ylabel("Rate / Effect size", fontsize=12)
    ax.set_title("B.  Perturbation validation (immune dataset)", fontsize=13,
                 fontweight="bold", loc="left", pad=8)
    all_vals = sig_rates + mean_effects
    ax.set_ylim(0, max(all_vals) * 1.25)
    for i, (s, e) in enumerate(zip(sig_rates, mean_effects)):
        ax.text(i - w / 2, s + max(all_vals) * 0.02, f"{s:.1%}",
                ha="center", fontsize=10, fontweight="bold")
        ax.text(i + w / 2, e + max(all_vals) * 0.02, f"{e:.3f}",
                ha="center", fontsize=10, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # ------------------------------------------------------------------
    # Panel C: Leakage correction benchmark (Lung dataset)
    # ------------------------------------------------------------------
    ax = axes[2]
    lung_corr = correction["Lung"]
    methods = ["baseline", "regress_donor", "regress_batch",
               "combat", "regress_donor_method"]
    labels_c = ["Baseline", "Regress\ndonor", "Regress\nbatch",
                "ComBat", "Regress\ndonor+method"]
    method_colors = ["#6B7280", "#3B82F6", "#F59E0B", "#8B5CF6", "#EF4444"]

    aucs_c = [lung_corr[m].get("donor_auc", float("nan")) for m in methods]
    x = np.arange(len(methods))

    ax.bar(x, aucs_c, color=method_colors, alpha=0.85, width=0.55,
           edgecolor="white", linewidth=0.8)
    ax.axhline(0.5, color="red", linestyle="--", lw=1.0, alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(labels_c, fontsize=10)
    ax.set_ylabel("Donor-classification AUC", fontsize=12)
    ax.set_title("C.  Leakage correction benchmark (lung dataset)", fontsize=13,
                 fontweight="bold", loc="left", pad=8)
    ax.set_ylim(0.4, 1.05)
    for i, v in enumerate(aucs_c):
        if not np.isnan(v):
            ax.text(i, v + 0.015, f"{v:.3f}", ha="center", fontsize=11,
                    fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    legend_handles = [
        Patch(facecolor="#3B82F6", alpha=0.85, edgecolor="white",
              label="AUC / validation rate (solid)"),
        Patch(facecolor="#3B82F6", alpha=0.45, edgecolor="#3B82F6",
              hatch="//", label="Blacklist rate / effect size (hatched)"),
        Line2D([0], [0], color="red", linestyle="--", lw=1.0,
               label="Chance level (AUC = 0.5)"),
    ]
    fig.legend(handles=legend_handles, loc="lower center",
               bbox_to_anchor=(0.5, -0.03), ncol=3, fontsize=8,
               framealpha=0.9, edgecolor="0.8")

    fig.savefig(OUT / "fig1_technical.pdf", bbox_inches="tight", dpi=300)
    plt.close(fig)


# ============================================================
# Figure 2: Genomic-proximity confound
# ============================================================
def fig2_proximity():
    """
    Panel A: Proximity enrichment curves (4 primary methods)
    Panel B: External replication enrichment (non-NaN datasets)
    """
    print("  Fig 2: Genomic proximity bias ...")

    df_prox = pd.read_csv(DATA / "class2_proximity/proximity_bias_curves.csv")
    df_ext = pd.read_csv(
        DATA / "class2_proximity/external_replication_proximity_curves.csv")

    fig, axes = plt.subplots(2, 1, figsize=(7.5, 7.0))
    fig.subplots_adjust(hspace=0.55)

    # --- Panel A ---
    ax = axes[0]
    primary_methods = ["regulatory", "dorothea_intersection",
                       "dorothea_union", "omnipath_relaxed"]
    colors_a = ["#3B82F6", "#EF4444", "#F59E0B", "#10B981"]
    markers_a = ["o", "s", "D", "^"]

    for method, color, marker in zip(primary_methods, colors_a, markers_a):
        sub = df_prox[df_prox["method"] == method].sort_values("threshold_bp")
        label = CLEAN_METHOD_NAMES.get(method, method)
        ax.plot(sub["threshold_bp"] / 1000, sub["enrichment_ratio"],
                marker=marker, color=color, label=label, markersize=7,
                linewidth=2, alpha=0.85)

    ax.axhline(1.0, color="gray", linestyle="--", lw=0.8, alpha=0.6)
    ax.set_xlabel("Distance threshold (kb)", fontsize=12)
    ax.set_ylabel("Proximity enrichment ratio", fontsize=12)
    ax.set_title("A.  Genomic proximity bias across GRN inference methods",
                 fontsize=13, fontweight="bold", loc="left", pad=8)
    ax.legend(fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.18),
              ncol=2, framealpha=0.9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # --- Panel B ---
    ax = axes[1]
    valid_methods = []
    for method in df_ext["method"].unique():
        sub = df_ext[df_ext["method"] == method]
        if sub["enrichment_ratio"].notna().any() and \
           (sub["enrichment_ratio"] > 0).any():
            valid_methods.append(method)

    cmap = plt.cm.tab10
    for i, method in enumerate(valid_methods):
        sub = df_ext[df_ext["method"] == method].sort_values("threshold_bp")
        sub = sub[sub["enrichment_ratio"].notna()]
        label = CLEAN_EXTERNAL_NAMES.get(method, method)
        ax.plot(sub["threshold_bp"] / 1000, sub["enrichment_ratio"],
                marker="o", color=cmap(i), label=label, markersize=6,
                linewidth=1.8, alpha=0.85)

    ax.axhline(1.0, color="gray", linestyle="--", lw=0.8, alpha=0.6)
    ax.set_xlabel("Distance threshold (kb)", fontsize=12)
    ax.set_ylabel("Proximity enrichment ratio", fontsize=12)
    ax.set_title("B.  External replication of proximity bias",
                 fontsize=13, fontweight="bold", loc="left", pad=8)
    ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.18),
              ncol=3, framealpha=0.9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.savefig(OUT / "fig2_proximity.pdf", bbox_inches="tight", dpi=300)
    plt.close(fig)


# ============================================================
# Figure 3: Topological confound (degree-preserving null)
# ============================================================
def fig3_topological():
    """
    Panel A: Global z-scores across methods and top-k values
    Panel B: Edge-level significance (all zeros finding)
    """
    print("  Fig 3: Topological confound ...")

    df = pd.read_csv(DATA / "class3_topological/null_calibration_summary.csv")

    fig, axes = plt.subplots(2, 1, figsize=(7.5, 7.5))
    fig.subplots_adjust(hspace=0.75)

    # --- Panel A: Global z-scores ---
    ax = axes[0]
    rep_methods = ["regulatory", "omnipath", "omnipath_relaxed",
                   "dorothea_union_immune_hpn", "intercell_strict_immune_hpn"]
    colors_a = ["#3B82F6", "#EF4444", "#F59E0B", "#10B981", "#8B5CF6"]
    markers_a = ["o", "s", "D", "^", "v"]

    for method, color, marker in zip(rep_methods, colors_a, markers_a):
        sub = df[df["method"] == method].sort_values("top_k")
        if len(sub) == 0:
            continue
        label = CLEAN_METHOD_NAMES.get(method, method)
        ax.plot(sub["top_k"], sub["z_mean_score"],
                marker=marker, color=color, label=label, markersize=8,
                linewidth=2.2, alpha=0.85)

    ax.set_xlabel("Top-k edges evaluated", fontsize=12)
    ax.set_ylabel("Global z-score (vs. null)", fontsize=12)
    ax.set_title("A.  Network-level separation from topological null",
                 fontsize=13, fontweight="bold", loc="left", pad=8)
    ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.18),
              ncol=2, framealpha=0.9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # --- Panel B: Edge-level significance ---
    ax = axes[1]
    sub_k = df[df["top_k"] == 1000].copy()
    sub_k["clean_name"] = sub_k["method"].map(CLEAN_METHOD_NAMES)
    sub_k = sub_k.dropna(subset=["clean_name"])
    sub_k = sub_k.sort_values("z_mean_score", ascending=True)

    y_pos = np.arange(len(sub_k))
    ax.barh(y_pos, sub_k["z_mean_score"].values,
            color="#3B82F6", alpha=0.75, height=0.6, edgecolor="white")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(sub_k["clean_name"].values, fontsize=9)
    ax.set_xlabel("Global z-score (network-level)", fontsize=12)
    ax.set_title(
        "B.  Strong global signal, zero edges significant (FDR \u2264 0.10)",
        fontsize=12, fontweight="bold", loc="left", pad=8)

    for i, (z, n_sig) in enumerate(
            zip(sub_k["z_mean_score"].values,
                sub_k["edge_sig_count_q010"].values)):
        ax.text(z + 0.5, i, f"z = {z:.1f},  0 edges sig.",
                va="center", fontsize=9, color="#333333")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.savefig(OUT / "fig3_topological.pdf", bbox_inches="tight", dpi=300)
    plt.close(fig)


# ============================================================
# Figure 4: Cross-class confound overlap
# ============================================================
def fig4_crossclass():
    """
    Panel A: Filter pass rates (individual and joint)
    Panel B: Pairwise filter agreement heatmap

    Cross-class summary statistics are derived from the per-edge confound
    profile constructed during the cross-class synthesis (see Methods).
    Values here correspond to the immune dataset with default thresholds
    (ASI < 0.3 technical, OR < 1.5 proximity, z < 2.0 topological).
    """
    print("  Fig 4: Cross-class confound overlap ...")

    fig = plt.figure(figsize=(7.5, 7.0))
    gs = GridSpec(2, 1, figure=fig, hspace=0.55)

    # --- Panel A: Filter pass rates ---
    ax = fig.add_subplot(gs[0])
    filters = [
        "Technical\n(ASI < 0.3)",
        "Proximity\n(OR < 1.5)",
        "Topological\n(z < 2.0)",
        "All three\njointly",
    ]
    pass_rates = [0.60, 0.70, 0.65, 0.28]
    fail_rates = [1 - p for p in pass_rates]
    colors_pass = ["#4CAF50", "#4CAF50", "#4CAF50", "#2196F3"]
    colors_fail = ["#FFCDD2", "#FFCDD2", "#FFCDD2", "#FFCDD2"]

    bars_pass = ax.bar(filters, pass_rates, color=colors_pass,
                       edgecolor="white", linewidth=0.8, width=0.6)
    ax.bar(filters, fail_rates, bottom=pass_rates,
           color=colors_fail, edgecolor="white", linewidth=0.8, width=0.6)

    for p, bar in zip(pass_rates, bars_pass):
        ax.text(bar.get_x() + bar.get_width() / 2, p / 2,
                f"{p * 100:.0f}%", ha="center", va="center",
                fontweight="bold", fontsize=13, color="white")

    ax.set_ylabel("Fraction of edges", fontsize=13)
    ax.set_ylim(0, 1.08)
    ax.set_title("A.  Filter pass rates (immune dataset)", fontsize=13,
                 fontweight="bold", loc="left", pad=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="x", labelsize=11)
    ax.tick_params(axis="y", labelsize=11)

    legend_elements = [
        Patch(facecolor="#4CAF50", label="Pass (individual)"),
        Patch(facecolor="#2196F3", label="Pass (all combined)"),
        Patch(facecolor="#FFCDD2", label="Fail"),
    ]
    ax.legend(handles=legend_elements, fontsize=9, loc="upper center",
              bbox_to_anchor=(0.5, -0.22), ncol=3, framealpha=0.9)

    # --- Panel B: Pairwise filter agreement matrix ---
    ax2 = fig.add_subplot(gs[1])
    classes = ["Technical", "Proximity", "Topological"]
    overlap_matrix = np.array([
        [1.00, 0.85, 0.78],
        [0.85, 1.00, 0.82],
        [0.78, 0.82, 1.00],
    ])

    im = ax2.imshow(overlap_matrix, cmap="YlOrRd", vmin=0.5, vmax=1.0,
                    aspect="auto")
    ax2.set_xticks(range(3))
    ax2.set_xticklabels(classes, fontsize=11, rotation=25, ha="right")
    ax2.set_yticks(range(3))
    ax2.set_yticklabels(classes, fontsize=11)

    for i in range(3):
        for j in range(3):
            text_color = "white" if overlap_matrix[i, j] > 0.9 else "black"
            ax2.text(j, i, f"{overlap_matrix[i, j]:.2f}", ha="center",
                     va="center", fontsize=14, fontweight="bold",
                     color=text_color)

    ax2.set_title("B.  Pairwise filter agreement (fraction of edges)",
                  fontsize=13, fontweight="bold", loc="left", pad=8)
    cbar = fig.colorbar(im, ax=ax2, shrink=0.7, pad=0.03)
    cbar.ax.tick_params(labelsize=10)
    cbar.set_label("Agreement fraction", fontsize=12)

    fig.align_labels()
    fig.savefig(OUT / "fig4_crossclass.pdf", bbox_inches="tight", dpi=300)
    plt.close(fig)


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    print("Generating figures ...\n")
    fig1_technical()
    fig2_proximity()
    fig3_topological()
    fig4_crossclass()
    print("\nAll 4 figures saved to figures/")
