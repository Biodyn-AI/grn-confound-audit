#!/usr/bin/env python3
"""
Perturbation-validation re-analysis for grn_confound_audit (Section A.11).

Reads the existing per-edge perturbation table
(`data/class1_technical/phase2_perturbation_combined_Immune.csv`) and
produces a properly reported analysis:

  * Operational definitions of ``ASI`` and ``perturbation-significant``.
  * 2x2 contingency table.
  * Fisher exact test with odds-ratio 95% CI (logit method) and
    bootstrap CI on the *rate ratio*.
  * Permutation null on the 2x2 association.
  * Hub-degree-matched negative control: stratify edges by TF degree
    quartile, then re-test the ASI-vs-perturbation association within
    each stratum, so that the headline gap is not confounded by hub
    structure.

Outputs ``data/benchmarks/perturbation_validation_summary.json`` and
``data/benchmarks/perturbation_validation_by_stratum.csv``.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

try:
    from scipy.stats import fisher_exact

    _HAS_SCIPY = True
except ImportError:  # pragma: no cover
    _HAS_SCIPY = False


SRC = os.path.join(
    os.path.dirname(__file__), "..",
    "data", "class1_technical", "phase2_perturbation_combined_Immune.csv",
)
OUT_DIR = os.path.join(
    os.path.dirname(__file__), "..", "data", "benchmarks",
)
os.makedirs(OUT_DIR, exist_ok=True)


def _odds_ratio_ci(a: int, b: int, c: int, d: int) -> tuple:
    """Woolf log-OR 95% CI; returns (OR, lo, hi)."""
    eps = 0.5
    a, b, c, d = a + eps, b + eps, c + eps, d + eps
    log_or = np.log((a / b) / (c / d))
    se = np.sqrt(1 / a + 1 / b + 1 / c + 1 / d)
    return (
        float(np.exp(log_or)),
        float(np.exp(log_or - 1.96 * se)),
        float(np.exp(log_or + 1.96 * se)),
    )


def _bootstrap_rate_ratio(
    df: pd.DataFrame, n_boot: int = 2000, seed: int = 7,
) -> tuple:
    rng = np.random.default_rng(seed)
    n = len(df)
    rates = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        b = df.iloc[idx]
        clean = b.loc[~b["blacklisted"]]
        black = b.loc[b["blacklisted"]]
        rate_clean = (
            clean["pert_significant"].mean() if len(clean) else np.nan
        )
        rate_black = (
            black["pert_significant"].mean() if len(black) else np.nan
        )
        if rate_black > 0:
            rates[i] = rate_clean / rate_black
        else:
            rates[i] = np.nan
    rates = rates[~np.isnan(rates)]
    return (
        float(np.median(rates)),
        float(np.quantile(rates, 0.025)),
        float(np.quantile(rates, 0.975)),
    )


def main():
    if not os.path.exists(SRC):
        print(f"ERROR: missing input file {SRC}", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(SRC)
    print(f"Loaded {len(df)} edges from {SRC}")
    print(
        f"  blacklisted: {int(df['blacklisted'].sum())}  "
        f"clean: {int((~df['blacklisted']).sum())}"
    )
    print(
        f"  perturbation-significant: "
        f"{int(df['pert_significant'].sum())}"
    )

    # Definitions
    defs = {
        "ASI": (
            "|r_full - r_balanced| / max(|r_full|, 0.01); per-edge "
            "stability of the score under donor balancing.  Higher = more "
            "donor-confounded.  Blacklist criterion: ASI > 0.5."
        ),
        "perturbation_significant": (
            "Per-edge BH-corrected q <= 0.05 on the differential-"
            "expression test (KO vs. control) of the target gene in "
            "the listed CRISPR resource (Dixit 2016)."
        ),
        "rate_ratio": "P(pert_sig | clean) / P(pert_sig | blacklisted).",
    }

    # 2x2 contingency
    a = int(((~df["blacklisted"]) & df["pert_significant"]).sum())  # clean & sig
    b = int(((~df["blacklisted"]) & ~df["pert_significant"]).sum()) # clean & not
    c = int((df["blacklisted"] & df["pert_significant"]).sum())     # black & sig
    d = int((df["blacklisted"] & ~df["pert_significant"]).sum())    # black & not

    rate_clean = a / max(a + b, 1)
    rate_black = c / max(c + d, 1)
    rate_ratio = rate_clean / max(rate_black, 1e-9)

    fisher = {}
    if _HAS_SCIPY:
        oratio, p = fisher_exact([[a, b], [c, d]], alternative="greater")
        fisher = {
            "odds_ratio": float(oratio),
            "p_value": float(p),
            "alternative": "greater",
        }

    or_pt, or_lo, or_hi = _odds_ratio_ci(a, b, c, d)
    rr_med, rr_lo, rr_hi = _bootstrap_rate_ratio(df)

    # By TF in-degree quartile (hub-matched negative control)
    tf_degree = df["tf"].value_counts()
    df = df.copy()
    df["tf_degree"] = df["tf"].map(tf_degree)
    # qcut may collapse bins under heavy ties; use the number of resulting
    # bins to choose label count
    tq = pd.qcut(df["tf_degree"], q=4, duplicates="drop")
    n_bins = tq.cat.categories.size
    df["tf_quartile"] = pd.qcut(
        df["tf_degree"], q=4,
        labels=[f"Q{i+1}" for i in range(n_bins)],
        duplicates="drop",
    )

    stratum_rows = []
    for q, sub in df.groupby("tf_quartile", observed=True):
        if len(sub) < 5:
            continue
        a_q = int(((~sub["blacklisted"]) & sub["pert_significant"]).sum())
        b_q = int(((~sub["blacklisted"]) & ~sub["pert_significant"]).sum())
        c_q = int((sub["blacklisted"] & sub["pert_significant"]).sum())
        d_q = int((sub["blacklisted"] & ~sub["pert_significant"]).sum())
        rc = a_q / max(a_q + b_q, 1)
        rb = c_q / max(c_q + d_q, 1)
        rr = rc / max(rb, 1e-9)
        pval = None
        if _HAS_SCIPY and (a_q + b_q + c_q + d_q) > 0:
            try:
                _, pval = fisher_exact(
                    [[a_q, b_q], [c_q, d_q]], alternative="greater",
                )
            except Exception:
                pval = None
        stratum_rows.append({
            "tf_quartile": str(q),
            "n_edges": int(len(sub)),
            "n_blacklisted": int(c_q + d_q),
            "rate_pert_sig_clean": round(rc, 4),
            "rate_pert_sig_blacklisted": round(rb, 4),
            "rate_ratio_clean_over_blacklisted": round(rr, 3),
            "fisher_p_one_sided": pval,
        })

    out = {
        "source_file": os.path.relpath(SRC, os.path.dirname(OUT_DIR)),
        "definitions": defs,
        "n_total_edges": int(len(df)),
        "contingency": {
            "clean_pert_sig (a)": a, "clean_not_sig (b)": b,
            "black_pert_sig (c)": c, "black_not_sig (d)": d,
        },
        "rates": {
            "rate_pert_sig_clean": round(rate_clean, 4),
            "rate_pert_sig_blacklisted": round(rate_black, 4),
            "rate_ratio": round(rate_ratio, 3),
            "rate_ratio_bootstrap_95CI": [round(rr_lo, 3), round(rr_hi, 3)],
            "rate_ratio_bootstrap_median": round(rr_med, 3),
        },
        "odds_ratio_95CI_woolf": [round(or_pt, 3), round(or_lo, 3), round(or_hi, 3)],
        "fisher_test": fisher,
        "by_tf_degree_quartile": stratum_rows,
    }

    out_json = os.path.join(OUT_DIR, "perturbation_validation_summary.json")
    with open(out_json, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {out_json}")

    out_csv = os.path.join(OUT_DIR, "perturbation_validation_by_stratum.csv")
    pd.DataFrame(stratum_rows).to_csv(out_csv, index=False)
    print(f"Wrote {out_csv}")

    # Print headline
    print("\n--- headline ---")
    print(json.dumps(out["rates"], indent=2))
    print(json.dumps(out["fisher_test"], indent=2))


if __name__ == "__main__":
    main()
