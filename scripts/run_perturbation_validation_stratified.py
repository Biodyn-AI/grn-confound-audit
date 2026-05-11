#!/usr/bin/env python3
"""
Perturbation-validation re-analysis with additional stratification.

Extends ``run_perturbation_validation.py`` with:

  * Per-TF rate-ratio table (which TFs drive the clean-vs-blacklisted
    rate-ratio gap?).
  * Bootstrap percentile CIs on the per-TF rate ratios.
  * Permutation null on the 2x2 association (1000 permutations of
    blacklist labels, holding the perturbation-significance vector
    fixed).
  * Sensitivity sweep: how does the rate ratio move as the ASI
    threshold is varied in {0.3, 0.4, 0.5, 0.6, 0.7}?

Outputs:

  * ``data/benchmarks/perturbation_validation_by_tf.csv``
  * ``data/benchmarks/perturbation_validation_threshold_sweep.csv``
  * ``data/benchmarks/perturbation_validation_summary.json`` (updated)
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
except ImportError:
    _HAS_SCIPY = False


SRC = os.path.join(
    os.path.dirname(__file__), "..",
    "data", "class1_technical", "phase2_perturbation_combined_Immune.csv",
)
OUT_DIR = os.path.join(
    os.path.dirname(__file__), "..", "data", "benchmarks",
)
os.makedirs(OUT_DIR, exist_ok=True)


def _rate_ratio_bootstrap(df: pd.DataFrame, n_boot: int = 2000, seed: int = 7):
    rng = np.random.default_rng(seed)
    n = len(df)
    out = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        b = df.iloc[idx]
        clean = b[~b["blacklisted"]]
        black = b[b["blacklisted"]]
        rc = clean["pert_significant"].mean() if len(clean) else np.nan
        rb = black["pert_significant"].mean() if len(black) else np.nan
        out[i] = rc / rb if rb > 0 else np.nan
    out = out[~np.isnan(out)]
    return (
        float(np.median(out)),
        float(np.quantile(out, 0.025)),
        float(np.quantile(out, 0.975)),
    )


def _permutation_test(df: pd.DataFrame, n_perm: int = 1000, seed: int = 11):
    rng = np.random.default_rng(seed)
    bl = df["blacklisted"].values
    ps = df["pert_significant"].values
    # observed rate ratio
    rc_obs = ps[~bl].mean() if (~bl).any() else np.nan
    rb_obs = ps[bl].mean() if bl.any() else np.nan
    rr_obs = rc_obs / rb_obs if rb_obs > 0 else np.nan
    perm_rrs = np.empty(n_perm, dtype=float)
    for i in range(n_perm):
        bl_perm = rng.permutation(bl)
        rc = ps[~bl_perm].mean() if (~bl_perm).any() else np.nan
        rb = ps[bl_perm].mean() if bl_perm.any() else np.nan
        perm_rrs[i] = rc / rb if rb > 0 else np.nan
    perm_rrs = perm_rrs[~np.isnan(perm_rrs)]
    p_two = float((np.abs(perm_rrs - 1.0) >= abs(rr_obs - 1.0)).mean())
    return float(rr_obs), p_two, perm_rrs


def main():
    if not os.path.exists(SRC):
        print(f"ERROR: missing input {SRC}", file=sys.stderr)
        sys.exit(1)
    df = pd.read_csv(SRC)
    print(f"Loaded {len(df)} edges; sources: {dict(df['source'].value_counts())}")

    # Permutation null
    rr_obs, p_perm, _ = _permutation_test(df)
    rr_med, rr_lo, rr_hi = _rate_ratio_bootstrap(df)

    # Per-TF rate ratio
    tf_rows = []
    for tf, sub in df.groupby("tf"):
        if len(sub) < 4 or sub["blacklisted"].nunique() < 2:
            continue
        a = int(((~sub["blacklisted"]) & sub["pert_significant"]).sum())
        b = int(((~sub["blacklisted"]) & ~sub["pert_significant"]).sum())
        c = int((sub["blacklisted"] & sub["pert_significant"]).sum())
        d = int((sub["blacklisted"] & ~sub["pert_significant"]).sum())
        rc = a / max(a + b, 1)
        rb = c / max(c + d, 1)
        rr = rc / max(rb, 1e-9)
        p = None
        if _HAS_SCIPY:
            try:
                _, p = fisher_exact([[a, b], [c, d]], alternative="greater")
            except Exception:
                pass
        tf_rows.append({
            "tf": tf, "n_edges": int(len(sub)),
            "n_blacklisted": int(c + d), "n_pert_sig": int(a + c),
            "rate_clean": round(rc, 4), "rate_blacklisted": round(rb, 4),
            "rate_ratio": round(rr, 3),
            "fisher_p_one_sided": p,
        })
    by_tf = pd.DataFrame(tf_rows)
    by_tf_path = os.path.join(OUT_DIR, "perturbation_validation_by_tf.csv")
    by_tf.to_csv(by_tf_path, index=False)
    print(f"Wrote {by_tf_path}  ({len(by_tf)} TFs)")

    # ASI threshold sweep
    sweep_rows = []
    asi_full = df["ASI"].astype(float)
    for thr in (0.3, 0.4, 0.5, 0.6, 0.7):
        sub = df.copy()
        sub["blacklisted"] = (asi_full > thr).values
        if sub["blacklisted"].nunique() < 2:
            continue
        rr_med2, rr_lo2, rr_hi2 = _rate_ratio_bootstrap(sub)
        rr2 = float((
            sub.loc[~sub["blacklisted"], "pert_significant"].mean()
            / max(sub.loc[sub["blacklisted"], "pert_significant"].mean(), 1e-9)
        ))
        sweep_rows.append({
            "asi_threshold": thr,
            "n_blacklisted": int(sub["blacklisted"].sum()),
            "rate_clean": round(
                float(sub.loc[~sub["blacklisted"], "pert_significant"].mean()),
                4,
            ),
            "rate_blacklisted": round(
                float(sub.loc[sub["blacklisted"], "pert_significant"].mean()),
                4,
            ),
            "rate_ratio": round(rr2, 3),
            "bootstrap_95CI_lo": round(rr_lo2, 3),
            "bootstrap_95CI_hi": round(rr_hi2, 3),
        })
    sweep = pd.DataFrame(sweep_rows)
    sweep_path = os.path.join(
        OUT_DIR, "perturbation_validation_threshold_sweep.csv",
    )
    sweep.to_csv(sweep_path, index=False)
    print(f"Wrote {sweep_path}")

    # Augment headline JSON
    headline_path = os.path.join(OUT_DIR, "perturbation_validation_summary.json")
    if os.path.exists(headline_path):
        with open(headline_path) as f:
            j = json.load(f)
    else:
        j = {}
    j["permutation_null"] = {
        "n_permutations": 1000,
        "observed_rate_ratio": round(rr_obs, 3),
        "two_sided_p_value": round(p_perm, 4),
    }
    j["by_tf_count"] = int(len(by_tf))
    j["threshold_sweep_summary"] = sweep_rows
    with open(headline_path, "w") as f:
        json.dump(j, f, indent=2)
    print(f"Updated {headline_path}")

    print("\n--- headline ---")
    print(json.dumps(j["permutation_null"], indent=2))
    print(f"per-TF table: {len(by_tf)} TFs")
    print(f"threshold sweep: {len(sweep)} rows")


if __name__ == "__main__":
    main()
