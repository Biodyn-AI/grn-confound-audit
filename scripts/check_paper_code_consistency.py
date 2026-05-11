#!/usr/bin/env python3
"""
Paper <-> code consistency check (Section C.2 of the revision plan).

Verifies that every load-bearing numeric claim in the manuscript matches
the data file it is supposed to come from.  Writes a report to
``data/benchmarks/consistency_report.txt`` and exits with non-zero if any
check fails.

Currently checks:

  1. Perturbation rate ratio in the manuscript == rate ratio in
     ``perturbation_validation_summary.json``.
  2. Joint retention rates per tissue in the manuscript ==
     rates in ``cross_class_synthesis_all_tissues.csv``.
  3. ASI threshold mentioned in the manuscript == default in
     ``technical.py``.
  4. n_null_replicates mentioned in the manuscript == default in
     ``topological.py``.
  5. Default top-k cuts in the manuscript == defaults in
     ``proximity.py``.

This is intentionally small but representative: any drift in these five
will catch the most common ways the paper and the code can diverge.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PAPER = (
    ROOT.parent / "subproject_merged_B_confound_bias_audit"
    / "paper" / "submission_bmc_bioinformatics" / "main.tex"
)
BENCH = ROOT / "data" / "benchmarks"
REPORT = BENCH / "consistency_report.txt"


def main():
    BENCH.mkdir(parents=True, exist_ok=True)
    log = []
    failures = 0

    def check(name: str, ok: bool, expected, found):
        nonlocal failures
        status = "OK" if ok else "MISMATCH"
        log.append(f"[{status}] {name}: expected={expected!r}  found={found!r}")
        if not ok:
            failures += 1

    paper_text = PAPER.read_text() if PAPER.exists() else ""

    # 1. Perturbation rate ratio
    pj = BENCH / "perturbation_validation_summary.json"
    if pj.exists():
        with pj.open() as f:
            pert = json.load(f)
        expected_rr = pert["rates"]["rate_ratio"]
        m = re.search(r"rate ratio is\s*\\textbf\{([\d\.]+)\}", paper_text)
        found = float(m.group(1)) if m else None
        check(
            "perturbation rate ratio",
            abs((found or 0) - expected_rr) < 0.05,
            expected_rr, found,
        )
    else:
        log.append("[SKIP] perturbation rate ratio (no summary file)")

    # 2. Joint retention rate per tissue.  We allow a tolerance of 0.03
    # because the synthesis runs B = 2000 stochastic rewirings and
    # bootstrap CIs on each retention rate span roughly +/- 0.03.
    cc = BENCH / "cross_class_synthesis_all_tissues.csv"
    if cc.exists():
        df = pd.read_csv(cc)
        for tissue in ("Immune", "Lung", "Kidney"):
            sub = df[df["tissue"] == tissue]
            if not len(sub):
                continue
            expected = float(sub["joint_retention_rate"].iloc[0])
            m = re.search(
                rf"{tissue}\s+&\s+[\d\.]+\s+&\s+[\d\.]+\s+&\s+[\d\.]+\s+&\s+"
                rf"([\d\.]+)\s*\[",
                paper_text,
            )
            found = float(m.group(1)) if m else None
            check(
                f"joint retention {tissue}",
                found is not None and abs(found - expected) < 0.03,
                round(expected, 3), found,
            )

    # 3. ASI threshold default
    tech_src = (ROOT / "grn_confound_audit" / "technical.py").read_text()
    m = re.search(r"asi_threshold:\s*float\s*=\s*([\d\.]+)", tech_src)
    code_asi = float(m.group(1)) if m else None
    paper_asi = "ASI~$>0.5$" in paper_text or "ASI $> 0.5$" in paper_text \
        or "ASI > 0.5" in paper_text
    check(
        "ASI threshold (paper says > 0.5)",
        code_asi == 0.5 and paper_asi,
        0.5, code_asi,
    )

    # 4. n_null_replicates default
    topo_src = (ROOT / "grn_confound_audit" / "topological.py").read_text()
    m = re.search(r"n_null_replicates:\s*int\s*=\s*(\d+)", topo_src)
    code_B = int(m.group(1)) if m else None
    paper_B = "B = 2{,}000" in paper_text or "B = 2,000" in paper_text \
        or "B=2000" in paper_text or "B = 2000" in paper_text
    check(
        "n_null_replicates default (paper says 2,000)",
        code_B == 2000 and paper_B,
        2000, code_B,
    )

    # 5. top-k cuts
    prox_src = (ROOT / "grn_confound_audit" / "proximity.py").read_text()
    m = re.search(r"top_k_values or \[(.*?)\]", prox_src)
    code_topk = (
        [int(x) for x in m.group(1).split(",")] if m else None
    )
    paper_topk = (
        "{100, 250, 500, 1000}" in paper_text
        or "\\{100, 250, 500, 1000\\}" in paper_text
        or "100, 250, 500, 1000" in paper_text
    )
    check(
        "top_k defaults (paper says {100, 250, 500, 1000})",
        code_topk == [100, 250, 500, 1000] and paper_topk,
        [100, 250, 500, 1000], code_topk,
    )

    REPORT.write_text("\n".join(log) + "\n")
    print("\n".join(log))
    print(f"\nWrote {REPORT}")
    if failures:
        print(f"FAILED: {failures} consistency check(s) mismatched.")
        sys.exit(1)
    print("All consistency checks passed.")


if __name__ == "__main__":
    main()
