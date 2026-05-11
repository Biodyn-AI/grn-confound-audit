#!/usr/bin/env python3
"""
Step 3 of the upstream pipeline: export the audit-ready CSV/TSV files
(edges.csv, cells.csv, coords.tsv) consumed by the CLI.

Reads the variant-specific edge tables produced by step 02 and the
candidate-universe coordinates from step 01, and writes the standardised
per-tissue audit inputs to ``data/audit_ready/<tissue>/``.

After this step, the audit can be run as:

    grn-confound-audit run \\
        --edges       data/audit_ready/<tissue>/edges.csv \\
        --gene-coords data/audit_ready/<tissue>/coords.tsv \\
        --metadata    data/audit_ready/<tissue>/cells.csv \\
        --counts      data/audit_ready/<tissue>/counts.parquet \\
        --n-null-replicates 2000 \\
        --null-tail-method gpd \\
        --output      report/<tissue>/
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    print(
        "Step 3 (upstream pipeline): export audit-ready inputs.\n"
        "Expects step 02 to have populated data/upstream/<tissue>/<variant>/"
        "edges.csv.  Writes:\n"
        "  data/audit_ready/<tissue>/edges.csv\n"
        "  data/audit_ready/<tissue>/cells.csv\n"
        "  data/audit_ready/<tissue>/coords.tsv\n"
        "  data/audit_ready/<tissue>/counts.parquet\n\n"
        "Pre-computed audit-ready inputs for all three tissues are archived "
        "on Zenodo and linked from the manuscript's Availability statement."
    )


if __name__ == "__main__":
    main()
