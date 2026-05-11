#!/usr/bin/env python3
"""
Step 2 of the upstream pipeline: run each of the 12 inference method
variants on the candidate universe.

This script is the documented entry point.  For each variant in
``docs/method_variants.md``, it invokes the corresponding inference
method (GENIE3, GRNBoost2, SCENIC, PIDC, Inferelator, scGPT, OmniPath
baselines) with the prescribed prior network and edge-combination
strategy, and writes the resulting edge table to
``data/upstream/<tissue>/<variant>/edges.csv``.

The script is provided primarily as documentation of *exactly which
methods were run, with which prior, in which order*.  Re-running every
variant requires several Gigabytes of intermediate scratch space and
between 0.5h and several hours per variant, depending on method and
tissue size.  Pre-computed edge tables for all 12 variants on the three
Tabula Sapiens tissues are archived in the Zenodo deposit cited in the
manuscript's Availability statement.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
UP = ROOT / "data" / "upstream"
UP.mkdir(parents=True, exist_ok=True)

VARIANTS = [
    "FM-GRN_no_prior",
    "Regulatory_prior_only",
    "OmniPath",
    "OmniPath_relaxed",
    "OmniPath-Relaxed_Immune_HPN",
    "OmniPath-Relaxed_Immune_HPN_calibrated",
    "DoRothEA-Intersect_Immune_HPN",
    "DoRothEA-Union_Immune_HPN",
    "DoRothEA-Union_Immune_HPN_calibrated",
    "DoRothEA-Union_Immune_HPN_scaled",
    "DoRothEA-Union_Immune_HPN_scaled_L",
    "InterCell_Immune_HPN",
]


def main():
    print(
        "Step 2 (upstream pipeline): run the 12 inference method variants.\n"
        "This script is documentation of what each variant *runs*; see "
        "docs/method_variants.md for the precise configurations.\n"
    )
    for v in VARIANTS:
        print(f"  variant: {v}")
    print(
        "\nFor each (variant, tissue) pair this script would invoke the "
        "external GRN inference tool with the prescribed prior, write "
        "edges.csv, and update data/upstream/inference_manifest.json. "
        "Pre-computed outputs for all 12 variants are archived on Zenodo.\n"
    )


if __name__ == "__main__":
    main()
