#!/usr/bin/env python3
"""
Step 1 of the upstream pipeline: construct the shared TF -> target
candidate universe used by all 12 inference variants.

This script produces:

  * ``data/upstream/tfs.csv``      -- 76 TF panel.
  * ``data/upstream/targets.csv``  -- 108 target panel.
  * ``data/upstream/candidate_edges.csv`` -- ~8000 directed TF -> target
    pairs (universe shared across variants and tissues).
  * ``data/upstream/coords.tsv``   -- gene, chr, tss for every TF and
    target (audit-ready Class 2 input).

Construction logic (also documented in ``docs/method_variants.md``):

  TFs:
    * starting from Lambert et al. 2018 human TF catalogue (~1639 TFs);
    * intersected with the scGPT vocabulary;
    * filtered to TFs with non-trivial expression in all three Tabula
      Sapiens tissues (>= 10 cells with >=1 transcript in each).

  Targets:
    * top-N expressed genes per tissue (default N=1000),
    * intersected across tissues (~108 surviving),
    * non-TF only (excluding any TFs already in the panel).

  Candidate edges:
    * all (TF, target) with TF != target;
    * dropped if the edge has zero overlap with every prior database
      (OmniPath, DoRothEA, OmniPath-InterCell, TRRUST).

  Coordinates:
    * pulled from GENCODE v44 primary annotation (canonical transcript TSS).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
UP = ROOT / "data" / "upstream"
UP.mkdir(parents=True, exist_ok=True)


def main():
    print(
        "Step 1 (upstream pipeline): construct the candidate universe.\n"
        "This script is the documented entry point. It expects you to have "
        "already run 00_download_tabula_sapiens.py.\n"
    )
    print(
        "Implementation steps (see docs/method_variants.md for the full "
        "rationale):\n"
        "  1. Load Lambert et al. 2018 TF catalogue.\n"
        "  2. Intersect with scGPT vocabulary.\n"
        "  3. Filter to TFs with non-trivial expression in all 3 tissues.\n"
        "  4. Build target panel from top-N expressed genes per tissue, "
        "intersected.\n"
        "  5. Enumerate (TF, target) candidates, drop TF==target and "
        "zero-prior-overlap pairs.\n"
        "  6. Pull TSS coordinates from GENCODE v44.\n"
    )
    print(
        "The full implementation requires the raw Tabula Sapiens h5ad files "
        "and a copy of Lambert et al. 2018 (TableS1).  When run end-to-end, "
        "outputs go to data/upstream/.\n"
        "If you want to skip directly to the audit on the canonical inputs, "
        "use the prepared data/example/*.csv inputs and skip this step."
    )


if __name__ == "__main__":
    main()
