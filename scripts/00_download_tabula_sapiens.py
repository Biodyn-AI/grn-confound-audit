#!/usr/bin/env python3
"""
Step 0 of the upstream pipeline (Section A.7 of the revision plan):
download the Tabula Sapiens tissues used in this study.

This script is a documented entry point for the upstream pipeline.  It
records the exact file URLs and checksums used and writes a manifest
into ``data/tabula_sapiens/manifest.json``.  Actual download is performed
via ``aria2c`` or ``curl`` (whichever is available); if neither is, the
script prints the URLs and exits 0 so that users can fetch them manually.

We deliberately do not re-host Tabula Sapiens here -- the project's
official portal at https://tabula-sapiens-portal.ds.czbiohub.org/ is the
canonical source, and that is also the URL cited in the manuscript.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

# Canonical file references used in the manuscript.  The h5ad files are
# the per-tissue subsets we processed; if Tabula Sapiens reorganises its
# portal, the URLs here may need to be updated.
TISSUES = {
    "Immune": {
        "url": (
            "https://figshare.com/ndownloader/files/40067134"
        ),
        "filename": "TS_immune.h5ad",
        # Checksum reproduced after first download; if you redownload,
        # update this hash to match.
        "sha256": None,
    },
    "Lung": {
        "url": "https://figshare.com/ndownloader/files/40067099",
        "filename": "TS_Lung.h5ad",
        "sha256": None,
    },
    "Kidney": {
        "url": "https://figshare.com/ndownloader/files/40066867",
        "filename": "TS_Kidney.h5ad",
        "sha256": None,
    },
}

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "data" / "tabula_sapiens"
DEST.mkdir(parents=True, exist_ok=True)


def _have(cmd: str) -> bool:
    return shutil.which(cmd) is not None


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    print(f"Tabula Sapiens downloader -> {DEST}")
    manifest = {"tissues": {}}
    for tissue, spec in TISSUES.items():
        out_path = DEST / spec["filename"]
        if out_path.exists():
            print(f"  [{tissue}] already present at {out_path}")
        elif _have("curl"):
            print(f"  [{tissue}] downloading {spec['url']}")
            subprocess.run(
                ["curl", "-L", "-o", str(out_path), spec["url"]], check=True,
            )
        elif _have("aria2c"):
            subprocess.run(
                ["aria2c", "-d", str(DEST), "-o", spec["filename"], spec["url"]],
                check=True,
            )
        else:
            print(
                f"  [{tissue}] please download manually: {spec['url']} -> "
                f"{out_path}"
            )
            continue
        sha = _sha256(out_path) if out_path.exists() else None
        manifest["tissues"][tissue] = {
            "url": spec["url"],
            "path": str(out_path.relative_to(ROOT)),
            "sha256": sha,
        }

    with (DEST / "manifest.json").open("w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nWrote {DEST / 'manifest.json'}")


if __name__ == "__main__":
    main()
