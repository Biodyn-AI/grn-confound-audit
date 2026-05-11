"""
Command-line interface for grn_confound_audit.

End-to-end usage:
    grn-confound-audit run \
        --edges edges.csv \
        --gene-coords coords.tsv \
        --metadata cells.csv \
        --counts counts.parquet \
        --scores-balanced balanced.csv \
        --output report/

Minimal usage (Class 2 + Class 3 only):
    grn-confound-audit run --edges edges.csv --gene-coords coords.tsv \
        --output report/
"""

import argparse
import os
import sys

import pandas as pd

from .pipeline import ConfoundAuditPipeline
from . import __version__


def _read_table(path: str) -> pd.DataFrame:
    """Read CSV/TSV/Parquet by extension."""
    if path.endswith((".tsv", ".txt")):
        return pd.read_csv(path, sep="\t")
    if path.endswith(".parquet"):
        return pd.read_parquet(path)
    return pd.read_csv(path)


def main():
    parser = argparse.ArgumentParser(
        prog="grn-confound-audit",
        description="Three-class confound audit for gene regulatory networks.",
    )
    parser.add_argument(
        "--version", action="version", version=f"grn-confound-audit {__version__}",
    )
    subparsers = parser.add_subparsers(dest="command")

    # --- run subcommand ---
    rp = subparsers.add_parser(
        "run",
        help="Run the three-class confound audit on a scored edge list.",
    )
    # Inputs
    rp.add_argument(
        "--edges", required=True,
        help="CSV/TSV with columns: tf, target, score.",
    )
    rp.add_argument(
        "--gene-coords",
        help="TSV/CSV with columns: gene, chr, tss. Required for Class 2.",
    )
    rp.add_argument(
        "--metadata",
        help="CSV/TSV with cell-level metadata: cell_id plus any subset of "
             "{donor, batch, method}. Required for Class 1 leakage tests.",
    )
    rp.add_argument(
        "--counts",
        help="Cell x gene expression matrix (CSV/TSV/Parquet) with cell_id "
             "as the first column. If provided together with --metadata, the "
             "CLI computes per-cell edge-product features on the fly using "
             "the top --n-top-features edges by variance.",
    )
    rp.add_argument(
        "--edge-features",
        help="Optional pre-computed cell x edge feature matrix "
             "(CSV/TSV/Parquet). Edge IDs in column names must follow "
             "the convention 'TF->TARGET'. Overrides on-the-fly "
             "feature construction from --counts.",
    )
    rp.add_argument(
        "--scores-balanced",
        help="CSV/TSV with balanced edge scores (columns: edge_id, score). "
             "Required for Class 1 ASI computation.",
    )
    # Outputs
    rp.add_argument(
        "--output", "-o", default="audit_output",
        help="Output directory (default: audit_output/).",
    )
    # Class 1 knobs
    rp.add_argument(
        "--asi-threshold", type=float, default=0.5,
        help="ASI threshold for technical blacklisting (default: 0.5).",
    )
    rp.add_argument(
        "--n-top-features", type=int, default=200,
        help="Top-variance edges used for leakage classifiers (default: 200).",
    )
    # Class 2 knobs
    rp.add_argument(
        "--n-permutations", type=int, default=1000,
        help="Permutations for the proximity null (default: 1000).",
    )
    rp.add_argument(
        "--proximity-null-families",
        default="source,same_chr,degree",
        help="Comma-separated proximity null families to evaluate "
             "(default: source,same_chr,degree).",
    )
    # Class 3 knobs
    rp.add_argument(
        "--n-null-replicates", type=int, default=2000,
        help="Degree-preserving null replicates for topological audit "
             "(default: 2000; >=2000 is recommended for edge-level FDR).",
    )
    rp.add_argument(
        "--null-tail-method",
        choices=["empirical", "gpd"], default="gpd",
        help="Per-edge p-value floor handling: 'empirical' uses (b+1)/(B+1); "
             "'gpd' fits a Generalised Pareto tail above a chosen threshold "
             "to extend p-values below the empirical floor (default: gpd).",
    )
    rp.add_argument(
        "--fdr-q", type=float, default=0.10,
        help="Benjamini-Hochberg q-value cut for edge-level significance "
             "(default: 0.10).",
    )
    # Misc
    rp.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42).",
    )

    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        sys.exit(1)

    if args.command == "run":
        _run_audit(args)


def _build_edge_features_from_counts(
    counts: pd.DataFrame,
    edges: pd.DataFrame,
    n_top_features: int,
) -> pd.DataFrame:
    """Construct a cell x edge feature matrix from a counts table.

    For each edge (TF -> target), the feature value in a cell is the
    log1p-normalised product of TF and target expression in that cell.
    Only the top-`n_top_features` edges by variance are retained.
    """
    # Cell index is the first column unless 'cell_id' explicitly present
    if "cell_id" in counts.columns:
        counts = counts.set_index("cell_id")
    else:
        counts = counts.set_index(counts.columns[0])

    # log1p normalisation
    import numpy as np

    cnt = np.log1p(counts.astype(float))

    tf_genes = edges["tf"].unique().tolist()
    tg_genes = edges["target"].unique().tolist()
    have = set(cnt.columns)

    missing_tf = [g for g in tf_genes if g not in have]
    missing_tg = [g for g in tg_genes if g not in have]
    if missing_tf or missing_tg:
        print(
            f"  WARNING: {len(missing_tf)} TFs and {len(missing_tg)} targets "
            f"in edge list are missing from --counts; affected edges are dropped."
        )

    feats = {}
    for _, row in edges.iterrows():
        tf, tg = row["tf"], row["target"]
        if tf in have and tg in have:
            feats[f"{tf}->{tg}"] = (cnt[tf] * cnt[tg]).values

    feat_df = pd.DataFrame(feats, index=cnt.index)

    if feat_df.shape[1] > n_top_features:
        top = feat_df.var(axis=0).nlargest(n_top_features).index
        feat_df = feat_df[top]

    return feat_df


def _run_audit(args):
    """Execute the audit pipeline from CLI arguments."""
    print(f"grn-confound-audit v{__version__}")
    print(f"Loading edges from {args.edges} ...")

    edges = _read_table(args.edges)
    required = {"tf", "target", "score"}
    if not required.issubset(edges.columns):
        print(f"ERROR: edges file must have columns {required}; got {set(edges.columns)}")
        sys.exit(1)
    print(
        f"  {len(edges)} edges  |  {edges['tf'].nunique()} TFs  |  "
        f"{edges['target'].nunique()} targets"
    )

    # Class 2 inputs
    gene_coords = None
    if args.gene_coords:
        gene_coords = _read_table(args.gene_coords)
        if not {"gene", "chr", "tss"}.issubset(gene_coords.columns):
            print(
                f"WARNING: --gene-coords needs columns gene/chr/tss; got "
                f"{set(gene_coords.columns)}. Skipping proximity audit."
            )
            gene_coords = None
        else:
            print(f"  {len(gene_coords)} gene coordinates loaded")

    # Class 1 inputs
    metadata = None
    covariates = None
    if args.metadata:
        metadata = _read_table(args.metadata)
        if "cell_id" not in metadata.columns:
            print(
                "WARNING: --metadata needs a 'cell_id' column; "
                "leakage classification will be skipped."
            )
            metadata = None
        else:
            metadata = metadata.set_index("cell_id")
            covariates = {
                c: metadata[c] for c in ("donor", "batch", "method")
                if c in metadata.columns
            }
            print(
                f"  metadata: {len(metadata)} cells, "
                f"covariates available: {list(covariates.keys())}"
            )

    edge_features = None
    if args.edge_features:
        edge_features = _read_table(args.edge_features)
        if "cell_id" in edge_features.columns:
            edge_features = edge_features.set_index("cell_id")
        print(
            f"  pre-computed edge features: {edge_features.shape[0]} cells x "
            f"{edge_features.shape[1]} edges"
        )
    elif args.counts and metadata is not None:
        print(
            f"  Building per-cell edge features from {args.counts} "
            f"(top {args.n_top_features} edges by variance) ..."
        )
        counts = _read_table(args.counts)
        edge_features = _build_edge_features_from_counts(
            counts, edges, args.n_top_features,
        )
        print(
            f"  edge_features: {edge_features.shape[0]} cells x "
            f"{edge_features.shape[1]} edges"
        )

    if edge_features is not None and metadata is not None:
        common = edge_features.index.intersection(metadata.index)
        if len(common) < len(edge_features):
            print(
                f"  Aligning to {len(common)} cells present in both "
                f"edge features and metadata."
            )
        edge_features = edge_features.loc[common]
        covariates = {k: v.loc[common] for k, v in (covariates or {}).items()}

    # Class 1 balanced scores (optional)
    scores_balanced = None
    if args.scores_balanced:
        bal = _read_table(args.scores_balanced)
        if {"edge_id", "score"}.issubset(bal.columns):
            scores_balanced = bal.set_index("edge_id")["score"]
            print(f"  {len(scores_balanced)} balanced edge scores loaded")
        else:
            print(
                "WARNING: --scores-balanced needs columns edge_id, score. "
                "ASI computation skipped."
            )

    # Pipeline
    null_families = [s.strip() for s in args.proximity_null_families.split(",") if s.strip()]
    pipeline = ConfoundAuditPipeline(
        asi_threshold=args.asi_threshold,
        n_permutations=args.n_permutations,
        n_null_replicates=args.n_null_replicates,
        null_tail_method=args.null_tail_method,
        fdr_q=args.fdr_q,
        proximity_null_families=null_families,
        n_top_features=args.n_top_features,
        random_state=args.seed,
    )

    print("\nRunning three-class confound audit ...")
    pipeline.run(
        edges=edges,
        gene_coords=gene_coords,
        scores_balanced=scores_balanced,
        edge_features=edge_features,
        covariates=covariates,
        output_dir=args.output,
    )

    print(f"\nResults written to {args.output}/")
    for fn in (
        "audit_results.json",
        "edge_quality_indices.csv",
        "cross_class_synthesis.csv",
        "audit_summary.txt",
    ):
        path = os.path.join(args.output, fn)
        if os.path.exists(path):
            print(f"  {fn}")

    summary_path = os.path.join(args.output, "audit_summary.txt")
    if os.path.exists(summary_path):
        with open(summary_path) as f:
            print("\n" + f.read())


if __name__ == "__main__":
    main()
