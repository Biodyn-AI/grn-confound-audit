"""
Command-line interface for grn_confound_audit.

Usage:
    grn-confound-audit run --edges edges.csv --output report/
    grn-confound-audit run --edges edges.csv --gene-coords coords.tsv --output report/
"""

import argparse
import sys
import os

import pandas as pd

from .pipeline import ConfoundAuditPipeline


def main():
    parser = argparse.ArgumentParser(
        prog="grn-confound-audit",
        description="Three-class confound audit for gene regulatory networks.",
    )
    subparsers = parser.add_subparsers(dest="command")

    # --- run subcommand ---
    run_parser = subparsers.add_parser(
        "run",
        help="Run the three-class confound audit on a scored edge list.",
    )
    run_parser.add_argument(
        "--edges", required=True,
        help="CSV file with columns: tf, target, score.",
    )
    run_parser.add_argument(
        "--gene-coords",
        help="TSV/CSV file with columns: gene, chr, tss. "
             "Required for Class 2 (proximity audit).",
    )
    run_parser.add_argument(
        "--metadata",
        help="CSV file with cell-level metadata. Columns should include "
             "cell_id plus technical covariates (donor, batch, method). "
             "Required for Class 1 leakage classification.",
    )
    run_parser.add_argument(
        "--scores-balanced",
        help="CSV file with balanced edge scores (columns: edge_id, score). "
             "Required for Class 1 ASI computation.",
    )
    run_parser.add_argument(
        "--output", "-o", default="audit_output",
        help="Output directory for results (default: audit_output/).",
    )
    run_parser.add_argument(
        "--asi-threshold", type=float, default=0.5,
        help="ASI threshold for technical blacklisting (default: 0.5).",
    )
    run_parser.add_argument(
        "--n-permutations", type=int, default=1000,
        help="Permutations for proximity null model (default: 1000).",
    )
    run_parser.add_argument(
        "--n-null-replicates", type=int, default=48,
        help="Null replicates for topological audit (default: 48).",
    )
    run_parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42).",
    )

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    if args.command == "run":
        _run_audit(args)


def _run_audit(args):
    """Execute the audit pipeline from CLI arguments."""
    print(f"grn_confound_audit v0.1.0")
    print(f"Loading edges from {args.edges}...")

    # Load edges
    edges = pd.read_csv(args.edges)
    required_cols = {"tf", "target", "score"}
    if not required_cols.issubset(edges.columns):
        print(f"ERROR: edges file must have columns: {required_cols}")
        print(f"  Found: {set(edges.columns)}")
        sys.exit(1)
    print(f"  {len(edges)} edges, {edges['tf'].nunique()} TFs, {edges['target'].nunique()} targets")

    # Load gene coordinates (optional, for Class 2)
    gene_coords = None
    if args.gene_coords:
        sep = "\t" if args.gene_coords.endswith(".tsv") else ","
        gene_coords = pd.read_csv(args.gene_coords, sep=sep)
        required_coord_cols = {"gene", "chr", "tss"}
        if not required_coord_cols.issubset(gene_coords.columns):
            print(f"WARNING: gene-coords file should have columns: {required_coord_cols}")
            print(f"  Found: {set(gene_coords.columns)}. Skipping proximity audit.")
            gene_coords = None
        else:
            print(f"  {len(gene_coords)} gene coordinates loaded")

    # Load balanced scores (optional, for Class 1 ASI)
    scores_balanced = None
    if args.scores_balanced:
        bal = pd.read_csv(args.scores_balanced)
        if "edge_id" in bal.columns and "score" in bal.columns:
            scores_balanced = bal.set_index("edge_id")["score"]
            print(f"  {len(scores_balanced)} balanced scores loaded")
        else:
            print("WARNING: balanced scores file should have columns: edge_id, score. Skipping ASI.")

    # Run pipeline
    pipeline = ConfoundAuditPipeline(
        asi_threshold=args.asi_threshold,
        n_permutations=args.n_permutations,
        n_null_replicates=args.n_null_replicates,
        random_state=args.seed,
    )

    print(f"\nRunning three-class confound audit...")
    report = pipeline.run(
        edges=edges,
        gene_coords=gene_coords,
        scores_balanced=scores_balanced,
        output_dir=args.output,
    )

    print(f"\nResults written to {args.output}/")
    print(f"  audit_results.json       - Machine-readable full results")
    print(f"  edge_quality_indices.csv - Per-edge quality profiles")
    print(f"  audit_summary.txt        - Human-readable summary")

    # Print quick summary to stdout
    summary_path = os.path.join(args.output, "audit_summary.txt")
    if os.path.exists(summary_path):
        with open(summary_path) as f:
            print(f"\n{f.read()}")


if __name__ == "__main__":
    main()
