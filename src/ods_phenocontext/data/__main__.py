"""
CLI for data preparation.

Commands:
  build-manifest   Generate train/val split manifest from a parquet file.
  process          Load instances from parquet + manifest and write JSONL.

Examples:
  uv run python -m ods_phenocontext.data build-manifest \\
      --input data/all_training_samples.parquet \\
      --out data/gold/split_manifest.jsonl

  uv run python -m ods_phenocontext.data process \\
      --input data/all_training_samples.parquet \\
      --manifest data/gold/split_manifest.jsonl \\
      --out data/processed/instances.jsonl
"""

from __future__ import annotations

import json
from pathlib import Path

import click

from ods_phenocontext.data.loader import build_manifest_from_parquet, load_instances
from ods_phenocontext.data.split_manifest import write_manifest


@click.group()
def cli() -> None:
    """PhenoContext data preparation utilities."""


@cli.command("build-manifest")
@click.option(
    "--input",
    "input_path",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="Source parquet file.",
)
@click.option(
    "--out",
    required=True,
    type=click.Path(path_type=Path),
    help="Output manifest JSONL path.",
)
@click.option(
    "--val-fraction",
    default=0.10,
    show_default=True,
    help="Fraction of rows assigned to validation split.",
)
@click.option(
    "--seed",
    default=42,
    show_default=True,
    help="Random seed for reproducibility.",
)
@click.option(
    "--max-confirmed",
    default=None,
    type=int,
    show_default=True,
    help="Cap on confirmed (all-negative-label) instances. Excess are excluded.",
)
def build_manifest_cmd(
    input_path: Path, out: Path, val_fraction: float, seed: int, max_confirmed: int | None
) -> None:
    """Generate a train/val split manifest from a parquet file."""
    rows = build_manifest_from_parquet(
        input_path, val_fraction=val_fraction, random_seed=seed, max_confirmed=max_confirmed
    )
    write_manifest(rows, out)

    train_n = sum(1 for r in rows if r.split == "train" and not r.exclusion_reason)
    val_n = sum(1 for r in rows if r.split == "val")
    excluded_n = sum(1 for r in rows if r.exclusion_reason)
    msg = f"Wrote {len(rows)} rows to {out}  (train={train_n}, val={val_n}"
    if excluded_n:
        msg += f", excluded={excluded_n}"
    click.echo(msg + ")")


@cli.command("process")
@click.option(
    "--input",
    "input_path",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="Source parquet file.",
)
@click.option(
    "--manifest",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="Split manifest JSONL.",
)
@click.option(
    "--out",
    required=True,
    type=click.Path(path_type=Path),
    help="Output instances JSONL path.",
)
def process_cmd(input_path: Path, manifest: Path, out: Path) -> None:
    """Load parquet + manifest and write validated Instance JSONL."""
    out.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with out.open("w") as fh:
        for instance in load_instances(input_path, manifest):
            fh.write(json.dumps(instance.to_dict()) + "\n")
            count += 1
    click.echo(f"Wrote {count} instances to {out}")


if __name__ == "__main__":
    cli()
