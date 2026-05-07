"""
Data loader: reads the training parquet and a split manifest, yielding
validated Instance objects.

Parquet schema expected:
  entity            str    — phenotype mention text
  text              str    — context window (may contain [ENT]...[/ENT] tags)
  label_negated     bool
  label_family      bool
  label_hypothetical bool

Label mapping to Instance.gold_labels (multi-hot, 4 positions):
  [confirmed, negated, associated_with_someone_else, other_non_patient]
  confirmed = not (negated OR family OR hypothetical)
  negated   = label_negated
  associated_with_someone_else = label_family
  other_non_patient = label_hypothetical

Instance IDs are generated as "inst-{row_index:07d}".
Note IDs match instance IDs (no note-level grouping in source data —
see docs/decision_log.md).
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pandas as pd

from ods_phenocontext.data.split_manifest import ManifestRow, load_manifest
from ods_phenocontext.schema import Instance


def _row_to_instance(row: pd.Series, manifest_row: ManifestRow) -> Instance:
    """Convert one parquet row + manifest row into a validated Instance."""
    negated = bool(row["label_negated"])
    family = bool(row["label_family"])
    hypothetical = bool(row["label_hypothetical"])
    confirmed = not (negated or family or hypothetical)

    gold_labels = [int(confirmed), int(negated), int(family), int(hypothetical)]

    # Keep [ENT]/[/ENT] tags — they mark entity span for both the rule engine
    # (pre/post splitting) and BioBERT (entity-span pooling).
    return Instance.from_raw(
        instance_id=manifest_row.instance_id,
        note_id=manifest_row.note_id,
        entity_text=str(row["entity"]),
        context_window=str(row["text"]),
        split=manifest_row.split,
        gold_labels=gold_labels,
    )


def load_instances(
    parquet_path: Path,
    manifest_path: Path,
) -> Iterator[Instance]:
    """Yield Instance objects for all non-excluded rows in the manifest.

    Rows whose manifest entry has a non-empty exclusion_reason are silently
    dropped.  Row count by split is logged to stdout (no raw text logged).

    Args:
        parquet_path:   Path to the source parquet file.
        manifest_path:  Path to the split manifest JSONL.
    """
    manifest_rows = load_manifest(manifest_path)
    manifest_index: dict[str, ManifestRow] = {r.instance_id: r for r in manifest_rows}

    df = pd.read_parquet(parquet_path)

    counts: dict[str, int] = {}
    excluded = 0

    for idx, row in df.iterrows():
        iid = _instance_id(int(idx))  # type: ignore[arg-type]
        manifest_row = manifest_index.get(iid)
        if manifest_row is None:
            continue
        if manifest_row.exclusion_reason:
            excluded += 1
            continue

        instance = _row_to_instance(row, manifest_row)
        counts[instance.split] = counts.get(instance.split, 0) + 1
        yield instance

    total = sum(counts.values())
    split_summary = ", ".join(f"{s}={n}" for s, n in sorted(counts.items()))
    print(f"Loaded {total} instances ({split_summary}); excluded {excluded}")


def _instance_id(row_index: int) -> str:
    return f"inst-{row_index:07d}"


def build_manifest_from_parquet(
    parquet_path: Path,
    val_fraction: float = 0.10,
    random_seed: int = 42,
    date_assigned: str | None = None,
    max_confirmed: int | None = None,
) -> list[ManifestRow]:
    """Generate a train/val split manifest from a parquet file.

    Confirmed rows (all three label columns False) are optionally capped at
    max_confirmed before the val split is computed — excess confirmed rows
    receive exclusion_reason="confirmed_cap".  All non-confirmed rows are
    always included.  The 10% val split is then drawn from the retained pool.

    Note IDs are set equal to instance IDs — the source data has no
    note-level grouping information (see docs/decision_log.md).
    """
    from datetime import date

    import numpy as np

    df = pd.read_parquet(parquet_path)
    rng = np.random.default_rng(random_seed)
    today = date_assigned or date.today().isoformat()

    is_confirmed = ~(df["label_negated"] | df["label_family"] | df["label_hypothetical"])

    confirmed_idx = is_confirmed[is_confirmed].index.tolist()
    other_idx = is_confirmed[~is_confirmed].index.tolist()

    excluded_idx: set[int] = set()
    if max_confirmed is not None and len(confirmed_idx) > max_confirmed:
        confirmed_arr = np.array(confirmed_idx)
        rng.shuffle(confirmed_arr)
        excluded_idx = set(confirmed_arr[max_confirmed:].tolist())
        confirmed_idx = confirmed_arr[:max_confirmed].tolist()

    included_idx = np.array(sorted(confirmed_idx + other_idx))
    rng.shuffle(included_idx)
    val_n = max(1, int(round(len(included_idx) * val_fraction)))
    val_idx = set(included_idx[:val_n].tolist())

    rows = []
    for i in range(len(df)):
        iid = _instance_id(i)
        if i in excluded_idx:
            rows.append(ManifestRow(iid, iid, "train", today, exclusion_reason="confirmed_cap"))
        else:
            split = "val" if i in val_idx else "train"
            rows.append(ManifestRow(iid, iid, split, today))
    return rows
