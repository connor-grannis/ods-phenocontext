"""
Split manifest: maps instance_id → split assignment.

Schema (one JSON object per line):
  {"instance_id": str, "note_id": str, "split": str,
   "date_assigned": str (ISO-8601), "exclusion_reason": str | null}

Constraint: a note_id must not appear in more than one split.  This is
enforced at load time, not write time, because manifest rows are usually
written in bulk from the same source (all instances from the same parquet).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path


@dataclass
class ManifestRow:
    instance_id: str
    note_id: str
    split: str
    date_assigned: str  # ISO-8601 date string
    exclusion_reason: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> ManifestRow:
        return cls(
            instance_id=d["instance_id"],
            note_id=d["note_id"],
            split=d["split"],
            date_assigned=d["date_assigned"],
            exclusion_reason=d.get("exclusion_reason"),
        )


def write_manifest(rows: list[ManifestRow], path: Path) -> None:
    """Write rows to a JSONL manifest file, overwriting any existing file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        for row in rows:
            fh.write(json.dumps(row.to_dict()) + "\n")


def load_manifest(path: Path) -> list[ManifestRow]:
    """Load and validate a split manifest.

    Raises ValueError if any note_id appears in more than one split.
    """
    rows: list[ManifestRow] = []
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(ManifestRow.from_dict(json.loads(line)))

    _check_note_cross_split(rows)
    return rows


def _check_note_cross_split(rows: list[ManifestRow]) -> None:
    """Raise if any note_id is assigned to more than one split."""
    note_splits: dict[str, set[str]] = {}
    for row in rows:
        if row.exclusion_reason:
            continue
        note_splits.setdefault(row.note_id, set()).add(row.split)

    violations = {nid: splits for nid, splits in note_splits.items() if len(splits) > 1}
    if violations:
        examples = list(violations.items())[:3]
        raise ValueError(
            f"note_id appears in multiple splits (first {len(examples)} of "
            f"{len(violations)} violations): {examples}"
        )


def build_split_manifest(
    instance_ids: list[str],
    note_ids: list[str],
    splits: list[str],
    date_assigned: str | None = None,
) -> list[ManifestRow]:
    """Construct ManifestRow objects from parallel lists."""
    today = date_assigned or date.today().isoformat()
    return [
        ManifestRow(
            instance_id=iid,
            note_id=nid,
            split=split,
            date_assigned=today,
        )
        for iid, nid, split in zip(instance_ids, note_ids, splits, strict=True)
    ]
