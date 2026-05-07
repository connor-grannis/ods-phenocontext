"""
Tests for M2: split manifest and data loader.

All fixtures use synthetic, fabricated text — no PHI.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from ods_phenocontext.data.loader import (
    _instance_id,
    build_manifest_from_parquet,
    load_instances,
)
from ods_phenocontext.data.split_manifest import (
    ManifestRow,
    build_split_manifest,
    load_manifest,
    write_manifest,
)
from ods_phenocontext.schema import NUM_LABELS, Instance

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

# 20 rows: mix of label combinations and a confirmed (all-false) majority
_SYNTHETIC_ROWS = [
    {
        "entity": "asthma",
        "label_negated": False,
        "label_family": False,
        "label_hypothetical": False,
        "text": "Patient has [ENT]asthma[/ENT].",
    },
    {
        "entity": "diabetes",
        "label_negated": True,
        "label_family": False,
        "label_hypothetical": False,
        "text": "No [ENT]diabetes[/ENT] noted.",
    },
    {
        "entity": "seizure",
        "label_negated": False,
        "label_family": True,
        "label_hypothetical": False,
        "text": "Mother has [ENT]seizure[/ENT] disorder.",
    },
    {
        "entity": "hypertension",
        "label_negated": False,
        "label_family": False,
        "label_hypothetical": True,
        "text": "Possible [ENT]hypertension[/ENT].",
    },
    {
        "entity": "anemia",
        "label_negated": True,
        "label_family": True,
        "label_hypothetical": False,
        "text": "Family denies [ENT]anemia[/ENT].",
    },
    {
        "entity": "obesity",
        "label_negated": False,
        "label_family": False,
        "label_hypothetical": False,
        "text": "Assessment: [ENT]obesity[/ENT].",
    },
    {
        "entity": "fracture",
        "label_negated": True,
        "label_family": False,
        "label_hypothetical": False,
        "text": "No [ENT]fracture[/ENT] seen.",
    },
    {
        "entity": "fever",
        "label_negated": False,
        "label_family": False,
        "label_hypothetical": False,
        "text": "Presents with [ENT]fever[/ENT].",
    },
    {
        "entity": "cough",
        "label_negated": False,
        "label_family": True,
        "label_hypothetical": False,
        "text": "Father had [ENT]cough[/ENT].",
    },
    {
        "entity": "rash",
        "label_negated": False,
        "label_family": False,
        "label_hypothetical": True,
        "text": "Screen for [ENT]rash[/ENT].",
    },
    {
        "entity": "edema",
        "label_negated": False,
        "label_family": False,
        "label_hypothetical": False,
        "text": "Bilateral [ENT]edema[/ENT] present.",
    },
    {
        "entity": "reflux",
        "label_negated": True,
        "label_family": False,
        "label_hypothetical": False,
        "text": "Denies [ENT]reflux[/ENT].",
    },
    {
        "entity": "infection",
        "label_negated": False,
        "label_family": False,
        "label_hypothetical": False,
        "text": "Active [ENT]infection[/ENT].",
    },
    {
        "entity": "pain",
        "label_negated": False,
        "label_family": False,
        "label_hypothetical": True,
        "text": "May develop [ENT]pain[/ENT].",
    },
    {
        "entity": "weakness",
        "label_negated": False,
        "label_family": True,
        "label_hypothetical": False,
        "text": "Sibling with [ENT]weakness[/ENT].",
    },
    {
        "entity": "headache",
        "label_negated": False,
        "label_family": False,
        "label_hypothetical": False,
        "text": "Reports [ENT]headache[/ENT].",
    },
    {
        "entity": "nausea",
        "label_negated": True,
        "label_family": False,
        "label_hypothetical": False,
        "text": "No [ENT]nausea[/ENT].",
    },
    {
        "entity": "fatigue",
        "label_negated": False,
        "label_family": False,
        "label_hypothetical": False,
        "text": "Significant [ENT]fatigue[/ENT].",
    },
    {
        "entity": "bleeding",
        "label_negated": False,
        "label_family": True,
        "label_hypothetical": False,
        "text": "Family hx [ENT]bleeding[/ENT].",
    },
    {
        "entity": "tremor",
        "label_negated": False,
        "label_family": False,
        "label_hypothetical": False,
        "text": "Essential [ENT]tremor[/ENT].",
    },
]


@pytest.fixture
def synthetic_parquet(tmp_path: Path) -> Path:
    path = tmp_path / "synthetic.parquet"
    pd.DataFrame(_SYNTHETIC_ROWS).to_parquet(path, index=False)
    return path


@pytest.fixture
def full_manifest(synthetic_parquet: Path, tmp_path: Path) -> Path:
    """Manifest covering all 20 rows, all assigned to train."""
    rows = [
        ManifestRow(
            instance_id=_instance_id(i),
            note_id=_instance_id(i),
            split="train",
            date_assigned="2026-05-07",
        )
        for i in range(len(_SYNTHETIC_ROWS))
    ]
    path = tmp_path / "manifest.jsonl"
    write_manifest(rows, path)
    return path


# ---------------------------------------------------------------------------
# ENT tags preserved in context_window
# ---------------------------------------------------------------------------


def test_ent_tags_preserved_in_context_window(synthetic_parquet: Path, full_manifest: Path):
    instances = list(load_instances(synthetic_parquet, full_manifest))
    tagged = [i for i in instances if "[ENT]" in i.context_window]
    assert len(tagged) == len(_SYNTHETIC_ROWS)


# ---------------------------------------------------------------------------
# ManifestRow / write_manifest / load_manifest
# ---------------------------------------------------------------------------


class TestManifest:
    def test_round_trip(self, tmp_path: Path):
        rows = [
            ManifestRow("i-001", "n-001", "train", "2026-05-07"),
            ManifestRow("i-002", "n-002", "val", "2026-05-07", exclusion_reason="duplicate"),
        ]
        path = tmp_path / "m.jsonl"
        write_manifest(rows, path)
        loaded = load_manifest(path)
        assert len(loaded) == 2
        assert loaded[0].instance_id == "i-001"
        assert loaded[1].exclusion_reason == "duplicate"

    def test_note_cross_split_raises(self, tmp_path: Path):
        # same note_id (non-excluded) in both train and val
        rows = [
            ManifestRow("i-001", "n-shared", "train", "2026-05-07"),
            ManifestRow("i-002", "n-shared", "val", "2026-05-07"),
        ]
        path = tmp_path / "bad.jsonl"
        write_manifest(rows, path)
        with pytest.raises(ValueError, match="multiple splits"):
            load_manifest(path)

    def test_excluded_note_allowed_in_multiple_splits(self, tmp_path: Path):
        # excluded rows are exempt from the cross-split constraint
        rows = [
            ManifestRow("i-001", "n-shared", "train", "2026-05-07"),
            ManifestRow("i-002", "n-shared", "val", "2026-05-07", exclusion_reason="excluded"),
        ]
        path = tmp_path / "ok.jsonl"
        write_manifest(rows, path)
        loaded = load_manifest(path)
        assert len(loaded) == 2

    def test_build_split_manifest(self):
        rows = build_split_manifest(
            ["i-001", "i-002"],
            ["n-001", "n-002"],
            ["train", "val"],
            date_assigned="2026-05-07",
        )
        assert rows[0].split == "train"
        assert rows[1].split == "val"
        assert rows[0].date_assigned == "2026-05-07"


# ---------------------------------------------------------------------------
# load_instances
# ---------------------------------------------------------------------------


class TestLoadInstances:
    def test_correct_count(self, synthetic_parquet: Path, full_manifest: Path):
        instances = list(load_instances(synthetic_parquet, full_manifest))
        assert len(instances) == len(_SYNTHETIC_ROWS)

    def test_exclusions_honored(self, synthetic_parquet: Path, tmp_path: Path):
        rows = [
            ManifestRow(
                _instance_id(i),
                _instance_id(i),
                "train",
                "2026-05-07",
                exclusion_reason="duplicate" if i == 0 else None,
            )
            for i in range(len(_SYNTHETIC_ROWS))
        ]
        manifest_path = tmp_path / "manifest_excl.jsonl"
        write_manifest(rows, manifest_path)
        instances = list(load_instances(synthetic_parquet, manifest_path))
        assert len(instances) == len(_SYNTHETIC_ROWS) - 1

    def test_gold_labels_length(self, synthetic_parquet: Path, full_manifest: Path):
        for inst in load_instances(synthetic_parquet, full_manifest):
            assert inst.gold_labels is not None
            assert len(inst.gold_labels) == NUM_LABELS

    def test_confirmed_label_correct(self, synthetic_parquet: Path, full_manifest: Path):
        # Row 0: all False → confirmed=1, rest=0
        instances = list(load_instances(synthetic_parquet, full_manifest))
        inst_0 = next(i for i in instances if i.entity_text == "asthma")
        assert inst_0.gold_labels == [1, 0, 0, 0]

    def test_negated_label_correct(self, synthetic_parquet: Path, full_manifest: Path):
        instances = list(load_instances(synthetic_parquet, full_manifest))
        inst = next(i for i in instances if i.entity_text == "diabetes")
        assert inst.gold_labels == [0, 1, 0, 0]

    def test_family_label_correct(self, synthetic_parquet: Path, full_manifest: Path):
        instances = list(load_instances(synthetic_parquet, full_manifest))
        inst = next(i for i in instances if i.entity_text == "seizure")
        assert inst.gold_labels == [0, 0, 1, 0]

    def test_hypothetical_label_correct(self, synthetic_parquet: Path, full_manifest: Path):
        instances = list(load_instances(synthetic_parquet, full_manifest))
        inst = next(i for i in instances if i.entity_text == "hypertension")
        assert inst.gold_labels == [0, 0, 0, 1]

    def test_multi_label_correct(self, synthetic_parquet: Path, full_manifest: Path):
        # Row 4: negated=True, family=True → [0,1,1,0], confirmed=False
        instances = list(load_instances(synthetic_parquet, full_manifest))
        inst = next(i for i in instances if i.entity_text == "anemia")
        assert inst.gold_labels == [0, 1, 1, 0]

    def test_ent_tags_preserved(self, synthetic_parquet: Path, full_manifest: Path):
        for inst in load_instances(synthetic_parquet, full_manifest):
            assert "[ENT]" in inst.context_window
            assert "[/ENT]" in inst.context_window

    def test_split_assigned_from_manifest(self, synthetic_parquet: Path, tmp_path: Path):
        rows = [
            ManifestRow(_instance_id(i), _instance_id(i), "val" if i < 5 else "train", "2026-05-07")
            for i in range(len(_SYNTHETIC_ROWS))
        ]
        manifest_path = tmp_path / "manifest_split.jsonl"
        write_manifest(rows, manifest_path)
        instances = list(load_instances(synthetic_parquet, manifest_path))
        val_count = sum(1 for i in instances if i.split == "val")
        train_count = sum(1 for i in instances if i.split == "train")
        assert val_count == 5
        assert train_count == 15

    def test_instance_id_format(self, synthetic_parquet: Path, full_manifest: Path):
        instances = list(load_instances(synthetic_parquet, full_manifest))
        assert instances[0].instance_id == "inst-0000000"
        assert instances[-1].instance_id == f"inst-{len(_SYNTHETIC_ROWS) - 1:07d}"


# ---------------------------------------------------------------------------
# build_manifest_from_parquet
# ---------------------------------------------------------------------------


class TestBuildManifestFromParquet:
    def test_total_count(self, synthetic_parquet: Path):
        rows = build_manifest_from_parquet(synthetic_parquet, val_fraction=0.10)
        assert len(rows) == len(_SYNTHETIC_ROWS)

    def test_val_fraction_approximate(self, synthetic_parquet: Path):
        rows = build_manifest_from_parquet(synthetic_parquet, val_fraction=0.10)
        val_n = sum(1 for r in rows if r.split == "val")
        expected = round(len(_SYNTHETIC_ROWS) * 0.10)
        assert val_n == expected

    def test_reproducible_with_same_seed(self, synthetic_parquet: Path):
        rows_a = build_manifest_from_parquet(synthetic_parquet, random_seed=0)
        rows_b = build_manifest_from_parquet(synthetic_parquet, random_seed=0)
        assert [r.split for r in rows_a] == [r.split for r in rows_b]

    def test_different_seeds_differ(self, synthetic_parquet: Path):
        rows_a = build_manifest_from_parquet(synthetic_parquet, random_seed=1)
        rows_b = build_manifest_from_parquet(synthetic_parquet, random_seed=2)
        assert [r.split for r in rows_a] != [r.split for r in rows_b]

    def test_note_id_equals_instance_id(self, synthetic_parquet: Path):
        rows = build_manifest_from_parquet(synthetic_parquet)
        for r in rows:
            assert r.note_id == r.instance_id

    def test_no_cross_split_violations(self, synthetic_parquet: Path, tmp_path: Path):
        rows = build_manifest_from_parquet(synthetic_parquet)
        path = tmp_path / "generated.jsonl"
        write_manifest(rows, path)
        load_manifest(path)  # raises if cross-split violation exists

    # --- max_confirmed cap ---

    def test_max_confirmed_caps_included_count(self, synthetic_parquet: Path):
        # Synthetic data has 8 confirmed rows; cap at 5 → 3 excluded
        rows = build_manifest_from_parquet(synthetic_parquet, max_confirmed=5)
        excluded = [r for r in rows if r.exclusion_reason == "confirmed_cap"]
        assert len(excluded) == 8 - 5

    def test_max_confirmed_exclusion_reason(self, synthetic_parquet: Path):
        rows = build_manifest_from_parquet(synthetic_parquet, max_confirmed=5)
        for r in rows:
            if r.exclusion_reason:
                assert r.exclusion_reason == "confirmed_cap"

    def test_max_confirmed_excluded_rows_not_in_val(self, synthetic_parquet: Path):
        rows = build_manifest_from_parquet(synthetic_parquet, max_confirmed=5)
        for r in rows:
            if r.exclusion_reason == "confirmed_cap":
                assert r.split == "train"

    def test_max_confirmed_none_keeps_all(self, synthetic_parquet: Path):
        rows_capped = build_manifest_from_parquet(synthetic_parquet, max_confirmed=None)
        rows_uncapped = build_manifest_from_parquet(synthetic_parquet)
        assert rows_capped == rows_uncapped

    def test_max_confirmed_above_total_excludes_nothing(self, synthetic_parquet: Path):
        rows = build_manifest_from_parquet(synthetic_parquet, max_confirmed=9999)
        assert all(not r.exclusion_reason for r in rows)

    def test_val_fraction_applied_to_included_pool(self, synthetic_parquet: Path):
        # With cap=5: 5 confirmed + 12 non-confirmed = 17 included
        # 10% of 17 = 2 val rows (rounded)
        rows = build_manifest_from_parquet(synthetic_parquet, max_confirmed=5, val_fraction=0.10)
        val_n = sum(1 for r in rows if r.split == "val")
        assert val_n == round(17 * 0.10)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


class TestCLI:
    def test_build_manifest_cmd(self, synthetic_parquet: Path, tmp_path: Path):
        from click.testing import CliRunner

        from ods_phenocontext.data.__main__ import cli

        out = tmp_path / "manifest.jsonl"
        result = CliRunner().invoke(
            cli, ["build-manifest", "--input", str(synthetic_parquet), "--out", str(out)]
        )
        assert result.exit_code == 0, result.output
        assert out.exists()
        rows = load_manifest(out)
        assert len(rows) == len(_SYNTHETIC_ROWS)
        assert "train=" in result.output
        assert "val=" in result.output

    def test_build_manifest_cmd_with_max_confirmed(self, synthetic_parquet: Path, tmp_path: Path):
        from click.testing import CliRunner

        from ods_phenocontext.data.__main__ import cli

        out = tmp_path / "manifest_capped.jsonl"
        result = CliRunner().invoke(
            cli,
            [
                "build-manifest",
                "--input",
                str(synthetic_parquet),
                "--out",
                str(out),
                "--max-confirmed",
                "5",
            ],
        )
        assert result.exit_code == 0, result.output
        assert "excluded=" in result.output

    def test_process_cmd(self, synthetic_parquet: Path, tmp_path: Path):
        from click.testing import CliRunner

        from ods_phenocontext.data.__main__ import cli

        manifest_path = tmp_path / "manifest.jsonl"
        CliRunner().invoke(
            cli, ["build-manifest", "--input", str(synthetic_parquet), "--out", str(manifest_path)]
        )
        out = tmp_path / "instances.jsonl"
        result = CliRunner().invoke(
            cli,
            [
                "process",
                "--input",
                str(synthetic_parquet),
                "--manifest",
                str(manifest_path),
                "--out",
                str(out),
            ],
        )
        assert result.exit_code == 0, result.output
        assert out.exists()
        lines = out.read_text().strip().splitlines()
        assert len(lines) == len(_SYNTHETIC_ROWS)
        # Verify round-trip: each line deserializes to a valid Instance
        for line in lines:
            inst = Instance.from_dict(json.loads(line))
            assert inst.gold_labels is not None
