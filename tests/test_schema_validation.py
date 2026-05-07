"""
Tests for M1: Instance validators, from_raw factory, and to_dict/from_dict
round-trips across all three schema dataclasses.

Uses only synthetic strings — no PHI.
"""

from __future__ import annotations

import json

import pytest

from ods_phenocontext.schema import (
    NUM_LABELS,
    Instance,
    SyntheticAudit,
    TrainingManifest,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _inst(**kwargs) -> Instance:
    defaults = dict(
        instance_id="i-001",
        note_id="n-001",
        entity_text="asthma",
        context_window="Patient has asthma.",
        split="train",
    )
    defaults.update(kwargs)
    return Instance(**defaults)


def _manifest(**kwargs) -> TrainingManifest:
    defaults = dict(
        iteration=0,
        base_model="dmis-lab/biobert-base-cased-v1.2",
        rule_version="v1.0",
        teacher_models=[],
        teacher_weights={},
        prompt_version="v1",
        num_original=100,
        num_silver=0,
        num_synthetic=0,
        synthetic_ratio=0.0,
        label_distribution={
            "confirmed": 60,
            "negated": 30,
            "associated_with_someone_else": 5,
            "other_non_patient": 5,
        },
        thresholds={
            "confirmed": 0.5,
            "negated": 0.5,
            "associated_with_someone_else": 0.5,
            "other_non_patient": 0.5,
        },
        validation_metrics={},
    )
    defaults.update(kwargs)
    return TrainingManifest(**defaults)


# ---------------------------------------------------------------------------
# Instance.__post_init__ — split validation
# ---------------------------------------------------------------------------


class TestSplitValidation:
    @pytest.mark.parametrize("split", ["train", "val", "test", "production"])
    def test_valid_splits_accepted(self, split):
        _inst(split=split)  # should not raise

    def test_invalid_split_raises(self):
        with pytest.raises(ValueError, match="split must be one of"):
            _inst(split="holdout")


# ---------------------------------------------------------------------------
# Instance.__post_init__ — source_type validation
# ---------------------------------------------------------------------------


class TestSourceTypeValidation:
    @pytest.mark.parametrize("source_type", ["original", "synthetic", "silver"])
    def test_valid_source_types_accepted(self, source_type):
        parent = None if source_type == "original" else "i-000"
        _inst(source_type=source_type, parent_instance_id=parent)

    def test_invalid_source_type_raises(self):
        with pytest.raises(ValueError, match="source_type must be one of"):
            _inst(source_type="augmented")


# ---------------------------------------------------------------------------
# Instance.__post_init__ — parent_instance_id constraint
# ---------------------------------------------------------------------------


class TestParentInstanceConstraint:
    def test_synthetic_without_parent_raises(self):
        with pytest.raises(ValueError, match="parent_instance_id is required"):
            _inst(source_type="synthetic")

    def test_silver_without_parent_raises(self):
        with pytest.raises(ValueError, match="parent_instance_id is required"):
            _inst(source_type="silver")

    def test_original_without_parent_accepted(self):
        _inst(source_type="original", parent_instance_id=None)

    def test_synthetic_with_parent_accepted(self):
        _inst(source_type="synthetic", parent_instance_id="i-000")


# ---------------------------------------------------------------------------
# Instance.__post_init__ — label vector length validation
# ---------------------------------------------------------------------------


class TestLabelLengthValidation:
    @pytest.mark.parametrize(
        "field_name",
        ["gold_labels", "rule_labels", "rule_probs", "biobert_probs", "biobert_labels"],
    )
    def test_wrong_length_raises(self, field_name):
        with pytest.raises(ValueError, match=f"{field_name} must have length"):
            _inst(**{field_name: [0, 1]})  # length 2, not NUM_LABELS

    @pytest.mark.parametrize(
        "field_name",
        ["gold_labels", "rule_labels", "rule_probs", "biobert_probs", "biobert_labels"],
    )
    def test_correct_length_accepted(self, field_name):
        _inst(**{field_name: [0] * NUM_LABELS})

    @pytest.mark.parametrize(
        "field_name",
        ["gold_labels", "rule_labels", "rule_probs", "biobert_probs", "biobert_labels"],
    )
    def test_none_accepted(self, field_name):
        _inst(**{field_name: None})


# ---------------------------------------------------------------------------
# Instance.from_raw
# ---------------------------------------------------------------------------


class TestFromRaw:
    def test_basic_construction(self):
        inst = Instance.from_raw(
            instance_id="i-002",
            note_id="n-002",
            entity_text="seizure",
            context_window="History of seizure disorder.",
            split="val",
        )
        assert inst.context_window == "History of seizure disorder."
        assert inst.split == "val"
        assert inst.instance_id == "i-002"

    def test_kwargs_passed_through(self):
        inst = Instance.from_raw(
            instance_id="i-003",
            note_id="n-003",
            entity_text="fever",
            context_window="Fever noted on admission.",
            split="train",
            gold_labels=[1, 0, 0, 0],
        )
        assert inst.gold_labels == [1, 0, 0, 0]

    def test_context_window_stored_verbatim(self):
        text = "  Patient has  asthma.  "
        inst = Instance.from_raw("i-x", "n-x", "asthma", text, "test")
        assert inst.context_window == text


# ---------------------------------------------------------------------------
# Instance to_dict / from_dict round-trip
# ---------------------------------------------------------------------------


class TestInstanceRoundTrip:
    def test_round_trip_minimal(self):
        original = _inst()
        restored = Instance.from_dict(original.to_dict())
        assert restored == original

    def test_round_trip_with_labels(self):
        original = _inst(
            gold_labels=[1, 0, 0, 0],
            rule_labels=[1, 0, 0, 0],
            rule_probs=[0.95, 0.0, 0.0, 0.0],
            rule_ids=["negation-pre-\\bno\\b"],
            biobert_probs=[0.8, 0.1, 0.05, 0.05],
            biobert_labels=[1, 0, 0, 0],
        )
        restored = Instance.from_dict(original.to_dict())
        assert restored == original

    def test_round_trip_via_json(self):
        original = _inst(gold_labels=[0, 1, 0, 0], split="val")
        serialized = json.dumps(original.to_dict())
        restored = Instance.from_dict(json.loads(serialized))
        assert restored == original

    def test_to_dict_has_no_private_fields(self):
        d = _inst().to_dict()
        assert not any(k.startswith("_VALID_") for k in d)

    def test_from_dict_ignores_unknown_keys(self):
        d = _inst().to_dict()
        d["future_field"] = "ignored"
        Instance.from_dict(d)  # should not raise

    def test_round_trip_synthetic_instance(self):
        original = _inst(source_type="synthetic", parent_instance_id="i-000")
        restored = Instance.from_dict(original.to_dict())
        assert restored.source_type == "synthetic"
        assert restored.parent_instance_id == "i-000"


# ---------------------------------------------------------------------------
# SyntheticAudit to_dict / from_dict round-trip
# ---------------------------------------------------------------------------


class TestSyntheticAuditRoundTrip:
    def _audit(self) -> SyntheticAudit:
        return SyntheticAudit(
            synthetic_id="s-001",
            parent_instance_id="i-001",
            target_labels=[1, 0, 0, 0],
            generation_prompt_version="v1",
            teacher_model="us.anthropic.claude-sonnet-4-6",
            rationale="Clearly confirmed.",
            validation_checks={"label_preserved": True, "dedup": True},
        )

    def test_round_trip(self):
        original = self._audit()
        restored = SyntheticAudit.from_dict(original.to_dict())
        assert restored == original

    def test_round_trip_via_json(self):
        original = self._audit()
        restored = SyntheticAudit.from_dict(json.loads(json.dumps(original.to_dict())))
        assert restored == original

    def test_approved_preserved(self):
        original = self._audit()
        restored = SyntheticAudit.from_dict(original.to_dict())
        assert restored.approved == original.approved


# ---------------------------------------------------------------------------
# TrainingManifest to_dict / from_dict round-trip
# ---------------------------------------------------------------------------


class TestTrainingManifestRoundTrip:
    def test_round_trip(self):
        original = _manifest()
        restored = TrainingManifest.from_dict(original.to_dict())
        assert restored == original

    def test_round_trip_via_json(self):
        original = _manifest(validation_metrics={"macro_f1": 0.82})
        restored = TrainingManifest.from_dict(json.loads(json.dumps(original.to_dict())))
        assert restored.validation_metrics["macro_f1"] == pytest.approx(0.82)

    def test_from_dict_ignores_unknown_keys(self):
        d = _manifest().to_dict()
        d["extra"] = "drop_me"
        TrainingManifest.from_dict(d)  # should not raise
