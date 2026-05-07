"""
Tests for schema.py dataclasses.

Uses only synthetic strings — no PHI, no real notes.
"""

from ods_phenocontext.schema import (
    LABEL_NAMES,
    NUM_LABELS,
    Instance,
    SyntheticAudit,
    TrainingManifest,
)


def _make_instance(**kwargs) -> Instance:
    """Minimal valid Instance with overridable fields."""
    defaults = dict(
        instance_id="i-001",
        note_id="n-001",
        entity_text="asthma",
        context_window="The patient denies any history of asthma.",
        split="train",
    )
    defaults.update(kwargs)
    return Instance(**defaults)


# --- Instance ---


def test_instance_minimal_fields():
    inst = _make_instance()
    assert inst.instance_id == "i-001"
    assert inst.rule_abstained is False
    assert inst.source_type == "original"
    assert inst.gold_labels is None
    assert inst.teacher_outputs == {}


def test_instance_with_gold_labels():
    inst = _make_instance(gold_labels=[1, 0, 0, 0])
    assert inst.gold_labels == [1, 0, 0, 0]
    assert len(inst.gold_labels) == NUM_LABELS


def test_instance_label_names_length():
    assert len(LABEL_NAMES) == NUM_LABELS == 4


def test_instance_round_trip_fields():
    inst = _make_instance(
        gold_labels=[0, 1, 0, 0],
        rule_labels=[0, 1, 0, 0],
        rule_probs=[0.05, 0.95, 0.1, 0.05],
        rule_ids=["negation-001"],
        rule_abstained=False,
        biobert_probs=None,
        biobert_labels=None,
        source_type="original",
    )
    assert inst.rule_abstained is False
    assert inst.rule_ids == ["negation-001"]
    assert inst.rule_probs[1] == 0.95  # type: ignore[index]


def test_instance_synthetic_provenance():
    inst = _make_instance(
        instance_id="s-001",
        source_type="synthetic",
        parent_instance_id="i-001",
    )
    assert inst.source_type == "synthetic"
    assert inst.parent_instance_id == "i-001"


# --- SyntheticAudit ---


def test_synthetic_audit_approved_when_all_checks_pass():
    audit = SyntheticAudit(
        synthetic_id="s-001",
        parent_instance_id="i-001",
        target_labels=[1, 0, 0, 0],
        generation_prompt_version="v1",
        teacher_model="us.anthropic.claude-sonnet-4-6",
        rationale="Clearly confirmed.",
        validation_checks={
            "label_preserved": True,
            "embedding_similarity": True,
            "lexical_diversity": True,
            "dedup": True,
            "manual_review": True,
        },
    )
    assert audit.approved is True


def test_synthetic_audit_rejected_when_any_check_fails():
    audit = SyntheticAudit(
        synthetic_id="s-002",
        parent_instance_id="i-001",
        target_labels=[1, 0, 0, 0],
        generation_prompt_version="v1",
        teacher_model="us.anthropic.claude-sonnet-4-6",
        rationale="Maybe confirmed.",
        validation_checks={
            "label_preserved": True,
            "embedding_similarity": False,  # failed
            "lexical_diversity": True,
            "dedup": True,
            "manual_review": True,
        },
    )
    assert audit.approved is False


def test_synthetic_audit_not_approved_when_no_checks():
    audit = SyntheticAudit(
        synthetic_id="s-003",
        parent_instance_id="i-001",
        target_labels=[0, 1, 0, 0],
        generation_prompt_version="v1",
        teacher_model="us.anthropic.claude-sonnet-4-6",
        rationale="Negated.",
    )
    assert audit.approved is False


# --- TrainingManifest ---


def test_training_manifest_fields():
    manifest = TrainingManifest(
        iteration=1,
        base_model="dmis-lab/biobert-base-cased-v1.2",
        rule_version="v1.0",
        teacher_models=["us.anthropic.claude-sonnet-4-6"],
        teacher_weights={"generalist": 0.4, "precision_biased": 0.25, "recall_biased": 0.2},
        prompt_version="v1",
        num_original=500,
        num_silver=0,
        num_synthetic=100,
        synthetic_ratio=0.2,
        label_distribution={
            "confirmed": 300,
            "negated": 150,
            "associated_with_someone_else": 50,
            "other_non_patient": 80,
        },
        thresholds={
            "confirmed": 0.5,
            "negated": 0.45,
            "associated_with_someone_else": 0.55,
            "other_non_patient": 0.5,
        },
        validation_metrics={"macro_f1": 0.82, "f1_confirmed": 0.88},
    )
    assert manifest.iteration == 1
    assert manifest.synthetic_ratio == 0.2
    assert manifest.thresholds["negated"] == 0.45
