"""
Tests for M9: Baseline 3 — rules-first + BioBERT fallback at default 0.5 threshold.

Verifies source distribution on a synthetic mix of confident-rule and
abstaining-rule instances. Does not assert metric values.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ods_phenocontext.experiments import ExperimentConfig, run_experiment, save_experiment
from ods_phenocontext.rules import RuleClassifier
from ods_phenocontext.schema import NUM_LABELS, Instance

# ---------------------------------------------------------------------------
# Synthetic fixture: mix of confident and abstaining cases
# ---------------------------------------------------------------------------

_INSTANCES = [
    # Rules confident — negation
    Instance(
        instance_id="i-001",
        note_id="n-001",
        entity_text="asthma",
        context_window="No [ENT] asthma [/ENT] noted.",
        split="val",
        gold_labels=[0, 1, 0, 0],
    ),
    # Rules confident — confirmed (no triggers)
    Instance(
        instance_id="i-002",
        note_id="n-002",
        entity_text="diabetes",
        context_window="Patient has [ENT] diabetes [/ENT].",
        split="val",
        gold_labels=[1, 0, 0, 0],
    ),
    # Rules confident — family
    Instance(
        instance_id="i-003",
        note_id="n-003",
        entity_text="hypertension",
        context_window="Family history of [ENT] hypertension [/ENT].",
        split="val",
        gold_labels=[0, 0, 1, 0],
    ),
    # Rules abstain — routed to BioBERT
    Instance(
        instance_id="i-004",
        note_id="n-004",
        entity_text="cancer",
        context_window="[ENT] cancer [/ENT] can cause fatigue.",
        split="val",
        gold_labels=[0, 0, 0, 1],
    ),
    # Train split — excluded
    Instance(
        instance_id="i-005",
        note_id="n-005",
        entity_text="fever",
        context_window="[ENT] fever [/ENT] resolved.",
        split="train",
        gold_labels=[1, 0, 0, 0],
    ),
]


class _FakeBioBERT:
    def predict_proba(self, instance: Instance) -> list[float]:
        return [0.2, 0.2, 0.2, 0.8]


@pytest.fixture(scope="module")
def combined_result():
    cfg = ExperimentConfig(name="baseline_rules_then_biobert", stage="rules_then_biobert_default")
    return run_experiment(
        cfg, _INSTANCES, rules_model=RuleClassifier(), biobert_model=_FakeBioBERT()
    )


# ---------------------------------------------------------------------------
# Source distribution
# ---------------------------------------------------------------------------


class TestSourceDistribution:
    def test_n_evaluated_excludes_train(self, combined_result):
        assert combined_result.metrics["n_evaluated"] == 4

    def test_rules_handled_confident_instances(self, combined_result):
        # i-001, i-002, i-003 are confident rule cases
        coverage = combined_result.metrics["source_coverage"]
        assert coverage["rules"]["count"] == 3

    def test_biobert_handled_abstaining_instance(self, combined_result):
        # i-004 triggers llm_review → routed to BioBERT
        coverage = combined_result.metrics["source_coverage"]
        assert coverage["biobert"]["count"] == 1

    def test_abstention_rate(self, combined_result):
        # 1 of 4 val instances routed to BioBERT
        assert combined_result.metrics["abstention_rate"] == pytest.approx(0.25)

    def test_rule_id_counts_nonempty(self, combined_result):
        assert len(combined_result.metrics["rule_id_counts"]) > 0


# ---------------------------------------------------------------------------
# Per-instance routing
# ---------------------------------------------------------------------------


class TestPerInstanceRouting:
    def test_confident_instances_use_rules(self, combined_result):
        rule_ids = {"i-001", "i-002", "i-003"}
        for pred in combined_result.predictions:
            if pred["instance_id"] in rule_ids:
                assert pred["source"] == "rules"

    def test_abstaining_instance_uses_biobert(self, combined_result):
        pred = next(p for p in combined_result.predictions if p["instance_id"] == "i-004")
        assert pred["source"] == "biobert"

    def test_biobert_threshold_applied(self, combined_result):
        # FakeBioBERT returns [0.2, 0.2, 0.2, 0.8]; at 0.5 only label 3 fires
        pred = next(p for p in combined_result.predictions if p["instance_id"] == "i-004")
        assert pred["labels"] == [0, 0, 0, 1]

    def test_negated_correctly_from_rules(self, combined_result):
        pred = next(p for p in combined_result.predictions if p["instance_id"] == "i-001")
        assert pred["labels"][1] == 1  # negated

    def test_labels_binary_all_instances(self, combined_result):
        for pred in combined_result.predictions:
            assert all(v in (0, 1) for v in pred["labels"])
            assert len(pred["labels"]) == NUM_LABELS

    def test_probs_in_range_all_instances(self, combined_result):
        for pred in combined_result.predictions:
            assert all(0.0 <= p <= 1.0 for p in pred["probs"])


# ---------------------------------------------------------------------------
# Artifact structure
# ---------------------------------------------------------------------------


class TestCombinedArtifacts:
    def test_artifacts_written(self, combined_result, tmp_path: Path):
        exp_dir = save_experiment(combined_result, tmp_path)
        assert (exp_dir / "config.json").exists()
        assert (exp_dir / "metrics.json").exists()
        assert (exp_dir / "predictions.jsonl").exists()

    def test_metrics_json_has_source_coverage(self, combined_result, tmp_path: Path):
        save_experiment(combined_result, tmp_path)
        data = json.loads((tmp_path / "baseline_rules_then_biobert" / "metrics.json").read_text())
        assert "source_coverage" in data["metrics"]

    def test_predictions_jsonl_line_count(self, combined_result, tmp_path: Path):
        save_experiment(combined_result, tmp_path)
        lines = (
            (tmp_path / "baseline_rules_then_biobert" / "predictions.jsonl")
            .read_text()
            .strip()
            .split("\n")
        )
        assert len(lines) == 4
