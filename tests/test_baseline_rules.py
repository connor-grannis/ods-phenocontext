"""
Tests for M7: Baseline 1 — rules-only stage.

Verifies artifact structure and rule-specific metrics (abstention rate,
rule_id_counts) using a synthetic fixture. Does not assert metric values
since those depend on the real model and data.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ods_phenocontext.experiments import ExperimentConfig, run_experiment, save_experiment
from ods_phenocontext.rules import RuleClassifier
from ods_phenocontext.schema import NUM_LABELS, Instance

# ---------------------------------------------------------------------------
# Synthetic fixture
# ---------------------------------------------------------------------------

_INSTANCES = [
    # Negated — rules should fire
    Instance(
        instance_id="i-001",
        note_id="n-001",
        entity_text="asthma",
        context_window="No [ENT] asthma [/ENT] noted.",
        split="val",
        gold_labels=[0, 1, 0, 0],
    ),
    # Confirmed — no triggers
    Instance(
        instance_id="i-002",
        note_id="n-002",
        entity_text="diabetes",
        context_window="Patient has [ENT] diabetes [/ENT].",
        split="val",
        gold_labels=[1, 0, 0, 0],
    ),
    # Family history — associated_with_someone_else
    Instance(
        instance_id="i-003",
        note_id="n-003",
        entity_text="hypertension",
        context_window="Family history of [ENT] hypertension [/ENT].",
        split="val",
        gold_labels=[0, 0, 1, 0],
    ),
    # Abstention trigger — should be treated as no-prediction in rules_only
    Instance(
        instance_id="i-004",
        note_id="n-004",
        entity_text="cancer",
        context_window="[ENT] cancer [/ENT] can cause fatigue.",
        split="val",
        gold_labels=[0, 0, 0, 1],
    ),
    # Train split — must be excluded from evaluation
    Instance(
        instance_id="i-005",
        note_id="n-005",
        entity_text="fever",
        context_window="[ENT] fever [/ENT] resolved.",
        split="train",
        gold_labels=[1, 0, 0, 0],
    ),
]


@pytest.fixture(scope="module")
def rules_result():
    cfg = ExperimentConfig(name="baseline_rules", stage="rules_only")
    return run_experiment(cfg, _INSTANCES, rules_model=RuleClassifier())


# ---------------------------------------------------------------------------
# Metric structure
# ---------------------------------------------------------------------------


class TestRulesOnlyMetrics:
    def test_n_evaluated_excludes_train(self, rules_result):
        assert rules_result.metrics["n_evaluated"] == 4

    def test_abstention_rate_present(self, rules_result):
        rate = rules_result.metrics["abstention_rate"]
        assert 0.0 <= rate <= 1.0

    def test_abstention_rate_nonzero(self, rules_result):
        # "can" triggers llm_review; at least i-004 should abstain
        assert rules_result.metrics["abstention_rate"] > 0.0

    def test_rule_id_counts_present(self, rules_result):
        assert "rule_id_counts" in rules_result.metrics
        assert isinstance(rules_result.metrics["rule_id_counts"], dict)

    def test_rule_id_counts_nonempty(self, rules_result):
        # At least the negation and family rules should have fired
        assert len(rules_result.metrics["rule_id_counts"]) > 0

    def test_per_label_f1_keys_present(self, rules_result):
        from ods_phenocontext.schema import LABEL_NAMES

        for name in LABEL_NAMES:
            assert f"f1_{name}" in rules_result.metrics

    def test_source_coverage_rules_only(self, rules_result):
        coverage = rules_result.metrics["source_coverage"]
        assert coverage["biobert"]["count"] == 0

    def test_macro_f1_present(self, rules_result):
        assert "macro_f1" in rules_result.metrics
        assert 0.0 <= rules_result.metrics["macro_f1"] <= 1.0


# ---------------------------------------------------------------------------
# Prediction structure
# ---------------------------------------------------------------------------


class TestRulesOnlyPredictions:
    def test_prediction_count(self, rules_result):
        assert len(rules_result.predictions) == 4

    def test_predictions_have_required_keys(self, rules_result):
        required = {"instance_id", "source", "labels", "probs", "gold_labels", "abstained"}
        for pred in rules_result.predictions:
            assert required.issubset(pred.keys())

    def test_predictions_source_is_rules(self, rules_result):
        assert all(p["source"] == "rules" for p in rules_result.predictions)

    def test_labels_are_binary(self, rules_result):
        for pred in rules_result.predictions:
            assert all(v in (0, 1) for v in pred["labels"])
            assert len(pred["labels"]) == NUM_LABELS

    def test_probs_in_range(self, rules_result):
        for pred in rules_result.predictions:
            assert all(0.0 <= p <= 1.0 for p in pred["probs"])

    def test_negated_instance_predicted_negated(self, rules_result):
        neg_pred = next(p for p in rules_result.predictions if p["instance_id"] == "i-001")
        assert neg_pred["labels"][1] == 1  # negated

    def test_confirmed_instance_predicted_confirmed(self, rules_result):
        conf_pred = next(p for p in rules_result.predictions if p["instance_id"] == "i-002")
        assert conf_pred["labels"][0] == 1  # confirmed

    def test_family_instance_predicted_family(self, rules_result):
        fam_pred = next(p for p in rules_result.predictions if p["instance_id"] == "i-003")
        assert fam_pred["labels"][2] == 1  # associated_with_someone_else


# ---------------------------------------------------------------------------
# Artifact structure
# ---------------------------------------------------------------------------


class TestRulesOnlyArtifacts:
    def test_artifacts_written(self, rules_result, tmp_path: Path):
        exp_dir = save_experiment(rules_result, tmp_path)
        assert (exp_dir / "config.json").exists()
        assert (exp_dir / "metrics.json").exists()
        assert (exp_dir / "predictions.jsonl").exists()

    def test_metrics_json_contains_abstention_rate(self, rules_result, tmp_path: Path):
        save_experiment(rules_result, tmp_path)
        data = json.loads((tmp_path / "baseline_rules" / "metrics.json").read_text())
        assert "abstention_rate" in data["metrics"]

    def test_metrics_json_contains_rule_id_counts(self, rules_result, tmp_path: Path):
        save_experiment(rules_result, tmp_path)
        data = json.loads((tmp_path / "baseline_rules" / "metrics.json").read_text())
        assert "rule_id_counts" in data["metrics"]

    def test_predictions_jsonl_has_abstained_field(self, rules_result, tmp_path: Path):
        save_experiment(rules_result, tmp_path)
        lines = (tmp_path / "baseline_rules" / "predictions.jsonl").read_text().strip().split("\n")
        for line in lines:
            pred = json.loads(line)
            assert "abstained" in pred
