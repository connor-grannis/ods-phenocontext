"""
Tests for M8: Baseline 2 — BioBERT-only stage.

Uses a fake BioBERT predictor to verify artifact structure and pipeline
routing. Does not assert metric values — the head is untrained.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ods_phenocontext.experiments import ExperimentConfig, run_experiment, save_experiment
from ods_phenocontext.schema import NUM_LABELS, Instance

# ---------------------------------------------------------------------------
# Synthetic fixture
# ---------------------------------------------------------------------------

_INSTANCES = [
    Instance(
        instance_id="i-001",
        note_id="n-001",
        entity_text="asthma",
        context_window="No [ENT] asthma [/ENT] noted.",
        split="val",
        gold_labels=[0, 1, 0, 0],
    ),
    Instance(
        instance_id="i-002",
        note_id="n-002",
        entity_text="diabetes",
        context_window="Patient has [ENT] diabetes [/ENT].",
        split="val",
        gold_labels=[1, 0, 0, 0],
    ),
    Instance(
        instance_id="i-003",
        note_id="n-003",
        entity_text="hypertension",
        context_window="Family history of [ENT] hypertension [/ENT].",
        split="val",
        gold_labels=[0, 0, 1, 0],
    ),
    # Train split — must be excluded from evaluation
    Instance(
        instance_id="i-004",
        note_id="n-004",
        entity_text="fever",
        context_window="[ENT] fever [/ENT] resolved.",
        split="train",
        gold_labels=[1, 0, 0, 0],
    ),
]


class _FakeBioBERT:
    """Returns fixed probabilities — deterministic, no model load required."""

    def predict_proba(self, instance: Instance) -> list[float]:
        return [0.8, 0.2, 0.1, 0.1]


@pytest.fixture(scope="module")
def biobert_result():
    cfg = ExperimentConfig(name="baseline_biobert", stage="biobert_only")
    return run_experiment(cfg, _INSTANCES, biobert_model=_FakeBioBERT())


# ---------------------------------------------------------------------------
# Metric structure
# ---------------------------------------------------------------------------


class TestBioBERTOnlyMetrics:
    def test_n_evaluated_excludes_train(self, biobert_result):
        assert biobert_result.metrics["n_evaluated"] == 3

    def test_abstention_rate_is_zero(self, biobert_result):
        # BioBERT never abstains
        assert biobert_result.metrics["abstention_rate"] == 0.0

    def test_rule_id_counts_empty(self, biobert_result):
        # No rules fired in biobert_only stage
        assert biobert_result.metrics["rule_id_counts"] == {}

    def test_per_label_f1_keys_present(self, biobert_result):
        from ods_phenocontext.schema import LABEL_NAMES

        for name in LABEL_NAMES:
            assert f"f1_{name}" in biobert_result.metrics

    def test_source_coverage_biobert_only(self, biobert_result):
        coverage = biobert_result.metrics["source_coverage"]
        assert coverage["rules"]["count"] == 0
        assert coverage["biobert"]["count"] == 3

    def test_macro_f1_present(self, biobert_result):
        assert "macro_f1" in biobert_result.metrics
        assert 0.0 <= biobert_result.metrics["macro_f1"] <= 1.0


# ---------------------------------------------------------------------------
# Prediction structure
# ---------------------------------------------------------------------------


class TestBioBERTOnlyPredictions:
    def test_prediction_count(self, biobert_result):
        assert len(biobert_result.predictions) == 3

    def test_predictions_have_required_keys(self, biobert_result):
        required = {"instance_id", "source", "labels", "probs", "gold_labels"}
        for pred in biobert_result.predictions:
            assert required.issubset(pred.keys())

    def test_predictions_source_is_biobert(self, biobert_result):
        assert all(p["source"] == "biobert" for p in biobert_result.predictions)

    def test_labels_are_binary(self, biobert_result):
        for pred in biobert_result.predictions:
            assert all(v in (0, 1) for v in pred["labels"])
            assert len(pred["labels"]) == NUM_LABELS

    def test_probs_in_range(self, biobert_result):
        for pred in biobert_result.predictions:
            assert all(0.0 <= p <= 1.0 for p in pred["probs"])

    def test_threshold_applied_at_0_5(self, biobert_result):
        # FakeBioBERT returns [0.8, 0.2, 0.1, 0.1]; at 0.5 only label 0 fires
        for pred in biobert_result.predictions:
            assert pred["labels"] == [1, 0, 0, 0]


# ---------------------------------------------------------------------------
# Artifact structure
# ---------------------------------------------------------------------------


class TestBioBERTOnlyArtifacts:
    def test_artifacts_written(self, biobert_result, tmp_path: Path):
        exp_dir = save_experiment(biobert_result, tmp_path)
        assert (exp_dir / "config.json").exists()
        assert (exp_dir / "metrics.json").exists()
        assert (exp_dir / "predictions.jsonl").exists()

    def test_config_json_stage(self, biobert_result, tmp_path: Path):
        save_experiment(biobert_result, tmp_path)
        data = json.loads((tmp_path / "baseline_biobert" / "config.json").read_text())
        assert data["stage"] == "biobert_only"

    def test_metrics_json_structure(self, biobert_result, tmp_path: Path):
        save_experiment(biobert_result, tmp_path)
        data = json.loads((tmp_path / "baseline_biobert" / "metrics.json").read_text())
        assert "metrics" in data
        assert "timestamp" in data
        assert data["eval_split"] == "val"

    def test_predictions_jsonl_line_count(self, biobert_result, tmp_path: Path):
        save_experiment(biobert_result, tmp_path)
        lines = (
            (tmp_path / "baseline_biobert" / "predictions.jsonl").read_text().strip().split("\n")
        )
        assert len(lines) == 3
