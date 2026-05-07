"""
Tests for M10: Baseline 4 — rules-first + BioBERT + tuned thresholds.

Verifies that tuned thresholds are applied correctly and that the
tune_and_save / run_experiment round-trip produces valid artifacts.
Does not assert final metric values.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from ods_phenocontext.experiments import ExperimentConfig, run_experiment, save_experiment
from ods_phenocontext.rules import RuleClassifier
from ods_phenocontext.schema import LABEL_NAMES, Instance
from ods_phenocontext.threshold_tuning import tune_thresholds

# ---------------------------------------------------------------------------
# Synthetic fixture
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
    # Rules confident — confirmed
    Instance(
        instance_id="i-002",
        note_id="n-002",
        entity_text="diabetes",
        context_window="Patient has [ENT] diabetes [/ENT].",
        split="val",
        gold_labels=[1, 0, 0, 0],
    ),
    # Rules abstain — routed to BioBERT
    Instance(
        instance_id="i-003",
        note_id="n-003",
        entity_text="cancer",
        context_window="[ENT] cancer [/ENT] can cause fatigue.",
        split="val",
        gold_labels=[0, 0, 0, 1],
    ),
    # Train split — excluded from eval
    Instance(
        instance_id="i-004",
        note_id="n-004",
        entity_text="fever",
        context_window="[ENT] fever [/ENT] resolved.",
        split="train",
        gold_labels=[1, 0, 0, 0],
    ),
]

# Tuned thresholds derived from synthetic val probs/gold
# Label 3 (other_non_patient) gets a low threshold so i-003 fires correctly
_TUNED_THRESHOLDS = {name: 0.5 for name in LABEL_NAMES}
_TUNED_THRESHOLDS["other_non_patient"] = 0.3


class _FakeBioBERT:
    """Returns probs that cross the tuned threshold for other_non_patient."""

    def predict_proba(self, instance: Instance) -> list[float]:
        return [0.1, 0.1, 0.1, 0.4]


@pytest.fixture(scope="module")
def tuned_result():
    cfg = ExperimentConfig(
        name="baseline_rules_then_biobert_tuned",
        stage="rules_then_biobert_tuned",
        thresholds=_TUNED_THRESHOLDS,
    )
    return run_experiment(
        cfg, _INSTANCES, rules_model=RuleClassifier(), biobert_model=_FakeBioBERT()
    )


# ---------------------------------------------------------------------------
# Threshold application
# ---------------------------------------------------------------------------


class TestTunedThresholdApplication:
    def test_tuned_threshold_fires_on_biobert_instance(self, tuned_result):
        # i-003: prob[3]=0.4 >= threshold 0.3 → label 3 should be 1
        pred = next(p for p in tuned_result.predictions if p["instance_id"] == "i-003")
        assert pred["labels"][3] == 1

    def test_default_threshold_blocks_low_prob(self, tuned_result):
        # i-003: prob[0]=0.1 < threshold 0.5 → label 0 should be 0
        pred = next(p for p in tuned_result.predictions if p["instance_id"] == "i-003")
        assert pred["labels"][0] == 0

    def test_rules_instances_unaffected_by_thresholds(self, tuned_result):
        # Rules-handled instances use rule probs, not BioBERT probs
        pred = next(p for p in tuned_result.predictions if p["instance_id"] == "i-001")
        assert pred["source"] == "rules"
        assert pred["labels"][1] == 1  # negated via rules


# ---------------------------------------------------------------------------
# Source distribution
# ---------------------------------------------------------------------------


class TestTunedSourceDistribution:
    def test_n_evaluated(self, tuned_result):
        assert tuned_result.metrics["n_evaluated"] == 3

    def test_rules_coverage(self, tuned_result):
        assert tuned_result.metrics["source_coverage"]["rules"]["count"] == 2

    def test_biobert_coverage(self, tuned_result):
        assert tuned_result.metrics["source_coverage"]["biobert"]["count"] == 1

    def test_abstention_rate(self, tuned_result):
        assert tuned_result.metrics["abstention_rate"] == pytest.approx(1 / 3)


# ---------------------------------------------------------------------------
# tune_thresholds round-trip
# ---------------------------------------------------------------------------


class TestTuneThresholdsRoundTrip:
    def test_tuned_thresholds_differ_from_default(self):
        # Construct val data where label 3 positives cluster at prob=0.4
        rng = np.random.default_rng(42)
        n = 60
        probs = np.full((n, 4), 0.1)
        gold = np.zeros((n, 4), dtype=int)
        gold[:20, 3] = 1
        probs[:20, 3] = rng.uniform(0.35, 0.55, size=20)
        probs[20:, 3] = rng.uniform(0.05, 0.25, size=40)

        thresholds = tune_thresholds(probs, gold, split="val")
        # Optimal threshold for label 3 should be well below 0.5
        assert thresholds["other_non_patient"] < 0.5

    def test_tuned_thresholds_have_all_labels(self):
        probs = np.random.default_rng(0).random((20, 4))
        gold = np.random.default_rng(0).integers(0, 2, size=(20, 4))
        thresholds = tune_thresholds(probs, gold, split="val")
        assert set(thresholds.keys()) == set(LABEL_NAMES)


# ---------------------------------------------------------------------------
# Artifact structure
# ---------------------------------------------------------------------------


class TestTunedArtifacts:
    def test_artifacts_written(self, tuned_result, tmp_path: Path):
        exp_dir = save_experiment(tuned_result, tmp_path)
        assert (exp_dir / "config.json").exists()
        assert (exp_dir / "metrics.json").exists()
        assert (exp_dir / "predictions.jsonl").exists()

    def test_config_records_thresholds(self, tuned_result, tmp_path: Path):
        save_experiment(tuned_result, tmp_path)
        data = json.loads(
            (tmp_path / "baseline_rules_then_biobert_tuned" / "config.json").read_text()
        )
        assert data["thresholds"]["other_non_patient"] == pytest.approx(0.3)

    def test_predictions_jsonl_line_count(self, tuned_result, tmp_path: Path):
        save_experiment(tuned_result, tmp_path)
        lines = (
            (tmp_path / "baseline_rules_then_biobert_tuned" / "predictions.jsonl")
            .read_text()
            .strip()
            .split("\n")
        )
        assert len(lines) == 3
