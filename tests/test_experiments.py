"""
Tests for M6: experiments.py

Verifies experiment runner orchestration, artifact structure, and stage routing.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ods_phenocontext.experiments import (
    ExperimentConfig,
    run_experiment,
    save_experiment,
)
from ods_phenocontext.schema import LABEL_NAMES, NUM_LABELS, Instance

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _inst(
    instance_id: str = "i-001",
    gold: list[int] | None = None,
    split: str = "val",
) -> Instance:
    return Instance(
        instance_id=instance_id,
        note_id="n-001",
        entity_text="fever",
        context_window="No [ENT] fever [/ENT] noted.",
        split=split,
        gold_labels=gold or [0, 1, 0, 0],
    )


class FakeRules:
    """Always returns negated with confidence."""

    def __call__(self, instance: Instance) -> dict:
        return {
            "abstained": False,
            "labels": [0, 1, 0, 0],
            "probs": [0.1, 0.95, 0.1, 0.1],
            "rule_ids": ["neg_prefix"],
        }


class AbstainingRules:
    """Always abstains."""

    def __call__(self, instance: Instance) -> dict:
        return {
            "abstained": True,
            "labels": [0, 0, 0, 0],
            "probs": [0.0, 0.0, 0.0, 0.0],
            "rule_ids": [],
        }


class FakeBioBERT:
    """Returns fixed probabilities."""

    def predict_proba(self, instance: Instance) -> list[float]:
        return [0.8, 0.2, 0.1, 0.1]


# ---------------------------------------------------------------------------
# ExperimentConfig
# ---------------------------------------------------------------------------


class TestExperimentConfig:
    def test_valid_config(self):
        cfg = ExperimentConfig(name="test_run", stage="rules_only")
        assert cfg.eval_split == "val"

    def test_invalid_stage_raises(self):
        with pytest.raises(ValueError, match="stage must be"):
            ExperimentConfig(name="bad", stage="unknown_stage")

    def test_invalid_eval_split_raises(self):
        with pytest.raises(ValueError, match="eval_split"):
            ExperimentConfig(name="bad", stage="rules_only", eval_split="train")

    def test_default_thresholds(self):
        cfg = ExperimentConfig(name="t", stage="rules_only")
        assert cfg.threshold_list == [0.5] * NUM_LABELS

    def test_custom_thresholds(self):
        thresholds = {name: 0.3 for name in LABEL_NAMES}
        cfg = ExperimentConfig(name="t", stage="biobert_only", thresholds=thresholds)
        assert cfg.threshold_list == [0.3] * NUM_LABELS


# ---------------------------------------------------------------------------
# run_experiment
# ---------------------------------------------------------------------------


class TestRunExperiment:
    def test_rules_only_stage(self):
        instances = [_inst("i-001"), _inst("i-002")]
        cfg = ExperimentConfig(name="baseline_rules", stage="rules_only")
        result = run_experiment(cfg, instances, rules_model=FakeRules())

        assert result.metrics["n_evaluated"] == 2
        assert all(p["source"] == "rules" for p in result.predictions)

    def test_biobert_only_stage(self):
        instances = [_inst("i-001")]
        cfg = ExperimentConfig(name="baseline_biobert", stage="biobert_only")
        result = run_experiment(cfg, instances, biobert_model=FakeBioBERT())

        assert result.metrics["n_evaluated"] == 1
        assert all(p["source"] == "biobert" for p in result.predictions)

    def test_rules_then_biobert_default(self):
        instances = [_inst("i-001")]
        cfg = ExperimentConfig(name="combined", stage="rules_then_biobert_default")
        result = run_experiment(
            cfg, instances, rules_model=FakeRules(), biobert_model=FakeBioBERT()
        )
        assert result.metrics["n_evaluated"] == 1

    def test_rules_then_biobert_tuned(self):
        thresholds = {name: 0.3 for name in LABEL_NAMES}
        instances = [_inst("i-001")]
        cfg = ExperimentConfig(
            name="tuned", stage="rules_then_biobert_tuned", thresholds=thresholds
        )
        result = run_experiment(
            cfg, instances, rules_model=AbstainingRules(), biobert_model=FakeBioBERT()
        )
        # With threshold 0.3 and probs [0.8, 0.2, 0.1, 0.1], label 0 should be 1
        assert result.predictions[0]["labels"][0] == 1
        assert result.predictions[0]["labels"][1] == 0

    def test_filters_to_eval_split(self):
        instances = [
            _inst("i-001", split="val"),
            _inst("i-002", split="train"),
            _inst("i-003", split="val"),
        ]
        cfg = ExperimentConfig(name="t", stage="rules_only")
        result = run_experiment(cfg, instances, rules_model=FakeRules())
        assert result.metrics["n_evaluated"] == 2

    def test_no_eval_instances_raises(self):
        instances = [_inst("i-001", split="train")]
        cfg = ExperimentConfig(name="t", stage="rules_only", eval_split="val")
        with pytest.raises(ValueError, match="No instances found"):
            run_experiment(cfg, instances, rules_model=FakeRules())

    def test_rules_only_without_model_raises(self):
        cfg = ExperimentConfig(name="t", stage="rules_only")
        with pytest.raises(ValueError, match="rules_model is required"):
            run_experiment(cfg, [_inst()], rules_model=None)

    def test_biobert_only_without_model_raises(self):
        cfg = ExperimentConfig(name="t", stage="biobert_only")
        with pytest.raises(ValueError, match="biobert_model is required"):
            run_experiment(cfg, [_inst()], biobert_model=None)

    def test_predictions_contain_gold_labels(self):
        instances = [_inst("i-001", gold=[1, 0, 0, 0])]
        cfg = ExperimentConfig(name="t", stage="rules_only")
        result = run_experiment(cfg, instances, rules_model=FakeRules())
        assert result.predictions[0]["gold_labels"] == [1, 0, 0, 0]

    def test_result_has_timestamp(self):
        instances = [_inst()]
        cfg = ExperimentConfig(name="t", stage="rules_only")
        result = run_experiment(cfg, instances, rules_model=FakeRules())
        assert result.timestamp is not None


# ---------------------------------------------------------------------------
# save_experiment
# ---------------------------------------------------------------------------


class TestSaveExperiment:
    def test_creates_experiment_dir(self, tmp_path: Path):
        instances = [_inst()]
        cfg = ExperimentConfig(name="test_exp", stage="rules_only")
        result = run_experiment(cfg, instances, rules_model=FakeRules())
        exp_dir = save_experiment(result, tmp_path)

        assert exp_dir == tmp_path / "test_exp"
        assert exp_dir.is_dir()

    def test_writes_config_json(self, tmp_path: Path):
        instances = [_inst()]
        cfg = ExperimentConfig(name="test_exp", stage="rules_only")
        result = run_experiment(cfg, instances, rules_model=FakeRules())
        save_experiment(result, tmp_path)

        config_data = json.loads((tmp_path / "test_exp" / "config.json").read_text())
        assert config_data["name"] == "test_exp"
        assert config_data["stage"] == "rules_only"

    def test_writes_metrics_json(self, tmp_path: Path):
        instances = [_inst()]
        cfg = ExperimentConfig(name="test_exp", stage="rules_only")
        result = run_experiment(cfg, instances, rules_model=FakeRules())
        save_experiment(result, tmp_path)

        metrics_data = json.loads((tmp_path / "test_exp" / "metrics.json").read_text())
        assert "metrics" in metrics_data
        assert "timestamp" in metrics_data
        assert metrics_data["stage"] == "rules_only"

    def test_writes_predictions_jsonl(self, tmp_path: Path):
        instances = [_inst("i-001"), _inst("i-002")]
        cfg = ExperimentConfig(name="test_exp", stage="rules_only")
        result = run_experiment(cfg, instances, rules_model=FakeRules())
        save_experiment(result, tmp_path)

        lines = (tmp_path / "test_exp" / "predictions.jsonl").read_text().strip().split("\n")
        assert len(lines) == 2
        pred = json.loads(lines[0])
        assert "instance_id" in pred
        assert "source" in pred
        assert "labels" in pred
        assert "probs" in pred

    def test_idempotent_overwrite(self, tmp_path: Path):
        instances = [_inst()]
        cfg = ExperimentConfig(name="test_exp", stage="rules_only")
        result = run_experiment(cfg, instances, rules_model=FakeRules())
        save_experiment(result, tmp_path)
        save_experiment(result, tmp_path)  # no error on second call
        assert (tmp_path / "test_exp" / "metrics.json").exists()
