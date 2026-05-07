"""
Tests for M5: threshold_tuning.py

Verifies per-label threshold sweep on synthetic probability distributions.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from ods_phenocontext.threshold_tuning import tune_and_save, tune_thresholds

# ---------------------------------------------------------------------------
# tune_thresholds
# ---------------------------------------------------------------------------


class TestTuneThresholds:
    def test_perfect_separation_picks_correct_threshold(self):
        # Label 0: all positives have prob=0.9, negatives have prob=0.1
        # Optimal threshold should be somewhere between 0.1 and 0.9
        n = 50
        probs = np.zeros((n, 4))
        gold = np.zeros((n, 4), dtype=int)
        gold[:25, 0] = 1
        probs[:25, 0] = 0.9
        probs[25:, 0] = 0.1

        result = tune_thresholds(probs, gold, split="val")
        assert 0.1 <= result["confirmed"] <= 0.9

    def test_returns_all_label_names(self):
        from ods_phenocontext.schema import LABEL_NAMES

        n = 20
        probs = np.random.default_rng(42).random((n, 4))
        gold = np.random.default_rng(42).integers(0, 2, size=(n, 4))

        result = tune_thresholds(probs, gold, split="val")
        assert set(result.keys()) == set(LABEL_NAMES)

    def test_thresholds_in_valid_range(self):
        n = 30
        probs = np.random.default_rng(7).random((n, 4))
        gold = np.random.default_rng(7).integers(0, 2, size=(n, 4))

        result = tune_thresholds(probs, gold, split="val")
        for t in result.values():
            assert 0.05 <= t <= 0.95

    def test_rejects_test_split(self):
        probs = np.zeros((5, 4))
        gold = np.zeros((5, 4), dtype=int)
        with pytest.raises(ValueError, match="val split"):
            tune_thresholds(probs, gold, split="test")

    def test_rejects_train_split(self):
        probs = np.zeros((5, 4))
        gold = np.zeros((5, 4), dtype=int)
        with pytest.raises(ValueError, match="val split"):
            tune_thresholds(probs, gold, split="train")

    def test_rejects_unsupported_objective(self):
        probs = np.zeros((5, 4))
        gold = np.zeros((5, 4), dtype=int)
        with pytest.raises(ValueError, match="Unsupported objective"):
            tune_thresholds(probs, gold, split="val", objective="accuracy")

    def test_wrong_column_count_raises(self):
        probs = np.zeros((5, 3))
        gold = np.zeros((5, 4), dtype=int)
        with pytest.raises(ValueError, match="Expected 4 columns"):
            tune_thresholds(probs, gold, split="val")

    def test_maximizes_f1(self):
        # Construct a case where the best threshold for label 0 is clearly ~0.3
        # (positives cluster at 0.4, negatives at 0.1)
        rng = np.random.default_rng(123)
        n = 100
        gold = np.zeros((n, 4), dtype=int)
        probs = np.full((n, 4), 0.1)
        gold[:40, 0] = 1
        probs[:40, 0] = rng.uniform(0.35, 0.55, size=40)
        probs[40:, 0] = rng.uniform(0.0, 0.2, size=60)

        result = tune_thresholds(probs, gold, split="val")
        # Threshold should be in the gap between distributions
        assert 0.15 <= result["confirmed"] <= 0.45

    def test_accepts_list_inputs(self):
        probs = [[0.9, 0.1, 0.1, 0.1], [0.1, 0.9, 0.1, 0.1]]
        gold = [[1, 0, 0, 0], [0, 1, 0, 0]]
        result = tune_thresholds(probs, gold, split="val")
        assert len(result) == 4


# ---------------------------------------------------------------------------
# tune_and_save
# ---------------------------------------------------------------------------


class TestTuneAndSave:
    def test_writes_json_file(self, tmp_path: Path):
        n = 20
        probs = np.random.default_rng(42).random((n, 4))
        gold = np.random.default_rng(42).integers(0, 2, size=(n, 4))

        out_path = tmp_path / "thresholds.json"
        tune_and_save(probs, gold, split="val", output_path=out_path)

        assert out_path.exists()
        data = json.loads(out_path.read_text())
        assert "thresholds" in data
        assert "val_metrics" in data
        assert data["split"] == "val"
        assert data["objective"] == "f1"

    def test_output_contains_per_label_f1(self, tmp_path: Path):
        from ods_phenocontext.schema import LABEL_NAMES

        n = 20
        probs = np.random.default_rng(42).random((n, 4))
        gold = np.random.default_rng(42).integers(0, 2, size=(n, 4))

        out_path = tmp_path / "sub" / "thresholds.json"
        result = tune_and_save(probs, gold, split="val", output_path=out_path)

        for name in LABEL_NAMES:
            assert f"f1_{name}" in result["val_metrics"]
            assert 0.0 <= result["val_metrics"][f"f1_{name}"] <= 1.0

    def test_creates_parent_dirs(self, tmp_path: Path):
        n = 10
        probs = np.random.default_rng(0).random((n, 4))
        gold = np.random.default_rng(0).integers(0, 2, size=(n, 4))

        out_path = tmp_path / "a" / "b" / "c" / "thresholds.json"
        tune_and_save(probs, gold, split="val", output_path=out_path)
        assert out_path.exists()
