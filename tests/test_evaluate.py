"""
Tests for M5: evaluate.py

Uses synthetic predictions with known outcomes to verify metric computation.
"""

from __future__ import annotations

import pytest

from ods_phenocontext.evaluate import compute_metrics, slice_metrics
from ods_phenocontext.schema import LABEL_NAMES, Instance


def _inst(
    gold: list[int],
    rule_labels: list[int] | None = None,
    rule_probs: list[float] | None = None,
    rule_abstained: bool = False,
    biobert_labels: list[int] | None = None,
    biobert_probs: list[float] | None = None,
    split: str = "val",
    context: str = "Patient has [ENT] asthma [/ENT].",
) -> Instance:
    return Instance(
        instance_id="i-001",
        note_id="n-001",
        entity_text="asthma",
        context_window=context,
        split=split,
        gold_labels=gold,
        rule_labels=rule_labels,
        rule_probs=rule_probs,
        rule_abstained=rule_abstained,
        biobert_labels=biobert_labels,
        biobert_probs=biobert_probs,
    )


# ---------------------------------------------------------------------------
# compute_metrics
# ---------------------------------------------------------------------------


class TestComputeMetrics:
    def test_perfect_predictions(self):
        instances = [
            _inst([1, 0, 0, 0], rule_labels=[1, 0, 0, 0], rule_probs=[0.9, 0.1, 0.1, 0.1]),
            _inst([0, 1, 0, 0], rule_labels=[0, 1, 0, 0], rule_probs=[0.1, 0.9, 0.1, 0.1]),
            _inst([0, 0, 1, 0], rule_labels=[0, 0, 1, 0], rule_probs=[0.1, 0.1, 0.9, 0.1]),
            _inst([0, 0, 0, 1], rule_labels=[0, 0, 0, 1], rule_probs=[0.1, 0.1, 0.1, 0.9]),
        ]
        m = compute_metrics(instances)
        assert m["macro_f1"] == pytest.approx(1.0)
        assert m["micro_f1"] == pytest.approx(1.0)
        for name in LABEL_NAMES:
            assert m[f"f1_{name}"] == pytest.approx(1.0)

    def test_all_wrong_predictions(self):
        instances = [
            _inst([1, 0, 0, 0], rule_labels=[0, 1, 0, 0], rule_probs=[0.1, 0.9, 0.1, 0.1]),
            _inst([0, 1, 0, 0], rule_labels=[1, 0, 0, 0], rule_probs=[0.9, 0.1, 0.1, 0.1]),
        ]
        m = compute_metrics(instances)
        assert m["f1_confirmed"] == 0.0
        assert m["f1_negated"] == 0.0

    def test_confusion_counts(self):
        instances = [
            _inst([1, 0, 0, 0], rule_labels=[1, 0, 0, 0], rule_probs=[0.9, 0.1, 0.1, 0.1]),
            _inst([1, 0, 0, 0], rule_labels=[0, 0, 0, 0], rule_probs=[0.1, 0.1, 0.1, 0.1]),
        ]
        m = compute_metrics(instances)
        assert m["tp_confirmed"] == 1
        assert m["fn_confirmed"] == 1
        assert m["fp_confirmed"] == 0

    def test_source_coverage_rules(self):
        instances = [
            _inst([1, 0, 0, 0], rule_labels=[1, 0, 0, 0], rule_probs=[0.9, 0.1, 0.1, 0.1]),
        ]
        m = compute_metrics(instances)
        assert m["source_coverage"]["rules"]["count"] == 1
        assert m["source_coverage"]["biobert"]["count"] == 0

    def test_source_coverage_biobert(self):
        instances = [
            _inst(
                [1, 0, 0, 0],
                rule_abstained=True,
                biobert_labels=[1, 0, 0, 0],
                biobert_probs=[0.9, 0.1, 0.1, 0.1],
            ),
        ]
        m = compute_metrics(instances)
        assert m["source_coverage"]["biobert"]["count"] == 1
        assert m["source_coverage"]["rules"]["count"] == 0

    def test_empty_instances_returns_empty(self):
        assert compute_metrics([]) == {}

    def test_instances_without_gold_skipped(self):
        inst = Instance(
            instance_id="i-x",
            note_id="n-x",
            entity_text="x",
            context_window="x",
            split="val",
            gold_labels=None,
            rule_labels=[1, 0, 0, 0],
        )
        assert compute_metrics([inst]) == {}

    def test_n_evaluated(self):
        instances = [
            _inst([1, 0, 0, 0], rule_labels=[1, 0, 0, 0], rule_probs=[0.9, 0.1, 0.1, 0.1]),
            _inst([0, 1, 0, 0], rule_labels=[0, 1, 0, 0], rule_probs=[0.1, 0.9, 0.1, 0.1]),
        ]
        m = compute_metrics(instances)
        assert m["n_evaluated"] == 2

    def test_pr_auc_present_when_computable(self):
        instances = [
            _inst([1, 0, 0, 0], rule_labels=[1, 0, 0, 0], rule_probs=[0.9, 0.1, 0.1, 0.1]),
            _inst([0, 0, 0, 0], rule_labels=[0, 0, 0, 0], rule_probs=[0.2, 0.1, 0.1, 0.1]),
        ]
        m = compute_metrics(instances)
        assert "pr_auc_confirmed" in m
        assert 0.0 <= m["pr_auc_confirmed"] <= 1.0


# ---------------------------------------------------------------------------
# slice_metrics
# ---------------------------------------------------------------------------


class TestSliceMetrics:
    def test_slices_by_split(self):
        instances = [
            _inst(
                [1, 0, 0, 0], rule_labels=[1, 0, 0, 0], rule_probs=[0.9, 0.1, 0.1, 0.1], split="val"
            ),
            _inst(
                [0, 1, 0, 0],
                rule_labels=[0, 1, 0, 0],
                rule_probs=[0.1, 0.9, 0.1, 0.1],
                split="train",
            ),
        ]
        result = slice_metrics(instances, lambda i: i.split)
        assert "val" in result
        assert "train" in result
        assert result["val"]["n_evaluated"] == 1
        assert result["train"]["n_evaluated"] == 1

    def test_slices_by_context_length(self):
        short = _inst(
            [1, 0, 0, 0],
            rule_labels=[1, 0, 0, 0],
            rule_probs=[0.9, 0.1, 0.1, 0.1],
            context="short [ENT] x [/ENT]",
        )
        long = _inst(
            [0, 1, 0, 0],
            rule_labels=[0, 1, 0, 0],
            rule_probs=[0.1, 0.9, 0.1, 0.1],
            context="a " * 100 + "[ENT] x [/ENT]",
        )
        result = slice_metrics(
            [short, long],
            lambda i: "short" if len(i.context_window) < 50 else "long",
        )
        assert "short" in result
        assert "long" in result
