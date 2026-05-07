"""
Smoke tests for the inference pipeline.

Uses lightweight fakes for the rule system and BioBERT — no model weights
loaded, no network calls.  Exercises both branches of phenocontext_predict:
  1. Rules confident → return rule output directly.
  2. Rules abstain → BioBERT probabilities thresholded and returned.
"""

from typing import Any

from ods_phenocontext.pipeline import phenocontext_predict
from ods_phenocontext.schema import NUM_LABELS, Instance

# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------

def _make_instance() -> Instance:
    return Instance(
        instance_id="t-001",
        note_id="n-001",
        entity_text="diabetes",
        context_window="The patient has a history of diabetes.",
        split="test",
    )


class _ConfidentRulesModel:
    """Fake rule system that always fires with high confidence."""

    def __call__(self, instance: Instance) -> dict[str, Any]:
        return {
            "abstained": False,
            "labels": [1, 0, 0, 0],
            "probs": [0.95, 0.02, 0.02, 0.01],
            "rule_ids": ["confirmed-history-001"],
        }


class _AbstainingRulesModel:
    """Fake rule system that always abstains."""

    def __call__(self, instance: Instance) -> dict[str, Any]:
        return {
            "abstained": True,
            "labels": [0] * NUM_LABELS,
            "probs": [0.0] * NUM_LABELS,
            "rule_ids": [],
        }


class _FakeBioBERT:
    """Fake BioBERT that returns fixed probabilities."""

    def __init__(self, probs: list[float]) -> None:
        self._probs = probs

    def predict_proba(self, instance: Instance) -> list[float]:
        return self._probs


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_pipeline_uses_rules_when_confident():
    result = phenocontext_predict(
        instance=_make_instance(),
        rules_model=_ConfidentRulesModel(),
        biobert_model=_FakeBioBERT([0.0] * NUM_LABELS),  # should not be called
        thresholds=[0.5] * NUM_LABELS,
    )
    assert result["source"] == "rules"
    assert result["labels"] == [1, 0, 0, 0]
    assert result["rule_ids"] == ["confirmed-history-001"]


def test_pipeline_falls_back_to_biobert_when_rules_abstain():
    # BioBERT returns probabilities that straddle the threshold
    probs = [0.8, 0.6, 0.3, 0.1]
    thresholds = [0.5, 0.5, 0.5, 0.5]

    result = phenocontext_predict(
        instance=_make_instance(),
        rules_model=_AbstainingRulesModel(),
        biobert_model=_FakeBioBERT(probs),
        thresholds=thresholds,
    )
    assert result["source"] == "biobert"
    assert result["labels"] == [1, 1, 0, 0]
    assert result["probs"] == probs
    assert "rule_ids" not in result


def test_pipeline_biobert_threshold_boundary():
    # Exactly at threshold → label is positive (>= is used)
    probs = [0.5, 0.5, 0.5, 0.5]
    thresholds = [0.5, 0.5, 0.5, 0.5]

    result = phenocontext_predict(
        instance=_make_instance(),
        rules_model=_AbstainingRulesModel(),
        biobert_model=_FakeBioBERT(probs),
        thresholds=thresholds,
    )
    assert result["labels"] == [1, 1, 1, 1]


def test_pipeline_output_label_length_matches_num_labels():
    result = phenocontext_predict(
        instance=_make_instance(),
        rules_model=_AbstainingRulesModel(),
        biobert_model=_FakeBioBERT([0.9] * NUM_LABELS),
        thresholds=[0.5] * NUM_LABELS,
    )
    assert len(result["labels"]) == NUM_LABELS
    assert len(result["probs"]) == NUM_LABELS
