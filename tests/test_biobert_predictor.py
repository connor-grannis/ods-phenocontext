"""
Tests for M4: BioBERTPredictor adapter.

Loads dmis-lab/biobert-base-cased-v1.2 (cached by test_environment.py).
All context windows are synthetic — no PHI.
"""

from __future__ import annotations

import pytest

from ods_phenocontext.models import BioBERTPredictor
from ods_phenocontext.pipeline import phenocontext_predict
from ods_phenocontext.rules import RuleClassifier
from ods_phenocontext.schema import NUM_LABELS, Instance

MODEL = "dmis-lab/biobert-base-cased-v1.2"


@pytest.fixture(scope="module")
def predictor() -> BioBERTPredictor:
    return BioBERTPredictor(model_path=MODEL)


def _inst(context: str = "Patient has [ENT] fever [/ENT].", entity: str = "fever") -> Instance:
    return Instance(
        instance_id="t-001",
        note_id="n-001",
        entity_text=entity,
        context_window=context,
        split="val",
    )


# ---------------------------------------------------------------------------
# predict_proba
# ---------------------------------------------------------------------------


class TestPredictProba:
    def test_returns_list_of_correct_length(self, predictor: BioBERTPredictor):
        probs = predictor.predict_proba(_inst())
        assert len(probs) == NUM_LABELS

    def test_probs_in_zero_one_range(self, predictor: BioBERTPredictor):
        probs = predictor.predict_proba(_inst())
        assert all(0.0 <= p <= 1.0 for p in probs)

    def test_returns_plain_list_not_tensor(self, predictor: BioBERTPredictor):
        probs = predictor.predict_proba(_inst())
        assert isinstance(probs, list)
        assert all(isinstance(p, float) for p in probs)

    def test_different_inputs_produce_different_probs(self, predictor: BioBERTPredictor):
        probs_a = predictor.predict_proba(_inst("Patient has [ENT] fever [/ENT]."))
        probs_b = predictor.predict_proba(_inst("No [ENT] fever [/ENT] noted."))
        assert probs_a != probs_b

    def test_long_input_truncated_without_error(self, predictor: BioBERTPredictor):
        long_text = "word " * 200 + "[ENT] fever [/ENT]" + " word" * 100
        probs = predictor.predict_proba(_inst(long_text))
        assert len(probs) == NUM_LABELS

    def test_entity_span_pooling_used(self, predictor: BioBERTPredictor):
        # With [ENT]/[/ENT] markers the model should use entity-span pooling;
        # verify token IDs are set (smoke check that setup was correct)
        assert predictor.model._ent_token_id is not None
        assert predictor.model._end_token_id is not None
        # The special tokens should tokenize to single IDs, not subwords
        ent_ids = predictor.tokenizer.encode("[ENT]", add_special_tokens=False)
        assert len(ent_ids) == 1


# ---------------------------------------------------------------------------
# predict
# ---------------------------------------------------------------------------


class TestPredict:
    def test_returns_binary_labels(self, predictor: BioBERTPredictor):
        labels = predictor.predict(_inst(), thresholds=[0.5] * NUM_LABELS)
        assert all(v in (0, 1) for v in labels)
        assert len(labels) == NUM_LABELS

    def test_threshold_zero_gives_all_ones(self, predictor: BioBERTPredictor):
        labels = predictor.predict(_inst(), thresholds=[0.0] * NUM_LABELS)
        assert labels == [1] * NUM_LABELS

    def test_threshold_one_gives_all_zeros(self, predictor: BioBERTPredictor):
        labels = predictor.predict(_inst(), thresholds=[1.0] * NUM_LABELS)
        assert labels == [0] * NUM_LABELS


# ---------------------------------------------------------------------------
# Pipeline integration: abstain case routed through real BioBERTPredictor
# ---------------------------------------------------------------------------


class TestPipelineIntegration:
    def test_abstain_instance_routed_to_biobert(self, predictor: BioBERTPredictor):
        # "can" triggers llm_review → rules abstain → BioBERT fallback
        instance = _inst("[ENT] Diabetes [/ENT] can cause neuropathy.", entity="Diabetes")
        result = phenocontext_predict(
            instance=instance,
            rules_model=RuleClassifier(),
            biobert_model=predictor,
            thresholds=[0.5] * NUM_LABELS,
        )
        assert result["source"] == "biobert"
        assert len(result["labels"]) == NUM_LABELS
        assert len(result["probs"]) == NUM_LABELS
        assert all(v in (0, 1) for v in result["labels"])
        assert all(0.0 <= p <= 1.0 for p in result["probs"])

    def test_confident_instance_not_routed_to_biobert(self, predictor: BioBERTPredictor):
        instance = _inst("No [ENT] fever [/ENT] noted.", entity="fever")
        result = phenocontext_predict(
            instance=instance,
            rules_model=RuleClassifier(),
            biobert_model=predictor,
            thresholds=[0.5] * NUM_LABELS,
        )
        assert result["source"] == "rules"
