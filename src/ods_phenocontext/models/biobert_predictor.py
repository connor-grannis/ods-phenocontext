"""
BioBERTPredictor: Instance-level adapter for BioBERTMultiLabel.

BioBERTMultiLabel.predict_proba takes raw tensors; this adapter handles
tokenization so the pipeline can pass an Instance directly.  Closes the
gap described in the baselines plan (M4).

Does not re-apply preprocessing — Instance.context_window is canonical.
"""

from __future__ import annotations

from ods_phenocontext.models.biobert import BioBERTMultiLabel
from ods_phenocontext.schema import Instance


class BioBERTPredictor:
    """Tokenizing adapter that wraps BioBERTMultiLabel for Instance-level inference.

    Conforms to the BioBERTModel Protocol defined in pipeline.py.

    Args:
        model_path:  HF model ID or local checkpoint directory.
        max_length:  Tokenizer max sequence length (truncates if exceeded).
    """

    def __init__(
        self,
        model_path: str = "dmis-lab/biobert-base-cased-v1.2",
        max_length: int = 256,
    ) -> None:
        from transformers import AutoTokenizer

        self.model_path = model_path
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = BioBERTMultiLabel(model_name=model_path)

    def predict_proba(self, instance: Instance) -> list[float]:
        """Tokenize instance.context_window and return per-label sigmoid probabilities."""
        inputs = self.tokenizer(
            instance.context_window,
            return_tensors="pt",
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
        )
        return self.model.predict_proba(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            token_type_ids=inputs.get("token_type_ids"),
        )

    def predict(self, instance: Instance, thresholds: list[float]) -> list[int]:
        """Return binary labels by thresholding predict_proba output."""
        probs = self.predict_proba(instance)
        return [int(p >= t) for p, t in zip(probs, thresholds, strict=True)]
