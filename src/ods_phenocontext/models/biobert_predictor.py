"""
BioBERTPredictor: Instance-level adapter for BioBERTMultiLabel.

Handles tokenization (with [ENT]/[/ENT] as special tokens) and delegates to
BioBERTMultiLabel's entity-span pooling forward pass.

Does not re-apply preprocessing — Instance.context_window is canonical and
expected to contain [ENT]...[/ENT] markup around the phenotype mention.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from ods_phenocontext.models.biobert import (
    ENTITY_END_TOKEN,
    ENTITY_START_TOKEN,
    SPECIAL_TOKENS,
    BioBERTMultiLabel,
)
from ods_phenocontext.schema import NUM_LABELS, Instance


class BioBERTPredictor:
    """Tokenizing adapter that wraps BioBERTMultiLabel for Instance-level inference.

    Conforms to the BioBERTModel Protocol defined in pipeline.py.

    On construction:
      1. Loads the tokenizer and adds [ENT]/[/ENT] as special tokens.
      2. Resizes model embeddings to match the expanded vocab.
      3. Registers entity marker token IDs with the model for span pooling.

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
        self.tokenizer.add_special_tokens({"additional_special_tokens": SPECIAL_TOKENS})

        self.model = BioBERTMultiLabel(model_name=model_path)
        self.model.resize_token_embeddings(len(self.tokenizer))

        ent_id = self.tokenizer.convert_tokens_to_ids(ENTITY_START_TOKEN)
        end_id = self.tokenizer.convert_tokens_to_ids(ENTITY_END_TOKEN)
        self.model.set_entity_token_ids(ent_id, end_id)

    @classmethod
    def from_checkpoint(cls, checkpoint_dir: str | Path) -> BioBERTPredictor:
        """Load a fine-tuned model from a train_biobert.py checkpoint.

        Expects checkpoint_dir/model/ containing:
          - model_config.json (model_name, num_labels, dropout, max_length, token IDs)
          - biobert_multilabel.pt (full BioBERTMultiLabel state_dict)
          - tokenizer files (from save_pretrained)

        Args:
            checkpoint_dir: Path to the training output directory
                            (the --out argument passed to train_biobert.py).
        """
        from transformers import AutoTokenizer

        checkpoint_dir = Path(checkpoint_dir)
        model_dir = checkpoint_dir / "model"

        config = json.loads((model_dir / "model_config.json").read_text())

        instance = object.__new__(cls)
        instance.model_path = config["model_name"]
        instance.max_length = config["max_length"]

        instance.tokenizer = AutoTokenizer.from_pretrained(model_dir)

        instance.model = BioBERTMultiLabel(
            num_labels=config.get("num_labels", NUM_LABELS),
            model_name=config["model_name"],
            dropout=config.get("dropout", 0.1),
        )
        instance.model.resize_token_embeddings(len(instance.tokenizer))
        instance.model.set_entity_token_ids(config["ent_token_id"], config["end_token_id"])

        state_dict = torch.load(
            model_dir / "biobert_multilabel.pt",
            map_location="cpu",
            weights_only=True,
        )
        instance.model.load_state_dict(state_dict)
        instance.model.eval()

        return instance

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
