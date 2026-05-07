"""
BioBERT multi-label classifier stub.

Wraps a HuggingFace encoder with a linear classification head.
Four sigmoid outputs — one per label in LABEL_NAMES — trained with
BCEWithLogitsLoss.  Per-label thresholds are set on the validation set
after training (not assumed to be 0.5).

This stub is functional enough for smoke tests and pipeline wiring.
The full training loop, threshold tuning, and checkpoint management
are implemented in the modeling roadmap training phase.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoModel

from ods_phenocontext.schema import NUM_LABELS


class BioBERTMultiLabel(nn.Module):
    """
    Encoder + linear head for multi-label phenotype context classification.

    Args:
        num_labels:   Number of output labels (default: NUM_LABELS = 4).
        model_name:   HF model ID to use as the encoder backbone.
        dropout:      Dropout rate applied before the classification head.
    """

    def __init__(
        self,
        num_labels: int = NUM_LABELS,
        model_name: str = "dmis-lab/biobert-base-cased-v1.2",
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_labels = num_labels
        # Load the encoder; weights are random here (stub) — real usage loads
        # a pretrained checkpoint via from_pretrained or load_state_dict.
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size: int = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        # Linear head: [CLS] hidden state → num_labels logits
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            input_ids:      (batch, seq_len)
            attention_mask: (batch, seq_len)
            token_type_ids: (batch, seq_len) — optional for RoBERTa-style encoders

        Returns:
            logits: (batch, num_labels) — raw logits (no sigmoid).
            Apply sigmoid for probabilities; use BCEWithLogitsLoss during training.
        """
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        # [CLS] token representation
        cls_repr = outputs.last_hidden_state[:, 0, :]
        return self.classifier(self.dropout(cls_repr))

    def predict_proba(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor | None = None,
    ) -> list[float]:
        """
        Convenience method: run a forward pass and return per-label sigmoid
        probabilities as a plain Python list.  Used by the pipeline.
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(input_ids, attention_mask, token_type_ids)
        return torch.sigmoid(logits).squeeze(0).tolist()
