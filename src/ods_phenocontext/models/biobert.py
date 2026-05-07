"""
BioBERT multi-label classifier with entity-span pooling.

Architecture:
  1. Tokenize context window (with [ENT]/[/ENT] as special tokens).
  2. Run encoder forward pass.
  3. Locate [ENT] and [/ENT] token positions in the sequence.
  4. Mean-pool hidden states between those markers → entity representation.
  5. Linear head maps entity representation → num_labels logits.
  6. Apply sigmoid for probabilities; use BCEWithLogitsLoss during training.

Entity-span pooling focuses the classifier on how the model contextualizes
the specific mention rather than a generic [CLS] sentence summary.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoModel

from ods_phenocontext.schema import NUM_LABELS

# Special tokens marking entity spans in context windows
ENTITY_START_TOKEN = "[ENT]"
ENTITY_END_TOKEN = "[/ENT]"
SPECIAL_TOKENS = [ENTITY_START_TOKEN, ENTITY_END_TOKEN]


class BioBERTMultiLabel(nn.Module):
    """
    Encoder + entity-span pooling + linear head for multi-label
    phenotype context classification.

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
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size: int = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_labels)

        # Token IDs for [ENT]/[/ENT] — set after resize_token_embeddings
        self._ent_token_id: int | None = None
        self._end_token_id: int | None = None

    def set_entity_token_ids(self, ent_id: int, end_id: int) -> None:
        """Store entity marker token IDs after tokenizer special-token addition."""
        self._ent_token_id = ent_id
        self._end_token_id = end_id

    def resize_token_embeddings(self, new_num_tokens: int) -> None:
        """Resize encoder embeddings (called after adding special tokens)."""
        self.encoder.resize_token_embeddings(new_num_tokens)

    def _entity_span_pool(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Mean-pool hidden states between [ENT] and [/ENT] markers.

        Falls back to [CLS] (position 0) if markers are not found.

        Args:
            hidden_states: (batch, seq_len, hidden_size)
            input_ids:     (batch, seq_len)

        Returns:
            (batch, hidden_size)
        """
        batch_size = input_ids.size(0)
        pooled = []

        for i in range(batch_size):
            seq_ids = input_ids[i]

            start_positions = (seq_ids == self._ent_token_id).nonzero(as_tuple=False)
            end_positions = (seq_ids == self._end_token_id).nonzero(as_tuple=False)

            if start_positions.numel() > 0 and end_positions.numel() > 0:
                start = start_positions[0].item()
                end = end_positions[0].item()

                if end > start + 1:
                    # Mean-pool tokens strictly between [ENT] and [/ENT]
                    entity_hidden = hidden_states[i, start + 1 : end]
                    pooled.append(entity_hidden.mean(dim=0))
                else:
                    # Adjacent markers with nothing between — use [ENT] position
                    pooled.append(hidden_states[i, start])
            else:
                # Markers not found — fall back to [CLS]
                pooled.append(hidden_states[i, 0])

        return torch.stack(pooled)

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
        """
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

        if self._ent_token_id is not None and self._end_token_id is not None:
            entity_repr = self._entity_span_pool(outputs.last_hidden_state, input_ids)
        else:
            # Entity token IDs not set — fall back to [CLS]
            entity_repr = outputs.last_hidden_state[:, 0, :]

        return self.classifier(self.dropout(entity_repr))

    def predict_proba(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor | None = None,
    ) -> list[float]:
        """
        Run a forward pass and return per-label sigmoid probabilities
        as a plain Python list.
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(input_ids, attention_mask, token_type_ids)
        return torch.sigmoid(logits).squeeze(0).tolist()
