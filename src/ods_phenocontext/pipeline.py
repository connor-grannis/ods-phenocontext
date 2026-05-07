"""
Production inference pipeline for PhenoContext.

The pipeline is rules-first with BioBERT fallback:
  1. Run the rule system on an Instance.
  2. If rules are confident (did not abstain), return rule labels.
  3. If rules abstain, run BioBERT and apply per-label thresholds.

Both the rules model and BioBERT model are passed in as arguments so the
pipeline is easy to test with fakes and easy to swap during development.
"""

from __future__ import annotations

from typing import Any, Protocol

from ods_phenocontext.schema import Instance

# ---------------------------------------------------------------------------
# Protocols — define the interface each model component must satisfy
# ---------------------------------------------------------------------------


class RulesModel(Protocol):
    """Interface for the rule system."""

    def __call__(self, instance: Instance) -> dict[str, Any]:
        """
        Returns a dict with keys:
          abstained: bool
          labels:    list[int]   (multi-hot, only meaningful if not abstained)
          probs:     list[float] (per-label confidence scores)
          rule_ids:  list[str]   (which rules fired)
        """
        ...


class BioBERTModel(Protocol):
    """Interface for the BioBERT classifier."""

    def predict_proba(self, instance: Instance) -> list[float]:
        """Return per-label sigmoid probabilities."""
        ...


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


def phenocontext_predict(
    instance: Instance,
    rules_model: RulesModel,
    biobert_model: BioBERTModel,
    thresholds: list[float],
) -> dict[str, Any]:
    """
    Run the rules-first, BioBERT-fallback inference pipeline.

    Args:
        instance:      The Instance to classify.
        rules_model:   Rule system; called first.
        biobert_model: BioBERT classifier; called only when rules abstain.
        thresholds:    Per-label decision thresholds (one float per label).
                       Tuned on the validation set — do not assume 0.5.

    Returns:
        Dict with keys:
          source:   "rules" | "biobert"
          labels:   list[int]   — final multi-hot predictions
          probs:    list[float] — probabilities from the active source
          rule_ids: list[str]   — present only when source == "rules"
    """
    rule_output = rules_model(instance)

    if not rule_output["abstained"]:
        # Rule system is confident — return its answer directly.
        return {
            "source": "rules",
            "labels": rule_output["labels"],
            "probs": rule_output["probs"],
            "rule_ids": rule_output["rule_ids"],
        }

    # Rules abstained — fall back to BioBERT.
    probs = biobert_model.predict_proba(instance)
    labels = [int(p >= t) for p, t in zip(probs, thresholds, strict=True)]

    return {
        "source": "biobert",
        "labels": labels,
        "probs": probs,
    }
