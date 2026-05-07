"""
Rule-based phenotype context classifier conforming to the RulesModel protocol.

Adapts the binary (confirmed/not-confirmed) logic to multi-hot 4-label output:
  [confirmed, negated, associated_with_someone_else, other_non_patient]

Category → label mapping:
  negation / implicit_negation → negated (index 1)
  family / other_person        → associated_with_someone_else (index 2)
  hypothetical                 → other_non_patient (index 3)
  llm_review                   → abstain (route to BioBERT)
  no triggers                  → confirmed (index 0)
"""

from __future__ import annotations

from typing import Any

from ods_phenocontext.schema import NUM_LABELS, Instance

from .patterns import CONTEXT_RULES

# Category → label index mapping
_CATEGORY_TO_LABEL: dict[str, int] = {
    "negation": 1,
    "implicit_negation": 1,
    "family": 2,
    "other_person": 2,
    "hypothetical": 3,
}

# Confidence scores per category (conservative heuristic defaults per PROJECT_OVERVIEW)
_CATEGORY_CONFIDENCE: dict[str, float] = {
    "negation": 0.95,
    "implicit_negation": 0.90,
    "family": 0.95,
    "other_person": 0.90,
    "hypothetical": 0.85,
}

# Token window size for context extraction
_WINDOW = 15
_ENTITY_START = "[ENT]"
_ENTITY_END = "[/ENT]"


def _extract_windows(instance: Instance) -> tuple[str, str, str]:
    """Extract (pre_context, entity_text, post_context) from an Instance.

    Prefer explicit [ENT]...[/ENT] marker positions when present.
    Falls back to using entity_text to locate the mention within context_window.
    """
    ctx = instance.context_window
    entity = instance.entity_text

    marker_start = ctx.find(_ENTITY_START)
    marker_end = ctx.find(_ENTITY_END)

    if marker_start >= 0 and marker_end >= 0 and marker_start < marker_end:
        entity_start = marker_start + len(_ENTITY_START)
        pre = ctx[:marker_start]
        entity = ctx[entity_start:marker_end].strip()
        post = ctx[marker_end + len(_ENTITY_END) :]
    else:
        # Try to find entity in context window
        idx = ctx.lower().find(entity.lower())
        if idx >= 0:
            pre = ctx[:idx]
            post = ctx[idx + len(entity) :]
        else:
            # Fallback: split at midpoint
            mid = len(ctx) // 2
            pre = ctx[:mid]
            post = ctx[mid:]

    # Trim to local window
    pre_tokens = pre.split()
    post_tokens = post.split()
    pre_window = " ".join(pre_tokens[-_WINDOW:])
    post_window = " ".join(post_tokens[:_WINDOW])
    return pre_window, entity, post_window


class RuleClassifier:
    """Rule-based phenotype context classifier.

    Conforms to the RulesModel protocol defined in pipeline.py.
    """

    def __call__(self, instance: Instance) -> dict[str, Any]:
        pre, entity, post = _extract_windows(instance)
        full = f"{pre} {post}"

        fired_categories: list[str] = []
        fired_rule_ids: list[str] = []
        seen: set[str] = set()

        for rule in CONTEXT_RULES:
            scope: str = rule["scope"]
            category: str = rule["category"]
            target = {"pre": pre, "post": post, "any": full, "entity": entity}[scope]

            for pat in rule["patterns"]:
                if pat.search(target):
                    if category not in seen:
                        seen.add(category)
                        fired_categories.append(category)
                        fired_rule_ids.append(f"{category}-{scope}-{pat.pattern[:30]}")
                    break

        # If llm_review fired, abstain
        if "llm_review" in seen:
            return {
                "abstained": True,
                "labels": [0] * NUM_LABELS,
                "probs": [0.0] * NUM_LABELS,
                "rule_ids": fired_rule_ids,
            }

        # Build multi-hot labels and confidence scores
        labels = [0] * NUM_LABELS
        probs = [0.0] * NUM_LABELS

        for cat in fired_categories:
            label_idx = _CATEGORY_TO_LABEL.get(cat)
            if label_idx is not None:
                labels[label_idx] = 1
                probs[label_idx] = max(probs[label_idx], _CATEGORY_CONFIDENCE[cat])

        # No triggers → confirmed
        if not fired_categories:
            labels[0] = 1
            probs[0] = 0.95

        return {
            "abstained": False,
            "labels": labels,
            "probs": probs,
            "rule_ids": fired_rule_ids,
        }
