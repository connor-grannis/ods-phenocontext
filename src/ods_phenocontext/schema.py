"""
Core data objects for PhenoContext.

These dataclasses are the shared currency across every pipeline stage:
labeling, rule execution, BioBERT inference, teacher annotation, synthetic
generation, and training.  Keep this file free of heavy dependencies so it
can be imported anywhere — including scripts that don't need torch or HF.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Label ontology (index → name mapping)
# ---------------------------------------------------------------------------

# Multi-hot vector index assignments.  Order is fixed; do not reorder.
LABEL_NAMES: list[str] = [
    "confirmed",  # 0 — phenotype affirmed for the patient
    "negated",  # 1 — explicitly negated
    "associated_with_someone_else",  # 2 — attributed to a non-patient experiencer
    "other_non_patient",  # 3 — hypothetical / historical / screening / uncertain
]
NUM_LABELS: int = len(LABEL_NAMES)


# ---------------------------------------------------------------------------
# Instance — primary data object
# ---------------------------------------------------------------------------


@dataclass
class Instance:
    """
    One phenotype mention in context, carrying labels and predictions from
    every pipeline stage that has touched it.

    Fields are populated incrementally: a freshly ingested note has only
    the identity + text fields; downstream stages fill in the rest.
    The split field controls which instances enter training vs. evaluation.
    """

    # --- Identity ---
    instance_id: str  # Unique across the project lifetime
    note_id: str  # Source note (used for group-aware splitting)
    entity_text: str  # Raw mention text from upstream NER
    context_window: str  # Text window around the mention (no PHI outside this)
    split: str  # "train" | "val" | "test" | "production"

    # --- Gold labels (primary truth) ---
    # Set once during annotation; never overwritten by teacher or model outputs.
    gold_labels: list[int] | None = None

    # --- Rule system outputs ---
    rule_labels: list[int] | None = None
    rule_probs: list[float] | None = None  # Per-label confidence scores
    rule_ids: list[str] | None = None  # Which rules fired (for audit)
    rule_abstained: bool = False  # True → routed to BioBERT

    # --- BioBERT outputs ---
    biobert_probs: list[float] | None = None  # Raw sigmoid probabilities
    biobert_labels: list[int] | None = None  # Post-threshold binary labels

    # --- Teacher committee outputs (dev/refresh only) ---
    # Keyed by teacher role name; values are raw TeacherOutput dicts.
    teacher_outputs: dict[str, dict] = field(default_factory=dict)
    aggregated_teacher_labels: list[int] | None = None
    # Proportion of label positions where teachers disagree; used for routing.
    disagreement_score: float | None = None

    # --- Provenance ---
    source_type: str = "original"  # "original" | "synthetic" | "silver"
    parent_instance_id: str | None = None  # Set for synthetic children


# ---------------------------------------------------------------------------
# SyntheticAudit — provenance record for every generated example
# ---------------------------------------------------------------------------


@dataclass
class SyntheticAudit:
    """
    Tracks how a synthetic Instance was generated and whether it passed
    quality validation.  Required before a synthetic example can enter
    training (see CLAUDE.md design constraints §6).
    """

    synthetic_id: str  # Matches Instance.instance_id
    parent_instance_id: str
    target_labels: list[int]  # Labels the generation prompt aimed for
    generation_prompt_version: str
    teacher_model: str  # Model that produced the synthetic text
    rationale: str  # Teacher's explanation for the generation
    # Keyed by check name (e.g. "label_preserved", "embedding_similarity",
    # "lexical_diversity", "dedup", "manual_review"); value is pass/fail.
    validation_checks: dict[str, bool] = field(default_factory=dict)

    @property
    def approved(self) -> bool:
        """True only if every validation check passed."""
        return bool(self.validation_checks) and all(self.validation_checks.values())


# ---------------------------------------------------------------------------
# TrainingManifest — required artifact for every retraining iteration
# ---------------------------------------------------------------------------


@dataclass
class TrainingManifest:
    """
    Complete provenance snapshot for one training iteration.  Must be
    serialized (e.g. to JSON) and stored in audits/training_manifests/
    before the resulting checkpoint is used in any evaluation.
    """

    iteration: int
    base_model: str  # HF model ID used as fine-tune base
    rule_version: str
    teacher_models: list[str]
    # Aggregation weights keyed by teacher role name (must sum to ~1.0)
    teacher_weights: dict[str, float]
    prompt_version: str
    # Training data counts by source
    num_original: int
    num_silver: int
    num_synthetic: int
    synthetic_ratio: float  # num_synthetic / num_original
    label_distribution: dict[str, int]  # label name → count of positive examples
    # Per-label decision thresholds (keyed by label name)
    thresholds: dict[str, float]
    # Validation metrics (e.g. "f1_confirmed", "auc_negated", "macro_f1")
    validation_metrics: dict[str, float]
