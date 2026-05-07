# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Status

This is a design-stage project. The two source-of-truth planning documents are:
- `PROJECT_OVERVIEW.md` — architecture, label ontology, data objects, design patterns
- `plan.md` — 14-phase development roadmap with milestones and decision criteria

No code exists yet. When implementing, follow these documents carefully — they contain deliberate design choices that should not be overridden without a clear reason.

## What This System Does

**PhenoContext** classifies the context of phenotype mentions in clinical notes. Given a phenotype mention (from an upstream NER system) + a context window, it assigns multi-hot labels from a 4-class ontology:

- `confirmed` — phenotype affirmed for the patient
- `negated` — explicitly negated
- `associated_with_someone_else` — attributed to a non-patient experiencer
- `other_non_patient` — umbrella for uncertainty, hypothetical, historical, screening, etc.

## Core Architecture

The deployed system is a **rules-first, BioBERT-fallback pipeline** — not an LLM in production:

1. Run rule system → if confident, return rule labels
2. If rules abstain → run BioBERT multi-label classifier
3. Apply per-label thresholds to BioBERT sigmoid outputs
4. Return labels, probabilities, and prediction source

LLMs (teacher committee) are used only during development and refresh cycles, not at inference time.

### BioBERT Classifier

- Single encoder model with 4 label outputs (not 4 separate classifiers)
- `sigmoid` activation + `BCEWithLogitsLoss`
- Per-label threshold tuning on validation set — do not assume 0.5
- Optional per-label positive weights and sample weights for data mixing

### Rule System

- Rules produce label-specific confidence scores (not binary decisions)
- Rules can abstain; abstention routes to BioBERT
- Rule confidence is empirically estimated by rule family, not assumed
- Rules are versioned; maintain a rule manifest (ID, definition, target label, confidence, version)

### Teacher Committee (Development Only)

- 3–4 LLMs with intentional role diversity: generalist, precision-biased, recall-biased, optional mechanistic/rule-aware
- Structured output: `{"labels": [...], "rationale": "...", "evidence_spans": [...], "confidence_bin": "high|medium|low"}`
- Aggregation uses heuristic weights (generalist: 0.4, precision: 0.25, recall: 0.2, mechanistic: 0.15); can evolve to learned per-label weights
- Run only on targeted subsets (rule/BioBERT discordant cases, low-confidence BioBERT, rare labels, FP/FN slices) — not full corpus

## Key Data Object

```python
@dataclass
class Instance:
    instance_id: str
    note_id: str
    entity_text: str
    context_window: str
    split: str                              # train / val / test / production
    gold_labels: Optional[List[int]] = None
    rule_labels: Optional[List[int]] = None
    rule_probs: Optional[List[float]] = None
    rule_ids: Optional[List[str]] = None
    rule_abstained: bool = False
    biobert_probs: Optional[List[float]] = None
    biobert_labels: Optional[List[int]] = None
    teacher_outputs: Dict[str, dict] = field(default_factory=dict)
    aggregated_teacher_labels: Optional[List[int]] = None
    disagreement_score: Optional[float] = None
    source_type: str = "original"           # original / synthetic / silver
    parent_instance_id: Optional[str] = None
```

## Non-Negotiable Design Constraints

These are explicit decisions from the design documents — do not change without justification:

1. **Gold labels are primary truth.** Teacher labels are development signals, not gold replacements.
2. **Fixed validation set.** Keep frozen throughout all development. Do not tune thresholds then evaluate on the same data in the same pass.
3. **Reinitialize from base checkpoint** each retraining iteration (not continual fine-tuning), unless warm-start is explicitly justified.
4. **Cap synthetic augmentation** at 20–40% of original gold training size. Never let synthetic data dominate training.
5. **Do not build a learned router first.** The rules-first + abstention → BioBERT pattern is the default. Only add a stacked ensemble or learned router if validation metrics justify it.
6. **Validate synthetic batches** before adding to training: label preservation by 2/3 teachers, embedding similarity threshold, lexical diversity, deduplication, manual spot review.

## Reproducibility: Required Artifacts Per Iteration

Every iteration must produce a `TrainingManifest` capturing: iteration number, base model checkpoint, rule version, teacher models + weights + prompt versions, training data counts by source and label, per-label thresholds, validation metrics.

Maintain across the project lifetime:
- Dataset split manifest (IDs, split assignment, date, exclusion criteria)
- Rule manifest
- Prompt registry (text, version, model, intended teacher role)
- Synthetic data audit (parent ID, target labels, validation checks, approval status)
- Experiment registry + stage comparison reports
- Decision log

## Suggested Directory Layout

```
project/
├── data/{gold,silver,synthetic,processed}/
├── prompts/                    # versioned teacher and generation prompts
├── rules/                      # versioned YAML rule files
├── models/                     # checkpoints per iteration
├── audits/{teacher_outputs,synthetic_provenance,training_manifests}/
├── docs/                       # label_ontology, annotation_guidelines, manifests, logs
├── experiments/                # per-stage result directories
├── src/                        # run_rules, run_teachers, aggregate_teachers,
│                               # generate_synthetics, validate_synthetics,
│                               # train_biobert, evaluate, pipeline
└── configs/                    # per-iteration YAML configs
```

## Clinical Data Constraints

This system processes clinical notes containing PHI. All data handling must comply with HIPAA. Do not log raw note text or entity spans to stdout or files outside controlled data directories.
