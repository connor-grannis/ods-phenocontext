# PhenoContext Project Overview and Development Plan

## Executive Summary

PhenoContext is a phenotype-context classification system for clinical text. It starts from phenotype mentions already identified by a phenotype named entity recognition (NER) step, then assigns one or more context labels to each mention based on the surrounding clinical text.

The project is designed as a two-stage production pipeline:

1. A high-precision rule system handles easy, low-cost cases first.
2. A BioBERT-style multi-label classifier handles rule-abstained or harder cases.

LLM teachers are used during development and periodic refresh cycles, not as the primary deployed production model. Their purpose is to analyze errors, generate silver labels for uncertain cases, propose rule refinements, and create tightly controlled hard-case synthetic examples. The deployed classifier remains a BioBERT-style encoder with a sigmoid multi-label head.

## What the Project Is

PhenoContext is a clinical NLP pipeline that determines the context of phenotype mentions in clinical notes. Each input instance represents a single phenotype mention plus a context window. For the first iteration, the output is a four-label multi-hot vector. Later iterations can expand the broad "other non-patient" label into more granular categories after the smaller problem is reproducibly evaluated.

The project combines several complementary techniques:

- Rule-based weak supervision for cheap, precise, interpretable decisions.
- BioBERT multi-label classification for learned generalization.
- Teacher-student distillation using LLM committees during development.
- Active learning to identify uncertain, rare, or discrepant examples.
- Controlled synthetic augmentation for hard clinical edge cases.
- Periodic model refresh with reproducible training manifests and fixed validation controls.

The system is not primarily a generative-AI-in-production design. Generative models help improve the data, rules, and model during development; they do not replace the deployed BioBERT classifier.

## Project Goals

### Primary Product Goals

- Build a reliable clinical phenotype-context classifier that operates on phenotype mentions and local context windows.
- Prioritize high precision for easy cases through a rules-first production path.
- Improve recall and robustness on hard cases through a BioBERT multi-label student model.
- Support four first-iteration context labels as a single multi-label prediction problem rather than four independent classifiers.
- Preserve a clean expansion path from the broad first-iteration label set to a more granular future ontology.
- Preserve a clear audit trail for labels, teacher outputs, synthetic examples, training data, and evaluation results.

### Modeling Goals

- Train one BioBERT-style encoder with a sigmoid output head over four first-iteration labels.
- Use `BCEWithLogitsLoss`, with optional per-label positive weights for imbalance.
- Tune thresholds per label on the validation set instead of using a uniform `0.5` threshold.
- Use the original gold labels as the primary source of truth.
- Use teacher labels as auxiliary development signals, not replacements for gold labels.
- Improve rare-label and hard-slice performance without allowing synthetic data to dominate training.
- Keep documentation detailed enough that each modeling decision can be reproduced, justified, and described in a manuscript.

### Operational Goals

- Keep the production pipeline fast, interpretable, and reproducible.
- Log all versions that affect predictions: data splits, rules, model checkpoints, tokenizer, prompts, teacher models, aggregation method, thresholds, and metrics.
- Enable periodic refreshes using new reviewed cases, teacher-assisted silver labels, and approved synthetic examples.
- Promote refreshed models only when they improve validation metrics and do not create unacceptable false-positive growth.

## Inputs and Outputs

### First-Iteration Label Scope

The first iteration should deliberately narrow the label ontology to four context attributes:

- `confirmed`: the phenotype is affirmed for the patient.
- `negated`: the phenotype is explicitly negated for the patient.
- `associated_with_someone_else`: the phenotype is associated with a non-patient experiencer, such as a family member.
- `other_non_patient`: the phenotype is not a confirmed or negated patient phenotype and does not fit the narrower "associated with someone else" definition.

The `other_non_patient` label is intentionally broad in the first iteration. It should absorb context categories that will later be split into more specific labels, such as uncertainty, hypothetical mentions, screening indications, historical discussion that is not patient-confirmed under the project definition, or other non-patient uses. During annotation and error analysis, store a secondary note or subtype when available, but do not train the first-iteration model to predict those subtypes.

This scoped ontology reduces annotation ambiguity, makes early performance easier to interpret, and creates a stronger baseline for future label expansion.

### Inputs

The project assumes the following already exist:

- A phenotype NER step that identifies phenotype mentions in notes.
- An initial rule-based context classifier.
- An initial BioBERT multi-label classifier.
- An initial teacher prompt for the LLM committee.
- Gold train, validation, and test splits.

Each model instance should represent one phenotype mention and its surrounding context. The row-level instance format should capture both model inputs and audit metadata.

```python
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class Instance:
    instance_id: str
    note_id: str
    entity_text: str
    context_window: str
    split: str  # train / val / test / production
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
    source_type: str = "original"  # original / synthetic / silver
    parent_instance_id: Optional[str] = None
```

### Outputs

For each phenotype mention, PhenoContext returns:

- The predicted multi-label context vector.
- The source of the prediction: rules or BioBERT.
- A probability or confidence score vector for the selected prediction source.
- Optional audit metadata for development and monitoring workflows.

## Core Architecture

### Production Inference Path

The production path should remain simple and controlled:

1. Run the rule system.
2. If the rule system produces a confident non-abstained answer, accept the rule labels.
3. If the rule system abstains, run the BioBERT multi-label classifier.
4. Apply per-label thresholds to BioBERT probabilities.
5. Return labels, probabilities, and prediction source.

```python
def phenocontext_predict(instance, rules_model, biobert_model, thresholds):
    rule_output = rules_model(instance)

    if not rule_output["abstained"]:
        return {
            "source": "rules",
            "labels": rule_output["labels"],
            "probs": rule_output["probs"],
            "rule_ids": rule_output["rule_ids"],
        }

    probs = biobert_model.predict_proba(instance)
    labels = [int(p >= t) for p, t in zip(probs, thresholds)]

    return {
        "source": "biobert",
        "labels": labels,
        "probs": probs,
    }
```

This design provides speed, high precision on easy cases, and learned generalization on harder cases.

### Rule Probability and Confidence Scores

Rules should return probability-like confidence scores when they assign labels. These scores are useful for auditability, downstream comparison with BioBERT probabilities, confidence gating, and stage-by-stage evaluation.

However, first-iteration rule scores should be described carefully. Unless they are calibrated against validation data, they are confidence scores rather than fully calibrated probabilities.

The preferred approach is rule-specific scoring:

- Assign each rule or rule family a score based on empirical precision on the training or validation development data.
- Store the exact `rule_id`, `rule_version`, and score source used for each decision.
- Use conservative defaults for new or low-support rules until enough evidence exists.
- Re-estimate scores when rules change, rather than treating scores as permanent constants.

A static score can be used as a temporary bootstrap, for example assigning all accepted high-precision rules `0.95`. This is simple, but it hides differences between rules. The better manuscript-defensible version is to associate different rules with different confidence scores once enough labeled examples exist.

Recommended first iteration:

1. Start with conservative rule-family scores, such as `0.90` to `0.98` for high-precision rules.
2. After the first baseline audit, estimate empirical positive predictive value by rule family and label.
3. Keep rule scores fixed within each experiment stage.
4. Document every rule score in the rule manifest.
5. Clearly report whether scores are heuristic confidence values or calibrated probabilities.

### Development and Refresh Path

The development path adds teacher models, error analysis, synthetic generation, and retraining. The LLM committee should be run on selected cases rather than the full corpus by default:

- Rule/BioBERT disagreements.
- Low-confidence BioBERT predictions.
- Rare-label positives.
- False-positive and false-negative slices.
- Drift slices from production monitoring.

The committee produces structured outputs:

```json
{
  "labels": [0, 1, 0, 0],
  "rationale": "...",
  "evidence_spans": ["..."],
  "confidence_bin": "high"
}
```

Teacher outputs are aggregated for error analysis, rule refinement, targeted augmentation, and optional silver-label candidate pools.

## Labeling and Model Design

PhenoContext should use one final multi-label model with four outputs in the first iteration. A separate model per label should only be introduced if one label becomes a clearly distinct task with different data, features, or lifecycle needs.

The BioBERT student should use:

- A pretrained BioBERT-compatible encoder.
- A dropout layer.
- A linear classification head from hidden size to four first-iteration labels.
- Sigmoid activation at inference.
- `BCEWithLogitsLoss` during training.
- Optional per-label positive weights for imbalance.
- Optional sample weights so synthetic or silver data contributes less than gold data.

Example model shape:

```python
import torch.nn as nn
from transformers import AutoModel


class BioBERTMultiLabel(nn.Module):
    def __init__(self, model_name: str, num_labels: int, dropout: float = 0.1):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls = outputs.last_hidden_state[:, 0]
        return self.classifier(self.dropout(cls))
```

## Teacher Committee Strategy

The teacher committee should be diverse in a controlled way. Each teacher should use the same ontology and label definitions, but may apply a different decision bias.

Recommended initial committee:

- Generalist teacher: balanced, optimized for overall correctness.
- Precision-biased teacher: conservative, labels only when evidence is explicit and unambiguous.
- Recall-biased teacher: permissive, catches borderline or implicit cases.

Optional fourth teacher:

- Mechanistic or rule-aware teacher: explicitly reasons about negation, scope, experiencer, temporality, uncertainty, and ambiguity.

The goal is not to make prompts arbitrarily different. The goal is to create meaningful disagreement signals that help identify true ambiguity, rule weaknesses, rare-label gaps, and synthetic augmentation opportunities.

### Teacher Aggregation

Do not rely only on unweighted majority vote. Start with simple heuristic weights, then learn per-label weights from validation performance once enough data exists.

Example starting weights:

```python
TEACHER_WEIGHTS = {
    "generalist": 0.4,
    "precision": 0.25,
    "recall": 0.2,
    "mechanistic": 0.15,
}
```

Aggregation can evolve through three stages:

1. Simple vote for the first baseline.
2. Weighted vote using known teacher roles.
3. Learned per-label meta-model or per-label teacher weights based on validation performance.

## Development Phases

### Phase 1: Freeze Gold Splits

Keep the original train, validation, and test splits fixed. The test set should remain untouched until final reporting. All threshold tuning and iteration decisions should use the validation set.

### Phase 2: Run Baselines

Run the initial rules and BioBERT model on the train and validation sets. Build an error table that includes gold labels, rule labels, rule abstention status, and BioBERT labels.

Analyze errors by:

- Label.
- Note type.
- Department.
- Context length.
- Overlap patterns.
- False positives.
- False negatives.

The result is the first discrepancy pool for teacher review and hard-case selection.

### Phase 3: Run Teacher Committee on Targeted Cases

Run the teacher committee on high-value subsets:

- Rule/BioBERT discordant cases.
- BioBERT low-confidence cases.
- Rare-label positives.
- False-positive and false-negative slices.
- Clinically plausible ambiguous cases.

Do not start by sending the full corpus through the committee unless there is a clear cost, privacy, and audit justification.

### Phase 4: Aggregate Teacher Outputs

Aggregate teacher labels and capture disagreement scores. Use the aggregation for development decisions, not automatic replacement of gold labels.

Important outputs include:

- Aggregated labels.
- Per-label teacher agreement.
- Teacher rationales.
- Evidence spans.
- Confidence bins.
- Disagreement score.

### Phase 5: Use Teachers for Error Analysis

Ask teachers to explain why rules or BioBERT failed. The useful outputs are:

- Rule refinement proposals.
- Scope and exclusion conditions.
- Note-type-specific adjustments.
- Synthetic seed candidates.
- Ambiguity flags.

The most valuable synthetic seeds are real, hard, clinically plausible cases whose failure mode is well understood.

### Phase 6: Generate Controlled Synthetic Variants

Generate a small number of variants per selected seed case, initially two or three. Synthetic examples should preserve:

- Phenotype mention type.
- Target context attribute.
- Ambiguity or error profile.
- Local clinical realism.

They may vary:

- Lexical wording.
- Syntax.
- Note style.
- Mention position.
- Trigger expression.

Each synthetic example must store provenance.

```python
@dataclass
class SyntheticAudit:
    synthetic_id: str
    parent_instance_id: str
    target_labels: List[int]
    generation_prompt_version: str
    teacher_model: str
    rationale: str
    validation_checks: Dict[str, bool]
```

### Phase 7: Validate Synthetic Data

Before synthetic data enters training, apply quality checks:

- Label preserved by at least two of three teachers.
- Intended rule profile preserved.
- BioBERT embedding similarity above a defined floor.
- Lexical overlap below a duplication threshold.
- Duplicate removal completed.
- Manual spot review completed for each batch.

Synthetic data should be treated as targeted augmentation, not as a substitute for real clinical data.

### Phase 8: Retrain BioBERT

Each iteration should train on:

- All original gold training examples.
- Approved synthetic hard-case examples.
- Optional newly reviewed gold examples.
- Optional lower-weight silver examples, only if clearly separated from gold.

The preferred retraining approach is to reinitialize from the original pretrained BioBERT checkpoint for each iteration. This improves reproducibility and prevents uncontrolled drift across repeated fine-tuning cycles.

Warm-starting from the previous best checkpoint is acceptable only with strict controls:

- Small learning rate.
- Early stopping.
- Limited additional epochs.
- Same fixed validation set.
- Clear manifest records.

### Phase 9: Control Data Mixing

Never train only on new synthetic examples. The training set should always include all original gold training data.

Initial synthetic contribution should be capped:

- Synthetic pool no larger than 20 to 40 percent of the original training size.
- Or per-label targeted augmentation only for rare classes and known weak slices.

Each training run should report source counts and label distributions for original, synthetic, silver, and reviewed examples.

The augmented training distribution does not need to match the natural label distribution in the validation or test sets. In fact, the purpose of hard-case augmentation is often to intentionally enrich labels and slices that are rare, difficult, or systematically missed.

The key constraint is that validation and test distributions must remain natural, fixed, and untouched. They estimate real-world performance. The training set may be deliberately rebalanced, but that choice must be documented and evaluated.

Recommended first-iteration policy:

- Keep validation and test sets unchanged.
- Preserve all original gold training examples.
- Add synthetic examples only to the training set.
- Target synthetic data toward hard cases, rare labels, and known error slices.
- Cap synthetic volume globally and per label.
- Report the natural gold training distribution and the augmented training distribution separately.
- Use sample weighting if synthetic enrichment causes overprediction or false-positive growth.

For manuscript reporting, explicitly state that augmentation was targeted rather than distribution-matched, and show whether this improved performance on natural validation and test distributions.

### Phase 10: Tune Thresholds

After training, tune per-label thresholds on the validation set. Thresholds should optimize the target operating point for each label, such as precision-sensitive or recall-sensitive behavior depending on clinical needs.

Do not assume `0.5` is appropriate for all labels.

### Phase 11: Evaluate and Stop

Evaluate each iteration on the fixed validation set and relevant validation slices.

Track:

- Per-label precision.
- Per-label recall.
- Per-label F1.
- Micro F1.
- Macro F1.
- PR-AUC where feasible.
- Calibration and threshold curves.
- Performance by note type.
- Performance by department.

Stop iterating when:

- Rare-label performance no longer improves.
- Synthetic data starts increasing false positives.
- Teacher additions stop improving validation slices.
- The complexity cost outweighs the measured gain.

For initial development, two to four iterations should be enough.

### Phase 12: Systematic Stage Comparisons

After each development stage, run a predefined comparison against the previous stage and the original baseline. This is necessary for reproducibility, decision-making, and manuscript reporting.

Each comparison should use the same frozen validation set, the same metric definitions, and the same slice definitions. Only the intended stage component should change.

Recommended comparison sequence:

1. Rules-only baseline.
2. BioBERT-only baseline.
3. Rules-first with BioBERT fallback.
4. Rules-first with BioBERT fallback and tuned per-label thresholds.
5. Rule-refined pipeline after teacher-assisted error analysis.
6. Retrained BioBERT with approved synthetic hard cases.
7. Optional retrained BioBERT with lower-weight silver labels.
8. Final selected production candidate.

For each stage, document:

- Dataset versions and split IDs.
- Rule version and rule confidence score version.
- Model checkpoint and tokenizer.
- Prompt versions and teacher models, if used.
- Training data composition by source and label.
- Thresholds used for each label.
- Overall metrics and per-label metrics.
- Performance by clinically relevant slices.
- Rule-path and BioBERT-path coverage.
- Error categories that improved, worsened, or remained unresolved.

The comparison should explicitly answer whether each added component provides measurable benefit. If a stage adds complexity without improving validation performance or interpretability, it should not be promoted.

### Phase 13: Final Test Reporting

Use the untouched test set only after development decisions are complete. Final reporting should include:

- Overall and per-label metrics.
- Rule-path coverage.
- BioBERT-path coverage.
- Metrics by prediction source.
- Metrics by key clinical slices.
- Thresholds used.
- Training manifest reference.
- Known limitations and residual failure modes.

### Phase 14: Post-Deployment Continuous Learning

For production monitoring, run the rules and BioBERT pipeline on new cases. Use teachers only on selected subsets:

- Rule/BioBERT conflicts.
- Low-confidence BioBERT cases.
- Rare-label positives.
- Drift slices.

A monthly refresh cycle can follow this pattern:

1. Collect production cases.
2. Detect disagreements, uncertainty, and drift.
3. Send a small subset for human review.
4. Committee-label another subset as silver candidates.
5. Generate limited hard-case synthetic augmentation.
6. Retrain from the base checkpoint using gold, reviewed gold, approved synthetic, and optional lower-weight silver data.
7. Validate against the fixed validation set and drift slices.
8. Promote only if metrics improve and error profiles remain acceptable.

## Router and Ensemble Guidance

The default production design should remain:

```text
rules -> if abstain -> BioBERT
```

A learned router should not be the first priority. A router only helps if the project can reliably predict when rules or BioBERT will fail. Otherwise, it adds noise and can override correct high-precision rules.

The better near-term upgrade is confidence gating:

- Add rule confidence or rule strength.
- Use BioBERT sigmoid probabilities.
- Track entropy and margin from threshold.
- Send uncertain cases to review, teacher analysis, or abstention workflows.

If the rules remain high precision, a learned router adds little. If rules begin firing incorrectly on identifiable subsets and BioBERT is clearly better on those subsets, then evaluate a simple router or stacked ensemble.

Stacking may be preferable to hard routing because both systems can contribute:

```python
meta_input = [bert_probs, rule_outputs, rule_confidence_features]
meta_model.fit(meta_input, gold_labels)
```

## Reproducibility and Audit Requirements

Every iteration should produce a training manifest.

```python
@dataclass
class TrainingManifest:
    iteration: int
    base_model: str
    rule_version: str
    teacher_models: List[str]
    teacher_weights: Dict[str, float]
    prompt_version: str
    num_original: int
    num_silver: int
    num_synthetic: int
    synthetic_ratio: float
    label_distribution: Dict[str, int]
    thresholds: Dict[str, float]
    validation_metrics: Dict[str, float]
```

For every iteration, save:

- Train, validation, and test split IDs.
- BioBERT checkpoint name.
- Tokenizer version.
- Rule version.
- Prompt versions.
- Teacher model names.
- Aggregation method and weights.
- Synthetic generation settings.
- Synthetic parent-child mappings.
- Inclusion and exclusion criteria.
- Counts by source and label.
- Validation-selected thresholds.
- Final metrics by slice.

### Manuscript-Grade Documentation

The project should maintain documentation detailed enough to support three needs:

- Reproducibility: another analyst can reconstruct each experiment.
- Justification: each design choice has a clinical, statistical, or operational rationale.
- Manuscript description: methods, ablations, validation, and limitations can be described without relying on memory.

Maintain the following documents or artifacts throughout development:

- Label ontology document with inclusion criteria, exclusion criteria, examples, and edge cases for each first-iteration label.
- Annotation guideline version used for each gold-label batch.
- Data split manifest with instance IDs, note IDs, split assignment, date created, and exclusion criteria.
- Rule manifest with rule IDs, rule text or pattern definition, target label, confidence score, score derivation method, and version.
- Prompt registry with prompt text, prompt version, teacher model, model settings, and intended teacher role.
- Synthetic data audit file with parent instance, generated instance, target label, validation checks, and approval status.
- Experiment registry with one row per run and links to configuration, training manifest, metrics, and error analysis.
- Stage comparison report after every major pipeline change.
- Decision log explaining why each stage was accepted, rejected, or deferred.

For manuscript readiness, each final result should be traceable to:

- A frozen dataset version.
- A fixed label ontology.
- A fixed rule version.
- A fixed model checkpoint.
- A fixed threshold file.
- A fixed evaluation script.
- A fixed metrics output file.

## Suggested Repository Structure

```text
project/
├── data/
│   ├── gold/
│   ├── silver/
│   ├── synthetic/
│   └── processed/
├── prompts/
│   ├── teacher_prompt_v1.txt
│   ├── error_analysis_prompt_v1.txt
│   └── synthetic_generation_prompt_v1.txt
├── rules/
│   ├── rules_v1.yaml
│   └── rules_v2.yaml
├── models/
│   ├── biobert_baseline/
│   ├── iteration_1/
│   └── iteration_2/
├── audits/
│   ├── teacher_outputs/
│   ├── synthetic_provenance/
│   └── training_manifests/
├── docs/
│   ├── label_ontology.md
│   ├── annotation_guidelines.md
│   ├── rule_manifest.md
│   ├── experiment_registry.md
│   ├── stage_comparisons.md
│   └── decision_log.md
├── experiments/
│   ├── baseline_rules/
│   ├── baseline_biobert/
│   ├── rules_then_biobert/
│   └── synthetic_augmented/
├── src/
│   ├── run_rules.py
│   ├── run_teachers.py
│   ├── aggregate_teachers.py
│   ├── generate_synthetics.py
│   ├── validate_synthetics.py
│   ├── train_biobert.py
│   ├── evaluate.py
│   └── pipeline.py
└── configs/
    ├── iteration_1.yaml
    └── iteration_2.yaml
```

## Key Risks and Mitigations

### Risk: Teacher Labels Replace Gold Labels

Mitigation: Treat gold labels as primary truth. Use teacher outputs for targeted analysis, augmentation, and silver-label candidates only.

### Risk: Synthetic Data Creates False Positives

Mitigation: Cap synthetic volume, validate every batch, track false-positive slices, and use lower sample weights for synthetic examples if needed. Keep validation and test distributions natural so synthetic overfitting is visible.

### Risk: Augmented Training Distribution Is Misinterpreted

Mitigation: Report original training, augmented training, validation, and test label distributions separately. State that synthetic augmentation is targeted toward hard cases and is not intended to match validation or test prevalence.

### Risk: Rule Scores Are Treated as Calibrated Probabilities

Mitigation: Label first-iteration rule scores as heuristic confidence scores unless calibration has been performed. Store score derivation methods and update rule scores only at defined experiment stages.

### Risk: Prompt Diversity Redefines Labels

Mitigation: Keep the same ontology and label definitions across all teachers. Vary decision thresholds and reasoning style, not the meaning of labels.

### Risk: First-Iteration "Other" Category Becomes Too Ambiguous

Mitigation: Keep clear inclusion and exclusion criteria for `other_non_patient`. Store optional subtype notes for later ontology expansion, but evaluate the first-iteration model only on the four-label schema.

### Risk: Model Refresh Causes Drift

Mitigation: Retrain from the base checkpoint when possible, use fixed validation controls, save complete manifests, and promote only when metrics improve.

### Risk: Learned Router Adds Noise

Mitigation: Start with rules-first abstention and BioBERT fallback. Add confidence gating before considering a learned router.

## Development Milestones

### Milestone 1: Baseline Audit

Deliverables:

- Frozen data split manifest.
- Baseline rules and BioBERT outputs.
- Error table.
- Error analysis by label and slice.

### Milestone 2: Teacher Committee

Deliverables:

- Shared label ontology and output schema.
- Generalist, precision-biased, and recall-biased prompts.
- Teacher outputs for targeted discrepancy cases.
- Initial aggregation method.

### Milestone 3: Rule Refinement and Synthetic Seeds

Deliverables:

- Rule failure taxonomy.
- Proposed rule updates.
- Approved hard-case seed list.
- Synthetic generation prompt and provenance schema.

### Milestone 4: First Retraining Iteration

Deliverables:

- Approved synthetic batch.
- Updated training set.
- Retrained BioBERT model from base checkpoint.
- Validation metrics and per-label thresholds.
- Training manifest.

### Milestone 5: Production Candidate

Deliverables:

- Final rules-first/BioBERT-fallback pipeline.
- Fixed thresholds.
- Test-set report.
- Audit documentation.
- Deployment and monitoring plan.

### Milestone 6: Refresh Workflow

Deliverables:

- Production disagreement sampling logic.
- Human-review queue definition.
- Silver-label candidate process.
- Monthly retraining and promotion criteria.

## Final Recommended Design

The strongest near-term design is:

- One four-label multi-label BioBERT student for the first iteration.
- Sigmoid output head with per-label threshold tuning.
- Rules as a first-pass high-precision filter.
- Rule-specific confidence scores where feasible, clearly distinguished from calibrated probabilities.
- LLM teacher committee for development and refresh cycles.
- Gold labels as the primary truth.
- Teacher outputs used for targeted refinement, not wholesale relabeling.
- Controlled synthetic augmentation for hard, rare, or systematically missed cases.
- Fixed validation and test discipline for reproducible improvement.
- Systematic stage comparisons after every major change.

This keeps the system simple enough to audit and operate while still using LLMs where they are most valuable: finding errors, explaining failures, improving rules, and expanding difficult training slices.
