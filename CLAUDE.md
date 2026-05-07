# CLAUDE.md

Guidance for Claude Code when working in this repository.

## Project Status

Environment setup complete (Phases 0–11). The repo is ready for the modeling
roadmap defined in `PROJECT_OVERVIEW.md`.

**Pending:** Phase 11 Docker build checkpoint not yet verified — see
`docs/decision_log.md` for the exact commands and the ARM64 platform note.

## What This System Does

**PhenoContext** classifies the context of phenotype mentions in clinical notes.
Given a phenotype mention (from upstream NER) + a context window, it assigns
multi-hot labels from a 4-class ontology (defined in `src/ods_phenocontext/schema.py`):

- `confirmed` — phenotype affirmed for the patient
- `negated` — explicitly negated
- `associated_with_someone_else` — attributed to a non-patient experiencer
- `other_non_patient` — umbrella for uncertainty, hypothetical, historical, screening

## Repository Layout

```
src/ods_phenocontext/
  __init__.py               # package root, __version__
  schema.py                 # Instance, SyntheticAudit, TrainingManifest + LABEL_NAMES
  pipeline.py               # phenocontext_predict, RulesModel/BioBERTModel Protocols
  rules/__init__.py          # placeholder; real rules loaded from rules/*.yaml
  models/
    __init__.py
    biobert.py              # BioBERTMultiLabel (encoder + linear head stub)
  teachers/
    __init__.py
    bedrock_client.py       # ChatBedrock wrapper, TeacherOutput schema, build_committee()

tests/
  test_smoke.py             # trivial harness check
  test_schema.py            # Instance / SyntheticAudit / TrainingManifest
  test_pipeline_smoke.py    # both pipeline branches (rules confident, rules abstain)
  test_environment.py       # Python 3.11, torch device, BERT forward pass, BioBERT shape
  integration/
    test_bedrock_live.py    # live Bedrock round-trip (RUN_BEDROCK_INTEGRATION=1)

data/{gold,silver,synthetic,processed}/   # gitignored — never commit PHI
audits/{teacher_outputs,synthetic_provenance,training_manifests}/
configs/                    # per-iteration YAML configs
prompts/                    # versioned teacher and generation prompts
rules/                      # versioned YAML rule files
docs/
  decision_log.md           # non-obvious choices, deferrals, and deferred items
```

## Core Architecture

The deployed system is a **rules-first, BioBERT-fallback pipeline** — not an LLM in production:

1. Run rule system → if confident, return rule labels
2. If rules abstain → run BioBERT multi-label classifier
3. Apply per-label thresholds to BioBERT sigmoid outputs
4. Return labels, probabilities, and prediction source

LLMs (teacher committee) are used **only during development and refresh cycles**,
not at inference time. Teacher deps (`langchain`, `langchain-aws`, `boto3`) are
in an opt-in `[teacher]` dependency group and are absent from the production image.

### Key source files

| File | Purpose |
|---|---|
| `schema.py` | Single source of truth for all data structures |
| `pipeline.py` | `phenocontext_predict` + `RulesModel`/`BioBERTModel` Protocols |
| `models/biobert.py` | `BioBERTMultiLabel` — backbone encoder + 4-label linear head |
| `teachers/bedrock_client.py` | `build_committee()` returning 3 role-tuned LangChain runnables |

## Environment

```bash
make dev          # install runtime + dev deps + pre-commit hooks
make test         # run pytest
make lint         # ruff check
make format       # ruff format + fix
make typecheck    # mypy src

# Teacher group (requires AWS credentials + Bedrock access in us-east-2)
uv sync --group dev --group teacher
RUN_BEDROCK_INTEGRATION=1 uv run --group teacher pytest tests/integration/ -v

# Docker (production image — excludes dev + teacher)
docker build --platform linux/amd64 -t ods-phenocontext:dev .
```

**Python:** 3.11 (pinned via `.python-version` and `requires-python`).
**Torch:** `>=2.6` — minimum enforced by transformers 5.x (CVE-2025-32434).
**Teacher model:** `us.anthropic.claude-sonnet-4-6` in `us-east-2`.

## Non-Negotiable Design Constraints

Do not change these without a clear documented reason in `docs/decision_log.md`:

1. **Gold labels are primary truth.** Teacher labels are development signals only.
2. **Fixed validation set.** Never tune thresholds and evaluate on the same data.
3. **Reinitialize from base checkpoint** each retraining iteration (no continual fine-tuning
   unless explicitly justified).
4. **Cap synthetic augmentation** at 20–40% of original gold training size.
5. **No learned router first.** Rules-first + abstention → BioBERT is the default.
   Add a stacked ensemble only if validation metrics justify it.
6. **Validate synthetic batches** before training: label preservation by 2/3 teachers,
   embedding similarity, lexical diversity, dedup, manual spot review.

## Reproducibility Artifacts (required per iteration)

Every training iteration must produce a `TrainingManifest` (see `schema.py`) and
update the following:

- `audits/training_manifests/` — serialized `TrainingManifest`
- `audits/teacher_outputs/` — raw teacher responses for audited instances
- `audits/synthetic_provenance/` — `SyntheticAudit` records for every generated example
- `docs/decision_log.md` — rationale for any new design choice or deferral

## Clinical Data Constraints

This system processes clinical notes containing PHI. All data handling must comply
with HIPAA:

- `data/` is gitignored — never commit clinical notes, PHI, or derived data
- Tests use only synthetic strings — no real note text in any test file
- Do not log raw note text or entity spans to stdout or files outside `data/`
- The production Docker image has no AWS credentials and no langchain/boto3
