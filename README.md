# ods-phenocontext

A hybrid phenotype context classifier: rules-first with BioBERT fallback,
trained via a teacher committee of LLMs (AWS Bedrock) and active
human-in-the-loop learning.

**Status:** Environment setup complete (Phases 0–9). Modeling roadmap in progress.

---

## What it does

Given a phenotype mention (from an upstream NER system) and a context window
from a clinical note, PhenoContext assigns multi-hot labels from a 4-class ontology:

| Label | Meaning |
|---|---|
| `confirmed` | Phenotype affirmed for the patient |
| `negated` | Explicitly negated |
| `associated_with_someone_else` | Attributed to a non-patient experiencer |
| `other_non_patient` | Hypothetical, historical, screening, or uncertain |

## Architecture

1. **Rule system** — fast, auditable, high-precision on easy cases
2. **BioBERT fallback** — multi-label classifier for cases where rules abstain
3. **Teacher committee** (dev only) — LLM committee via AWS Bedrock used to
   label hard cases and generate synthetic training data; never runs at inference

## Quickstart

```bash
# Install uv (if needed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install runtime + dev deps and pre-commit hooks
make dev

# Run tests
make test

# Lint / format / type-check
make lint
make format
make typecheck
```

## Teacher group (dev/refresh only)

The teacher group (LangChain + AWS Bedrock) is opt-in and not installed by default:

```bash
uv sync --group dev --group teacher

# Requires AWS credentials + Bedrock model access in us-east-2
# See .env.example for required environment variables
```

Live Bedrock round-trip tests are gated behind an env var to prevent
accidental AWS spend:

```bash
RUN_BEDROCK_INTEGRATION=1 uv run --group teacher pytest tests/integration/ -v
```

## Data handling (HIPAA)

- `data/` is gitignored — never commit clinical notes or PHI
- Tests use only synthetic strings
- No raw note text is logged to stdout or files outside `data/`
- See `CLAUDE.md` for full clinical data constraints
