# Decision Log

Tracks non-obvious design choices, deferred items, and the rationale behind them.

---

## 2026-05-07 — Phase 11 Docker build incomplete (ARM64 platform mismatch)

**Status:** In progress — Dockerfile and `.dockerignore` committed; build
checkpoint not yet verified.

**What works:** The multi-stage Dockerfile builds correctly when run with
`--platform linux/amd64`. The lockfile was updated with `required-environments`
so both the macOS MPS wheel and the Linux CUDA wheel are pre-resolved.

**What's pending:** The build was interrupted mid-download during the initial
run. To complete the Phase 11 checkpoint, run:

```bash
docker build --platform linux/amd64 -t ods-phenocontext:dev .
# Should print "torch X.X.X | CUDA: False" (no GPU on Mac — acceptable)
docker run --rm ods-phenocontext:dev python -c "import torch; print(torch.cuda.is_available())"
# Must fail with ModuleNotFoundError
docker run --rm ods-phenocontext:dev python -c "import langchain_aws"
```

**Root cause of ARM64 issue:** Docker Desktop on Apple Silicon runs Linux
containers as `manylinux_2_36_aarch64`, but the PyTorch CUDA index
(`pytorch-cu124`) only publishes `linux_x86_64` wheels. Fixed by adding
`--platform linux/amd64` to the build command and adding `required-environments`
to `pyproject.toml` so the lockfile pre-resolves the x86_64 wheel on macOS.

---

## 2026-05-07 — rules_v1.yaml dropped; rules implemented in Python

**Decision:** The rule system uses `rules/patterns.py` (Python-native compiled
regexes) rather than a YAML file loaded at runtime.

**Reason:** The patterns are regex-heavy and gain no readability from YAML
serialization.  A Python module is easier to test, type-check, and version
than a YAML + loader pair.  `docs/rule_manifest.md` serves as the
human-readable documentation layer.

---

## 2026-05-07 — Confirmed class capped at 15,000 to address class imbalance

**Decision:** Instances where all three context labels are False (i.e., the
mention is "confirmed" for the patient) are downsampled to a maximum of 15,000
during manifest generation via `--max-confirmed 15000`.

**Reason:** The raw training data has ~23,500 confirmed instances against ~7,665
negated, ~2,924 family, and ~1,674 hypothetical — a roughly 3:1 ratio of
confirmed to all other classes combined.  Leaving this imbalance uncorrected
would bias the BioBERT head toward the confirmed class and inflate macro F1.
Downsampling confirmed to 15,000 brings the ratio closer to 2:1 without
discarding any minority-class examples.  Excess confirmed rows are marked
`exclusion_reason="confirmed_cap"` in the manifest so they can be recovered
if the balance target is revised.

**How to apply:** The cap is applied before the 90/10 train/val split so both
splits reflect the same class balance.  Re-evaluate the cap value after the
first baseline iteration once per-label val metrics are available.

---

## 2026-05-07 — note_id equals instance_id in split manifest

**Decision:** `ManifestRow.note_id` is set equal to `instance_id` when generating
the manifest from `all_training_samples.parquet`.

**Reason:** The source parquet has no note-level grouping column — each row is
an independent mention with no identifier linking it back to a source document.
The cross-split constraint (a note must not appear in both train and val) is
therefore vacuous but still enforced; it will become meaningful if a future data
source includes note IDs.

---

## 2026-05-07 — No text preprocessing needed; upstream pipeline handles it

**Decision:** `src/ods_phenocontext/preprocessing/` will not be built. The
`preprocessing` placeholder module described in the baselines plan is dropped.

**Reason:** Text arrives from the upstream NER pipeline already in a form
suitable for both the rule-based classifier and BioBERT — tokenization,
casing, and whitespace are handled before this system sees the text.
`Instance.context_window` is treated as canonical on ingestion; no further
normalization is applied at any pipeline stage.

**Impact:** M0 of the baselines plan skips the preprocessing placeholder.
`Instance.from_raw` (M1) will not accept a `preprocessor` argument — it stores
`raw_context_window` directly as `context_window`.

---

## 2026-05-07 — medspacy deferred

**Decision:** `medspacy` (ConText / NegEx) not added in Phase 5.

**Reason:** medspacy adds significant transitive deps and ConText logic is better
implemented as a custom rule family with explicit confidence calibration (per
CLAUDE.md). Re-evaluate when the rules module in Phase 8 reaches ConText-style
negation/experiencer rules.

**Re-evaluate at:** modeling Phase 2–3 (rule system build-out).

---

## 2026-05-07 — torch lower bound raised to >=2.6

**Decision:** `torch>=2.4,<2.6` changed to `torch>=2.6`.

**Reason:** transformers 5.x blocks `torch.load` on torch < 2.6 for all
`.bin`-format weights (CVE-2025-32434). `biobert-base-cased-v1.2` ships
`.bin` weights; the environment test failed with `ValueError` until torch
was upgraded to 2.6+. The CUDA wheel index (`pytorch-cu124`) still resolves
correctly on Linux for 2.6+.

---

## 2026-05-07 — prajjwal1/bert-tiny replaced with bert-base-uncased

**Decision:** Checkpoint tokenizer changed from `prajjwal1/bert-tiny` to `bert-base-uncased`.

**Reason:** `prajjwal1/bert-tiny` tokenizer is incompatible with transformers 5.x
(missing fast tokenizer files; raises `ValueError` on `from_pretrained`).
`bert-base-uncased` uses the same BERT WordPiece tokenizer and is the correct
base for BioBERT fine-tuning anyway.

---

## 2026-05-07 — LLM calls routed through AWS Bedrock via LangChain

**Decision:** All teacher LLM calls use `langchain_aws.ChatBedrock`.
No direct OpenAI or Anthropic SDK usage.

**Reason:** Single auth path (AWS IAM), single audit surface, consistent with
hospital infrastructure. Bedrock model access must be granted per-region in
the AWS console before use; record enabled model IDs and region here.

**Enabled models:** `us.anthropic.claude-sonnet-4-6` in `us-east-2`.
Bedrock model access must be granted in the AWS console for this region before
any live call. Update this entry if additional model IDs are enabled.

---

## 2026-05-07 — Threshold tuning and evaluation use the same val set in Baseline 4

**Decision:** In the `rules_then_biobert_tuned` baseline (M10), thresholds are
tuned on the validation set and metrics are also reported on the validation set.

**Caveat:** This means Baseline 4 val metrics are optimistically biased — the
thresholds were chosen specifically to maximize F1 on this set. The comparison
between Baseline 3 (default 0.5) and Baseline 4 (tuned) is still useful for
confirming that threshold tuning works mechanically and quantifying the ceiling
it provides, but the Baseline 4 val numbers should not be interpreted as a
true hold-out estimate. Once real training data is available, thresholds will
be tuned on val and reported on a held-out test split.
