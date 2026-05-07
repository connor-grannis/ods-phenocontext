# Decision Log

Tracks non-obvious design choices, deferred items, and the rationale behind them.

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
