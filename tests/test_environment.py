"""
End-to-end environment smoke test (Phase 10 contract).

Asserts that every major dependency is wired up correctly and that the
BioBERTMultiLabel stub produces output of the expected shape.  If any
phase 2–8 regression occurs, this test will fail first.

No network calls — all models loaded from the local HF cache.
No PHI — all inputs are synthetic strings.
"""

import sys

import torch
from transformers import AutoModel, AutoTokenizer

from ods_phenocontext.models.biobert import BioBERTMultiLabel
from ods_phenocontext.schema import NUM_LABELS

# ---------------------------------------------------------------------------
# Python version
# ---------------------------------------------------------------------------


def test_python_version_is_311():
    assert sys.version_info[:2] == (3, 11), f"Expected Python 3.11, got {sys.version_info[:2]}"


# ---------------------------------------------------------------------------
# Torch device
# ---------------------------------------------------------------------------


def test_torch_imports_and_reports_device():
    # On MPS-capable Macs the device is "mps"; on CUDA hosts "cuda"; otherwise "cpu".
    # All three are acceptable — what matters is that torch imports and a device exists.
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    assert device is not None
    # Smoke-test that we can allocate a tensor on the chosen device
    t = torch.zeros(2, 2, device=device)
    assert t.shape == (2, 2)


# ---------------------------------------------------------------------------
# Transformers: bert-base-uncased tokenizer + forward pass
# ---------------------------------------------------------------------------


def test_bert_base_uncased_forward_pass():
    """
    Loads bert-base-uncased (cached; no network call), runs a 4-token batch,
    and checks that the hidden states have the expected shape.
    """
    model_name = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()

    # Synthetic inputs — no clinical text
    texts = [
        "The patient has asthma.",
        "No history of diabetes.",
        "Mother has hypertension.",
        "Rule out pneumonia.",
    ]
    batch = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=64)

    with torch.no_grad():
        outputs = model(**batch)

    hidden = outputs.last_hidden_state
    # Shape: (batch_size=4, seq_len, hidden_size=768)
    assert hidden.shape[0] == 4
    assert hidden.shape[2] == 768


# ---------------------------------------------------------------------------
# BioBERTMultiLabel stub: logits shape
# ---------------------------------------------------------------------------


def test_biobert_multilabel_logits_shape():
    """
    Instantiates BioBERTMultiLabel with biobert-base-cased-v1.2 (cached),
    runs a single-example forward pass, and asserts logits shape is (1, NUM_LABELS).
    """
    model = BioBERTMultiLabel(
        num_labels=NUM_LABELS,
        model_name="dmis-lab/biobert-base-cased-v1.2",
    )
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained("dmis-lab/biobert-base-cased-v1.2")
    enc = tokenizer(
        "The patient denies asthma.",
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=64,
    )

    with torch.no_grad():
        logits = model(
            input_ids=enc["input_ids"],
            attention_mask=enc["attention_mask"],
            token_type_ids=enc.get("token_type_ids"),
        )

    assert logits.shape == (1, NUM_LABELS), (
        f"Expected logits shape (1, {NUM_LABELS}), got {logits.shape}"
    )


def test_biobert_multilabel_predict_proba_length():
    """predict_proba should return a list of NUM_LABELS floats in [0, 1]."""
    model = BioBERTMultiLabel(
        num_labels=NUM_LABELS,
        model_name="dmis-lab/biobert-base-cased-v1.2",
    )

    tokenizer = AutoTokenizer.from_pretrained("dmis-lab/biobert-base-cased-v1.2")
    enc = tokenizer(
        "History of asthma, now resolved.",
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=64,
    )

    # predict_proba currently expects tensors; pass them directly
    probs = model.predict_proba(
        input_ids=enc["input_ids"],
        attention_mask=enc["attention_mask"],
        token_type_ids=enc.get("token_type_ids"),
    )

    assert len(probs) == NUM_LABELS
    assert all(0.0 <= p <= 1.0 for p in probs), f"Probabilities out of [0,1]: {probs}"


# ---------------------------------------------------------------------------
# Schema sanity (regression guard for Phase 8)
# ---------------------------------------------------------------------------


def test_num_labels_is_4():
    assert NUM_LABELS == 4
