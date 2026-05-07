"""
Smoke tests for train_biobert.py.

Uses a tiny model and synthetic instances to verify the training loop,
checkpoint saving, and from_checkpoint loading work end-to-end on CPU.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from ods_phenocontext.models.biobert_predictor import BioBERTPredictor
from ods_phenocontext.schema import LABEL_NAMES, NUM_LABELS, Instance
from ods_phenocontext.train_biobert import (
    PhenoContextDataset,
    compute_pos_weight,
    eval_epoch,
    seed_everything,
    train_epoch,
)

# Use bert-base-uncased for tests (tiny models don't work well with transformers 5.x)
_TEST_MODEL = "dmis-lab/biobert-base-cased-v1.2"


def _synthetic_instances(n: int = 8) -> list[Instance]:
    """Generate a small set of synthetic instances for testing."""
    instances = []
    labels_pool = [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ]
    contexts = [
        "Patient has [ENT] asthma [/ENT].",
        "No [ENT] fever [/ENT] noted.",
        "Family history of [ENT] diabetes [/ENT].",
        "[ENT] Cancer [/ENT] can cause fatigue.",
    ]
    for i in range(n):
        instances.append(
            Instance(
                instance_id=f"t-{i:03d}",
                note_id=f"n-{i:03d}",
                entity_text="test",
                context_window=contexts[i % len(contexts)],
                split="train" if i < n - 2 else "val",
                gold_labels=labels_pool[i % len(labels_pool)],
            )
        )
    return instances


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------


class TestSeedEverything:
    def test_reproducible_torch(self):
        seed_everything(123)
        a = torch.randn(5)
        seed_everything(123)
        b = torch.randn(5)
        assert torch.allclose(a, b)


class TestComputePosWeight:
    def test_returns_correct_length(self):
        instances = _synthetic_instances()
        w = compute_pos_weight(instances)
        assert w.shape == (NUM_LABELS,)

    def test_weights_positive(self):
        instances = _synthetic_instances()
        w = compute_pos_weight(instances)
        assert all(v > 0 for v in w.tolist())

    def test_imbalanced_gives_higher_weight(self):
        # All confirmed — confirmed has low weight, others have high weight
        instances = [
            Instance(
                instance_id=f"t-{i}",
                note_id=f"n-{i}",
                entity_text="x",
                context_window="[ENT] x [/ENT]",
                split="train",
                gold_labels=[1, 0, 0, 0],
            )
            for i in range(20)
        ]
        w = compute_pos_weight(instances)
        # confirmed (index 0) should have the lowest weight
        assert w[0] < w[1]


class TestPhenoContextDataset:
    def test_filters_none_labels(self):
        from transformers import AutoTokenizer

        instances = _synthetic_instances()
        instances[0].gold_labels = None
        tokenizer = AutoTokenizer.from_pretrained(_TEST_MODEL)
        ds = PhenoContextDataset(instances, tokenizer, max_length=32)
        assert len(ds) == len(instances) - 1

    def test_getitem_shapes(self):
        from transformers import AutoTokenizer

        instances = _synthetic_instances()
        tokenizer = AutoTokenizer.from_pretrained(_TEST_MODEL)
        ds = PhenoContextDataset(instances, tokenizer, max_length=32)
        item = ds[0]
        assert item["input_ids"].shape == (32,)
        assert item["attention_mask"].shape == (32,)
        assert item["labels"].shape == (NUM_LABELS,)
        assert "instance_id" in item

    def test_token_type_ids_dtype(self):
        from transformers import AutoTokenizer

        instances = _synthetic_instances()
        tokenizer = AutoTokenizer.from_pretrained(_TEST_MODEL)
        ds = PhenoContextDataset(instances, tokenizer, max_length=32)
        item = ds[0]
        assert item["token_type_ids"].dtype == torch.long


# ---------------------------------------------------------------------------
# Integration: 1-epoch training loop
# ---------------------------------------------------------------------------


class TestTrainingLoop:
    @pytest.fixture(scope="class")
    def trained_checkpoint(self, tmp_path_factory) -> Path:
        """Run a 1-epoch training loop and save checkpoint."""
        from torch.utils.data import DataLoader
        from transformers import AutoTokenizer

        from ods_phenocontext.models.biobert import SPECIAL_TOKENS, BioBERTMultiLabel
        from ods_phenocontext.threshold_tuning import tune_and_save

        out = tmp_path_factory.mktemp("ckpt")
        instances = _synthetic_instances(n=8)
        train_insts = [i for i in instances if i.split == "train"]
        val_insts = [i for i in instances if i.split == "val"]

        tokenizer = AutoTokenizer.from_pretrained(_TEST_MODEL)
        tokenizer.add_special_tokens({"additional_special_tokens": SPECIAL_TOKENS})

        train_ds = PhenoContextDataset(train_insts, tokenizer, max_length=32)
        val_ds = PhenoContextDataset(val_insts, tokenizer, max_length=32)
        train_loader = DataLoader(train_ds, batch_size=4, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=4, shuffle=False)

        model = BioBERTMultiLabel(num_labels=NUM_LABELS, model_name=_TEST_MODEL, dropout=0.0)
        model.resize_token_embeddings(len(tokenizer))
        ent_id = tokenizer.convert_tokens_to_ids("[ENT]")
        end_id = tokenizer.convert_tokens_to_ids("[/ENT]")
        model.set_entity_token_ids(ent_id, end_id)

        device = torch.device("cpu")
        pos_weight = compute_pos_weight(train_ds.instances).to(device)
        criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

        seed_everything(42)
        loss = train_epoch(model, train_loader, optimizer, None, criterion, device)
        assert loss > 0

        val_loss, val_id_probs = eval_epoch(model, val_loader, device, criterion)
        assert val_loss > 0

        # Save model
        model_dir = out / "model"
        model_dir.mkdir(parents=True, exist_ok=True)
        tokenizer.save_pretrained(model_dir)
        torch.save(model.state_dict(), model_dir / "biobert_multilabel.pt")
        config = {
            "model_name": _TEST_MODEL,
            "num_labels": NUM_LABELS,
            "dropout": 0.0,
            "max_length": 32,
            "ent_token_id": ent_id,
            "end_token_id": end_id,
        }
        (model_dir / "model_config.json").write_text(json.dumps(config))

        # Save thresholds
        val_probs = [val_id_probs[inst.instance_id] for inst in val_ds.instances]
        val_gold = [inst.gold_labels for inst in val_ds.instances]
        tune_and_save(val_probs, val_gold, split="val", output_path=out / "thresholds.json")

        return out

    def test_checkpoint_has_model_files(self, trained_checkpoint: Path):
        model_dir = trained_checkpoint / "model"
        assert (model_dir / "biobert_multilabel.pt").exists()
        assert (model_dir / "model_config.json").exists()
        assert (model_dir / "tokenizer_config.json").exists()

    def test_checkpoint_has_thresholds(self, trained_checkpoint: Path):
        thresholds = json.loads((trained_checkpoint / "thresholds.json").read_text())
        assert "thresholds" in thresholds
        assert set(thresholds["thresholds"].keys()) == set(LABEL_NAMES)

    def test_from_checkpoint_loads(self, trained_checkpoint: Path):
        predictor = BioBERTPredictor.from_checkpoint(trained_checkpoint)
        assert predictor.model is not None
        assert predictor.max_length == 32

    def test_from_checkpoint_predicts(self, trained_checkpoint: Path):
        predictor = BioBERTPredictor.from_checkpoint(trained_checkpoint)
        inst = Instance(
            instance_id="x",
            note_id="x",
            entity_text="test",
            context_window="Patient has [ENT] test [/ENT].",
            split="val",
        )
        probs = predictor.predict_proba(inst)
        assert len(probs) == NUM_LABELS
        assert all(0.0 <= p <= 1.0 for p in probs)
