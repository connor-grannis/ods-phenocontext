"""
BioBERT fine-tuning script for phenotype context classification.

Usage:
    uv run python -m ods_phenocontext.train_biobert \\
        --parquet  data/all_training_samples.parquet \\
        --manifest data/gold/split_manifest.jsonl \\
        --out      checkpoints/biobert_v1/

Trains from the base BioBERT checkpoint each run (no continual fine-tuning —
see CLAUDE.md design constraints). Writes:
    checkpoints/<out>/model/          — HF-format model + tokenizer
    checkpoints/<out>/thresholds.json — per-label thresholds tuned on val
    checkpoints/<out>/train_metrics.json — final epoch train/val metrics
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import click
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

from ods_phenocontext.data.loader import load_instances
from ods_phenocontext.evaluate import compute_metrics
from ods_phenocontext.models.biobert import SPECIAL_TOKENS, BioBERTMultiLabel
from ods_phenocontext.schema import LABEL_NAMES, NUM_LABELS, Instance
from ods_phenocontext.threshold_tuning import tune_and_save

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class PhenoContextDataset(Dataset):
    """Tokenized dataset from a list of Instances with gold_labels."""

    def __init__(
        self,
        instances: list[Instance],
        tokenizer,
        max_length: int = 256,
    ) -> None:
        self.instances = [i for i in instances if i.gold_labels is not None]
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.instances)

    def __getitem__(self, idx: int) -> dict:
        inst = self.instances[idx]
        enc = self.tokenizer(
            inst.context_window,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "token_type_ids": enc.get("token_type_ids", torch.zeros(1, self.max_length)).squeeze(0),
            "labels": torch.tensor(inst.gold_labels, dtype=torch.float),
        }


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def train_epoch(
    model: BioBERTMultiLabel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        token_type_ids = batch["token_type_ids"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()
        logits = model(input_ids, attention_mask, token_type_ids)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader)


@torch.no_grad()
def eval_epoch(
    model: BioBERTMultiLabel,
    instances: list[Instance],
    loader: DataLoader,
    device: torch.device,
    thresholds: list[float],
) -> tuple[float, list[list[float]]]:
    """Run evaluation pass. Returns (mean_loss, probs_matrix)."""
    model.eval()
    criterion = nn.BCEWithLogitsLoss()
    total_loss = 0.0
    all_probs: list[list[float]] = []

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        token_type_ids = batch["token_type_ids"].to(device)
        labels = batch["labels"].to(device)

        logits = model(input_ids, attention_mask, token_type_ids)
        loss = criterion(logits, labels)
        total_loss += loss.item()
        probs = torch.sigmoid(logits).cpu().tolist()
        all_probs.extend(probs)

    return total_loss / len(loader), all_probs


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@click.command()
@click.option("--parquet", required=True, type=click.Path(exists=True, path_type=Path))
@click.option("--manifest", required=True, type=click.Path(exists=True, path_type=Path))
@click.option("--out", required=True, type=click.Path(path_type=Path))
@click.option(
    "--model",
    default="dmis-lab/biobert-base-cased-v1.2",
    show_default=True,
    help="HF model ID for the encoder backbone.",
)
@click.option("--epochs", default=5, show_default=True, type=int)
@click.option("--batch-size", default=32, show_default=True, type=int)
@click.option("--lr", default=2e-5, show_default=True, type=float)
@click.option("--max-length", default=256, show_default=True, type=int)
@click.option("--dropout", default=0.1, show_default=True, type=float)
@click.option(
    "--device",
    default=None,
    help="Force device (cpu/cuda/mps). Auto-detected if omitted.",
)
def main(
    parquet: Path,
    manifest: Path,
    out: Path,
    model: str,
    epochs: int,
    batch_size: int,
    lr: float,
    max_length: int,
    dropout: float,
    device: str | None,
) -> None:
    """Fine-tune BioBERT on the phenotype context classification task."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    # Device selection
    if device:
        dev = torch.device(device)
    elif torch.cuda.is_available():
        dev = torch.device("cuda")
    elif torch.backends.mps.is_available():
        dev = torch.device("mps")
    else:
        dev = torch.device("cpu")
    log.info("Using device: %s", dev)

    # Load and split instances
    log.info("Loading instances from %s", parquet)
    all_instances = list(load_instances(parquet, manifest))
    train_instances = [i for i in all_instances if i.split == "train"]
    val_instances = [i for i in all_instances if i.split == "val"]
    log.info("train=%d  val=%d", len(train_instances), len(val_instances))

    # Tokenizer + special tokens
    tokenizer = AutoTokenizer.from_pretrained(model)
    tokenizer.add_special_tokens({"additional_special_tokens": SPECIAL_TOKENS})

    # Datasets and loaders
    train_ds = PhenoContextDataset(train_instances, tokenizer, max_length)
    val_ds = PhenoContextDataset(val_instances, tokenizer, max_length)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    # Model — always reinitialize from base checkpoint
    log.info("Loading backbone: %s", model)
    biobert = BioBERTMultiLabel(num_labels=NUM_LABELS, model_name=model, dropout=dropout)
    biobert.resize_token_embeddings(len(tokenizer))
    ent_id = tokenizer.convert_tokens_to_ids("[ENT]")
    end_id = tokenizer.convert_tokens_to_ids("[/ENT]")
    biobert.set_entity_token_ids(ent_id, end_id)
    biobert.to(dev)

    optimizer = torch.optim.AdamW(biobert.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    # Training loop
    history: list[dict] = []
    for epoch in range(1, epochs + 1):
        train_loss = train_epoch(biobert, train_loader, optimizer, criterion, dev)
        val_loss, val_probs = eval_epoch(
            biobert, val_instances, val_loader, dev, [0.5] * NUM_LABELS
        )
        log.info("Epoch %d/%d  train_loss=%.4f  val_loss=%.4f", epoch, epochs, train_loss, val_loss)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})

    # Threshold tuning on val
    val_gold = [i.gold_labels for i in val_ds.instances]
    thresholds_path = out / "thresholds.json"
    log.info("Tuning thresholds on val set → %s", thresholds_path)
    threshold_output = tune_and_save(val_probs, val_gold, split="val", output_path=thresholds_path)
    thresholds_list = [threshold_output["thresholds"][name] for name in LABEL_NAMES]

    # Final val metrics at tuned thresholds
    for inst, probs in zip(val_ds.instances, val_probs, strict=True):
        inst.rule_abstained = True
        inst.biobert_probs = probs
        inst.biobert_labels = [int(p >= t) for p, t in zip(probs, thresholds_list, strict=True)]
    val_metrics = compute_metrics(val_ds.instances)

    # Save model + tokenizer
    model_dir = out / "model"
    model_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(model_dir)
    biobert.encoder.save_pretrained(model_dir)
    # Save classifier head weights separately (not part of the HF encoder)
    torch.save(biobert.classifier.state_dict(), model_dir / "classifier_head.pt")
    log.info("Model saved to %s", model_dir)

    # Training metrics artifact
    train_metrics_path = out / "train_metrics.json"
    train_metrics_path.write_text(
        json.dumps(
            {
                "model": model,
                "epochs": epochs,
                "batch_size": batch_size,
                "lr": lr,
                "max_length": max_length,
                "dropout": dropout,
                "history": history,
                "val_metrics_tuned": val_metrics,
                "thresholds": threshold_output["thresholds"],
            },
            indent=2,
        )
    )
    log.info("Training metrics saved to %s", train_metrics_path)
    log.info("macro_f1 (tuned thresholds): %.4f", val_metrics.get("macro_f1", 0.0))


if __name__ == "__main__":
    main()
