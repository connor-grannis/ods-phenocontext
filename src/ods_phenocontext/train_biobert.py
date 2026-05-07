"""
BioBERT fine-tuning script for phenotype context classification.

Usage:
    uv run python -m ods_phenocontext.train_biobert \\
        --parquet  data/all_training_samples.parquet \\
        --manifest data/gold/split_manifest.jsonl \\
        --out      checkpoints/biobert_v1/

Trains from the base BioBERT checkpoint each run (no continual fine-tuning —
see CLAUDE.md design constraints). Writes:
    <out>/model/              — full model state (encoder + head + tokenizer)
    <out>/thresholds.json     — per-label thresholds tuned on val
    <out>/train_metrics.json  — loss/F1 history + hyperparameters
    <out>/training_manifest.json — TrainingManifest per schema.py
"""

from __future__ import annotations

import json
import logging
import random
import subprocess
from collections import Counter
from pathlib import Path

import click
import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LinearLR, SequentialLR
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

from ods_phenocontext.data.loader import load_instances
from ods_phenocontext.evaluate import compute_metrics
from ods_phenocontext.models.biobert import SPECIAL_TOKENS, BioBERTMultiLabel
from ods_phenocontext.schema import LABEL_NAMES, NUM_LABELS, Instance, TrainingManifest
from ods_phenocontext.threshold_tuning import tune_and_save

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _git_sha() -> str:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
            )
            .decode()
            .strip()
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


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
            "token_type_ids": enc.get(
                "token_type_ids",
                torch.zeros(1, self.max_length, dtype=torch.long),
            ).squeeze(0),
            "labels": torch.tensor(inst.gold_labels, dtype=torch.float),
            "instance_id": inst.instance_id,
        }


# ---------------------------------------------------------------------------
# Pos-weight computation
# ---------------------------------------------------------------------------


def compute_pos_weight(instances: list[Instance]) -> torch.Tensor:
    """Compute BCEWithLogitsLoss pos_weight from training label frequencies."""
    n = len(instances)
    pos_counts = [0] * NUM_LABELS
    for inst in instances:
        if inst.gold_labels is None:
            continue
        for i, v in enumerate(inst.gold_labels):
            pos_counts[i] += v

    weights = []
    for count in pos_counts:
        neg_count = n - count
        # Avoid div-by-zero; cap weight at 10x
        w = min(neg_count / max(count, 1), 10.0)
        weights.append(w)

    return torch.tensor(weights, dtype=torch.float)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def train_epoch(
    model: BioBERTMultiLabel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    criterion: nn.Module,
    device: torch.device,
    max_grad_norm: float = 1.0,
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
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        total_loss += loss.item()

    return total_loss / len(loader)


@torch.no_grad()
def eval_epoch(
    model: BioBERTMultiLabel,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
) -> tuple[float, dict[str, list[float]]]:
    """Run evaluation pass. Returns (mean_loss, {instance_id: probs})."""
    model.eval()
    total_loss = 0.0
    id_to_probs: dict[str, list[float]] = {}

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        token_type_ids = batch["token_type_ids"].to(device)
        labels = batch["labels"].to(device)
        instance_ids = batch["instance_id"]

        logits = model(input_ids, attention_mask, token_type_ids)
        loss = criterion(logits, labels)
        total_loss += loss.item()
        probs = torch.sigmoid(logits).cpu().tolist()
        for iid, p in zip(instance_ids, probs, strict=True):
            id_to_probs[iid] = p

    return total_loss / len(loader), id_to_probs


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@click.command()
@click.option("--parquet", required=True, type=click.Path(exists=True, path_type=Path))
@click.option("--manifest", required=True, type=click.Path(exists=True, path_type=Path))
@click.option("--out", required=True, type=click.Path(path_type=Path))
@click.option(
    "--model",
    "model_name",
    default="dmis-lab/biobert-base-cased-v1.2",
    show_default=True,
    help="HF model ID for the encoder backbone.",
)
@click.option("--epochs", default=5, show_default=True, type=int)
@click.option("--batch-size", default=32, show_default=True, type=int)
@click.option("--lr", default=2e-5, show_default=True, type=float)
@click.option("--max-length", default=256, show_default=True, type=int)
@click.option("--dropout", default=0.1, show_default=True, type=float)
@click.option("--seed", default=42, show_default=True, type=int)
@click.option("--warmup-fraction", default=0.1, show_default=True, type=float)
@click.option(
    "--device",
    default=None,
    help="Force device (cpu/cuda/mps). Auto-detected if omitted.",
)
def main(
    parquet: Path,
    manifest: Path,
    out: Path,
    model_name: str,
    epochs: int,
    batch_size: int,
    lr: float,
    max_length: int,
    dropout: float,
    seed: int,
    warmup_fraction: float,
    device: str | None,
) -> None:
    """Fine-tune BioBERT on the phenotype context classification task."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    seed_everything(seed)
    out.mkdir(parents=True, exist_ok=True)

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
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.add_special_tokens({"additional_special_tokens": SPECIAL_TOKENS})

    # Datasets and loaders
    train_ds = PhenoContextDataset(train_instances, tokenizer, max_length)
    val_ds = PhenoContextDataset(val_instances, tokenizer, max_length)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    # Model — always reinitialize from base checkpoint
    log.info("Loading backbone: %s", model_name)
    biobert = BioBERTMultiLabel(num_labels=NUM_LABELS, model_name=model_name, dropout=dropout)
    biobert.resize_token_embeddings(len(tokenizer))
    ent_id = tokenizer.convert_tokens_to_ids("[ENT]")
    end_id = tokenizer.convert_tokens_to_ids("[/ENT]")
    biobert.set_entity_token_ids(ent_id, end_id)
    biobert.to(dev)

    # Pos-weight for class imbalance
    pos_weight = compute_pos_weight(train_ds.instances).to(dev)
    log.info("pos_weight: %s", pos_weight.tolist())
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # Optimizer + LR schedule (linear warmup → linear decay)
    optimizer = torch.optim.AdamW(biobert.parameters(), lr=lr)
    total_steps = len(train_loader) * epochs
    warmup_steps = max(1, int(total_steps * warmup_fraction))
    decay_steps = total_steps - warmup_steps
    warmup_sched = LinearLR(optimizer, start_factor=0.01, total_iters=warmup_steps)
    decay_sched = LinearLR(optimizer, start_factor=1.0, end_factor=0.0, total_iters=decay_steps)
    scheduler = SequentialLR(optimizer, [warmup_sched, decay_sched], milestones=[warmup_steps])

    # Training loop with best-checkpoint tracking
    history: list[dict] = []
    best_val_loss = float("inf")
    best_state: dict | None = None
    best_probs: dict[str, list[float]] = {}
    best_epoch = 0

    for epoch in range(1, epochs + 1):
        train_loss = train_epoch(biobert, train_loader, optimizer, scheduler, criterion, dev)
        val_loss, val_id_probs = eval_epoch(biobert, val_loader, dev, criterion)
        log.info(
            "Epoch %d/%d  train_loss=%.4f  val_loss=%.4f",
            epoch,
            epochs,
            train_loss,
            val_loss,
        )
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in biobert.state_dict().items()}
            best_probs = val_id_probs
            best_epoch = epoch

    log.info("Best epoch: %d (val_loss=%.4f)", best_epoch, best_val_loss)

    # Restore best model
    if best_state is not None:
        biobert.load_state_dict(best_state)
    biobert.to(dev)

    # Threshold tuning on val (using best-epoch probs)
    # Align probs to val_ds.instances order via instance_id
    val_probs_ordered = [best_probs[inst.instance_id] for inst in val_ds.instances]
    val_gold = [inst.gold_labels for inst in val_ds.instances]
    thresholds_path = out / "thresholds.json"
    log.info("Tuning thresholds on val set → %s", thresholds_path)
    threshold_output = tune_and_save(
        val_probs_ordered, val_gold, split="val", output_path=thresholds_path
    )
    thresholds_list = [threshold_output["thresholds"][name] for name in LABEL_NAMES]

    # Final val metrics at tuned thresholds
    for inst in val_ds.instances:
        probs = best_probs[inst.instance_id]
        inst.rule_abstained = True
        inst.biobert_probs = probs
        inst.biobert_labels = [int(p >= t) for p, t in zip(probs, thresholds_list, strict=True)]
    val_metrics = compute_metrics(val_ds.instances)

    # Save full model (encoder + classifier head + tokenizer + config)
    model_dir = out / "model"
    model_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(model_dir)
    torch.save(biobert.state_dict(), model_dir / "biobert_multilabel.pt")
    # Save model config for reproducible loading
    model_config = {
        "model_name": model_name,
        "num_labels": NUM_LABELS,
        "dropout": dropout,
        "max_length": max_length,
        "ent_token_id": ent_id,
        "end_token_id": end_id,
    }
    (model_dir / "model_config.json").write_text(json.dumps(model_config, indent=2))
    log.info("Model saved to %s", model_dir)

    # Training metrics artifact
    git_sha = _git_sha()
    train_metrics_path = out / "train_metrics.json"
    train_metrics_path.write_text(
        json.dumps(
            {
                "model": model_name,
                "epochs": epochs,
                "best_epoch": best_epoch,
                "batch_size": batch_size,
                "lr": lr,
                "max_length": max_length,
                "dropout": dropout,
                "seed": seed,
                "warmup_fraction": warmup_fraction,
                "pos_weight": pos_weight.tolist(),
                "git_sha": git_sha,
                "history": history,
                "val_metrics_tuned": val_metrics,
                "thresholds": threshold_output["thresholds"],
            },
            indent=2,
        )
    )

    # TrainingManifest
    source_counts: Counter = Counter()
    for inst in train_ds.instances:
        source_counts[inst.source_type] += 1
    label_dist = {name: 0 for name in LABEL_NAMES}
    for inst in train_ds.instances:
        if inst.gold_labels:
            for i, v in enumerate(inst.gold_labels):
                if v:
                    label_dist[LABEL_NAMES[i]] += v

    training_manifest = TrainingManifest(
        iteration=0,
        base_model=model_name,
        rule_version="v1",
        teacher_models=[],
        teacher_weights={},
        prompt_version="n/a",
        num_original=source_counts.get("original", 0),
        num_silver=source_counts.get("silver", 0),
        num_synthetic=source_counts.get("synthetic", 0),
        synthetic_ratio=(
            source_counts.get("synthetic", 0) / max(source_counts.get("original", 1), 1)
        ),
        label_distribution=label_dist,
        thresholds=threshold_output["thresholds"],
        validation_metrics={k: v for k, v in val_metrics.items() if isinstance(v, (int, float))},
    )
    manifest_path = out / "training_manifest.json"
    manifest_path.write_text(json.dumps(training_manifest.to_dict(), indent=2))

    log.info("Training metrics saved to %s", train_metrics_path)
    log.info("Training manifest saved to %s", manifest_path)
    log.info("macro_f1 (tuned thresholds): %.4f", val_metrics.get("macro_f1", 0.0))


if __name__ == "__main__":
    main()
