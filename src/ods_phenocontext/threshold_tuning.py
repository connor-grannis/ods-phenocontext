"""
Per-label threshold tuning on the validation set.

Sweeps thresholds in [0.05, 0.95] step 0.01 per label independently,
selecting the threshold that maximizes the chosen objective (default: F1).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

from ods_phenocontext.schema import LABEL_NAMES, NUM_LABELS


def tune_thresholds(
    probs_matrix: list[list[float]] | np.ndarray,
    gold_matrix: list[list[int]] | np.ndarray,
    split: str,
    objective: str = "f1",
) -> dict[str, float]:
    """Find per-label optimal thresholds on a validation set.

    Args:
        probs_matrix: (n_samples, NUM_LABELS) predicted probabilities.
        gold_matrix:  (n_samples, NUM_LABELS) binary gold labels.
        split:        Must be "val" — raises ValueError otherwise.
        objective:    Optimization target (currently only "f1" supported).

    Returns:
        Dict mapping label_name → optimal threshold.

    Raises:
        ValueError: if split != "val" or objective is unsupported.
    """
    if split != "val":
        raise ValueError(
            f"Threshold tuning must be done on the val split, got {split!r}. "
            "Tuning on test is forbidden — see CLAUDE.md."
        )
    if objective != "f1":
        raise ValueError(f"Unsupported objective: {objective!r}. Only 'f1' is supported.")

    probs = np.asarray(probs_matrix)
    gold = np.asarray(gold_matrix)

    if probs.shape[1] != NUM_LABELS or gold.shape[1] != NUM_LABELS:
        raise ValueError(
            f"Expected {NUM_LABELS} columns, got probs={probs.shape[1]}, gold={gold.shape[1]}"
        )

    thresholds_range = np.arange(0.05, 0.96, 0.01)
    best_thresholds: dict[str, float] = {}

    for label_idx, label_name in enumerate(LABEL_NAMES):
        y_true = gold[:, label_idx]
        y_prob = probs[:, label_idx]

        best_f1 = -1.0
        best_t = 0.5

        for t in thresholds_range:
            y_pred = (y_prob >= t).astype(int)
            f1 = f1_score(y_true, y_pred, zero_division=0.0)
            if f1 > best_f1:
                best_f1 = f1
                best_t = float(t)

        best_thresholds[label_name] = round(best_t, 2)

    return best_thresholds


def tune_and_save(
    probs_matrix: list[list[float]] | np.ndarray,
    gold_matrix: list[list[int]] | np.ndarray,
    split: str,
    output_path: Path,
    objective: str = "f1",
) -> dict:
    """Tune thresholds and write results to a JSON file.

    Args:
        probs_matrix: Predicted probabilities.
        gold_matrix:  Gold labels.
        split:        Must be "val".
        output_path:  Where to write the thresholds JSON.
        objective:    Optimization target.

    Returns:
        Full output dict (thresholds + metadata).
    """
    thresholds = tune_thresholds(probs_matrix, gold_matrix, split, objective)

    # Compute val F1 at the chosen thresholds for documentation
    probs = np.asarray(probs_matrix)
    gold = np.asarray(gold_matrix)
    val_metrics: dict[str, float] = {}
    for label_idx, label_name in enumerate(LABEL_NAMES):
        y_true = gold[:, label_idx]
        y_pred = (probs[:, label_idx] >= thresholds[label_name]).astype(int)
        val_metrics[f"f1_{label_name}"] = float(f1_score(y_true, y_pred, zero_division=0.0))

    output = {
        "thresholds": thresholds,
        "objective": objective,
        "split": split,
        "val_metrics": val_metrics,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as fh:
        json.dump(output, fh, indent=2)

    return output
