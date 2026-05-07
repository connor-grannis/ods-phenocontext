"""
Evaluation utilities for PhenoContext.

compute_metrics: per-label P/R/F1, micro/macro F1, PR-AUC, source coverage,
and per-label confusion counts from a list of predicted Instances.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from sklearn.metrics import (
    auc,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
)

from ods_phenocontext.schema import LABEL_NAMES, NUM_LABELS, Instance


def compute_metrics(instances: list[Instance]) -> dict:
    """Compute evaluation metrics from instances that have gold_labels and predictions.

    Predictions are taken from the "final" labels — rule_labels if not abstained,
    biobert_labels otherwise.  Probabilities (for PR-AUC) come from rule_probs or
    biobert_probs accordingly.

    Args:
        instances: List of Instances with gold_labels and prediction fields populated.

    Returns:
        Dict with per-label and aggregate metrics.
    """
    gold_matrix = []
    pred_matrix = []
    prob_matrix = []
    sources: dict[str, int] = {"rules": 0, "biobert": 0, "none": 0}

    for inst in instances:
        if inst.gold_labels is None:
            continue

        gold_matrix.append(inst.gold_labels)

        if not inst.rule_abstained and inst.rule_labels is not None:
            pred_matrix.append(inst.rule_labels)
            prob_matrix.append(inst.rule_probs or [0.0] * NUM_LABELS)
            sources["rules"] += 1
        elif inst.biobert_labels is not None:
            pred_matrix.append(inst.biobert_labels)
            prob_matrix.append(inst.biobert_probs or [0.0] * NUM_LABELS)
            sources["biobert"] += 1
        else:
            pred_matrix.append([0] * NUM_LABELS)
            prob_matrix.append([0.0] * NUM_LABELS)
            sources["none"] += 1

    if not gold_matrix:
        return {}

    gold_arr = np.array(gold_matrix)
    pred_arr = np.array(pred_matrix)
    prob_arr = np.array(prob_matrix)

    metrics: dict = {}

    # Per-label metrics
    for i, label_name in enumerate(LABEL_NAMES):
        y_true = gold_arr[:, i]
        y_pred = pred_arr[:, i]
        y_prob = prob_arr[:, i]

        tp = int(((y_true == 1) & (y_pred == 1)).sum())
        fp = int(((y_true == 0) & (y_pred == 1)).sum())
        fn = int(((y_true == 1) & (y_pred == 0)).sum())
        tn = int(((y_true == 0) & (y_pred == 0)).sum())

        p = precision_score(y_true, y_pred, zero_division=0.0)
        r = recall_score(y_true, y_pred, zero_division=0.0)
        f1 = f1_score(y_true, y_pred, zero_division=0.0)

        metrics[f"precision_{label_name}"] = float(p)
        metrics[f"recall_{label_name}"] = float(r)
        metrics[f"f1_{label_name}"] = float(f1)
        metrics[f"tp_{label_name}"] = tp
        metrics[f"fp_{label_name}"] = fp
        metrics[f"fn_{label_name}"] = fn
        metrics[f"tn_{label_name}"] = tn

        # PR-AUC (only if there's at least one positive example)
        if y_true.sum() > 0 and len(np.unique(y_prob)) > 1:
            prec_curve, rec_curve, _ = precision_recall_curve(y_true, y_prob)
            metrics[f"pr_auc_{label_name}"] = float(auc(rec_curve, prec_curve))

    # Micro/macro F1
    metrics["micro_f1"] = float(f1_score(gold_arr.ravel(), pred_arr.ravel(), zero_division=0.0))
    per_label_f1 = [metrics[f"f1_{name}"] for name in LABEL_NAMES]
    metrics["macro_f1"] = float(np.mean(per_label_f1))

    # Source coverage
    total = sum(sources.values())
    metrics["source_coverage"] = {
        k: {"count": v, "fraction": v / total if total else 0.0} for k, v in sources.items()
    }
    metrics["n_evaluated"] = total

    return metrics


def slice_metrics(
    instances: list[Instance],
    slice_fn: Callable[[Instance], str],
) -> dict[str, dict]:
    """Compute metrics per slice (e.g. by context length, department).

    Args:
        instances: Evaluated Instances with predictions.
        slice_fn:  Function mapping Instance → slice name string.

    Returns:
        Dict mapping slice_name → metrics dict.
    """
    slices: dict[str, list[Instance]] = {}
    for inst in instances:
        key = slice_fn(inst)
        slices.setdefault(key, []).append(inst)

    return {name: compute_metrics(insts) for name, insts in slices.items()}
