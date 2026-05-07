"""
Experiment runner for PhenoContext baselines.

Orchestrates a single experiment run: loads instances, runs the pipeline on
the eval split, computes metrics, and writes structured artifacts to
experiments/<name>/.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from ods_phenocontext.evaluate import compute_metrics
from ods_phenocontext.pipeline import BioBERTModel, RulesModel, phenocontext_predict
from ods_phenocontext.schema import LABEL_NAMES, NUM_LABELS, Instance

VALID_STAGES = frozenset(
    {
        "rules_only",
        "biobert_only",
        "rules_then_biobert_default",
        "rules_then_biobert_tuned",
    }
)


@dataclass
class ExperimentConfig:
    """Configuration for a single experiment run."""

    name: str
    stage: str
    eval_split: str = "val"
    thresholds: dict[str, float] | None = None
    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.stage not in VALID_STAGES:
            raise ValueError(f"stage must be one of {sorted(VALID_STAGES)!r}, got {self.stage!r}")
        if self.eval_split not in ("val", "test"):
            raise ValueError(f"eval_split must be 'val' or 'test', got {self.eval_split!r}")

    @property
    def threshold_list(self) -> list[float]:
        """Return thresholds as an ordered list matching LABEL_NAMES."""
        if self.thresholds is None:
            return [0.5] * NUM_LABELS
        return [self.thresholds[name] for name in LABEL_NAMES]


@dataclass
class ExperimentResult:
    """Output of a completed experiment run."""

    config: ExperimentConfig
    metrics: dict
    predictions: list[dict]
    timestamp: str = field(default_factory=lambda: datetime.now(UTC).isoformat())


class _NullBioBERT:
    """Stub — should never be called; exists only to satisfy type signatures."""

    def predict_proba(self, instance: Instance) -> list[float]:
        raise RuntimeError("BioBERT called in a stage that should not use it")


class _NullRules:
    """Stub that always abstains — used for biobert_only stage."""

    def __call__(self, instance: Instance) -> dict:
        return {
            "abstained": True,
            "labels": [0] * NUM_LABELS,
            "probs": [0.0] * NUM_LABELS,
            "rule_ids": [],
        }


def run_experiment(
    config: ExperimentConfig,
    instances: list[Instance],
    rules_model: RulesModel | None = None,
    biobert_model: BioBERTModel | None = None,
) -> ExperimentResult:
    """Execute a single experiment run.

    Args:
        config:       Experiment configuration.
        instances:    All instances (will be filtered to config.eval_split).
        rules_model:  Rule classifier. Required unless stage is biobert_only.
        biobert_model: BioBERT predictor. Required unless stage is rules_only.

    Returns:
        ExperimentResult with metrics and per-instance predictions.
    """
    eval_instances = [i for i in instances if i.split == config.eval_split]
    if not eval_instances:
        raise ValueError(f"No instances found with split={config.eval_split!r}")

    # Wire up models based on stage
    if config.stage == "rules_only":
        if rules_model is None:
            raise ValueError("rules_model is required for rules_only stage")
        effective_rules: RulesModel = rules_model
        effective_biobert: BioBERTModel = _NullBioBERT()
    elif config.stage == "biobert_only":
        if biobert_model is None:
            raise ValueError("biobert_model is required for biobert_only stage")
        effective_rules = _NullRules()
        effective_biobert = biobert_model
    else:
        if rules_model is None:
            raise ValueError(f"rules_model is required for {config.stage} stage")
        if biobert_model is None:
            raise ValueError(f"biobert_model is required for {config.stage} stage")
        effective_rules = rules_model
        effective_biobert = biobert_model

    thresholds = config.threshold_list
    predictions: list[dict] = []
    rule_id_counts: dict[str, int] = {}
    n_abstained = 0

    for inst in eval_instances:
        if config.stage == "rules_only":
            # Run rules directly; treat abstentions as no-prediction (all zeros)
            rule_output = effective_rules(inst)
            inst.rule_abstained = rule_output["abstained"]
            if rule_output["abstained"]:
                n_abstained += 1
                inst.rule_labels = [0] * NUM_LABELS
                inst.rule_probs = [0.0] * NUM_LABELS
            else:
                inst.rule_labels = rule_output["labels"]
                inst.rule_probs = rule_output["probs"]
            for rid in rule_output.get("rule_ids", []):
                rule_id_counts[rid] = rule_id_counts.get(rid, 0) + 1
            predictions.append(
                {
                    "instance_id": inst.instance_id,
                    "source": "rules",
                    "abstained": rule_output["abstained"],
                    "labels": inst.rule_labels,
                    "probs": inst.rule_probs,
                    "rule_ids": rule_output.get("rule_ids", []),
                    "gold_labels": inst.gold_labels,
                }
            )
        else:
            result = phenocontext_predict(
                instance=inst,
                rules_model=effective_rules,
                biobert_model=effective_biobert,
                thresholds=thresholds,
            )
            if result["source"] == "rules":
                inst.rule_labels = result["labels"]
                inst.rule_probs = result["probs"]
                inst.rule_abstained = False
                for rid in result.get("rule_ids", []):
                    rule_id_counts[rid] = rule_id_counts.get(rid, 0) + 1
            else:
                inst.rule_abstained = True
                n_abstained += 1
                inst.biobert_labels = result["labels"]
                inst.biobert_probs = result["probs"]
            predictions.append(
                {
                    "instance_id": inst.instance_id,
                    "source": result["source"],
                    "labels": result["labels"],
                    "probs": result["probs"],
                    "gold_labels": inst.gold_labels,
                }
            )

    metrics = compute_metrics(eval_instances)
    n_total = len(eval_instances)
    metrics["abstention_rate"] = n_abstained / n_total if n_total else 0.0
    metrics["rule_id_counts"] = rule_id_counts

    return ExperimentResult(
        config=config,
        metrics=metrics,
        predictions=predictions,
    )


def save_experiment(result: ExperimentResult, base_dir: Path) -> Path:
    """Write experiment artifacts to base_dir/<name>/.

    Creates:
        - config.json
        - metrics.json
        - predictions.jsonl

    Returns:
        Path to the experiment directory.
    """
    exp_dir = base_dir / result.config.name
    exp_dir.mkdir(parents=True, exist_ok=True)

    # config.json
    config_data = asdict(result.config)
    (exp_dir / "config.json").write_text(json.dumps(config_data, indent=2))

    # metrics.json
    metrics_data = {
        "timestamp": result.timestamp,
        "stage": result.config.stage,
        "eval_split": result.config.eval_split,
        "metrics": result.metrics,
    }
    (exp_dir / "metrics.json").write_text(json.dumps(metrics_data, indent=2))

    # predictions.jsonl
    with (exp_dir / "predictions.jsonl").open("w") as fh:
        for pred in result.predictions:
            fh.write(json.dumps(pred) + "\n")

    return exp_dir
