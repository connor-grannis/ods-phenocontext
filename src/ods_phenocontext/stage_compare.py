"""
Stage comparison report generator.

Reads metrics.json files from multiple experiment directories and produces
a Markdown summary with per-label F1 deltas, source coverage shifts, and
abstention rate changes across stages.
"""

from __future__ import annotations

import json
from pathlib import Path

from ods_phenocontext.schema import LABEL_NAMES


def load_experiment_metrics(exp_dir: Path) -> dict:
    """Load metrics from an experiment directory's metrics.json."""
    metrics_path = exp_dir / "metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(f"No metrics.json found in {exp_dir}")
    data = json.loads(metrics_path.read_text())
    return data


def compare_stages(
    experiment_dirs: dict[str, Path],
    baseline_stage: str,
) -> dict:
    """Compute per-label deltas and coverage shifts relative to a baseline stage.

    Args:
        experiment_dirs: Mapping of stage_name → experiment directory path.
        baseline_stage:  The stage name to use as the delta baseline.

    Returns:
        Dict with keys:
          stages:    Per-stage metrics dicts.
          deltas:    Per-stage deltas vs. baseline (macro_f1, per-label f1).
          coverage:  Per-stage source coverage counts.
    """
    if baseline_stage not in experiment_dirs:
        raise ValueError(f"baseline_stage {baseline_stage!r} not in experiment_dirs")

    stages: dict[str, dict] = {}
    for name, path in experiment_dirs.items():
        data = load_experiment_metrics(path)
        stages[name] = data["metrics"]

    baseline_metrics = stages[baseline_stage]
    deltas: dict[str, dict] = {}

    for name, metrics in stages.items():
        if name == baseline_stage:
            continue
        stage_deltas: dict[str, float] = {}
        stage_deltas["macro_f1"] = metrics.get("macro_f1", 0.0) - baseline_metrics.get(
            "macro_f1", 0.0
        )
        for label in LABEL_NAMES:
            key = f"f1_{label}"
            stage_deltas[key] = metrics.get(key, 0.0) - baseline_metrics.get(key, 0.0)
        deltas[name] = stage_deltas

    coverage: dict[str, dict] = {}
    for name, metrics in stages.items():
        sc = metrics.get("source_coverage", {})
        coverage[name] = {source: info.get("count", 0) for source, info in sc.items()}

    return {"stages": stages, "deltas": deltas, "coverage": coverage}


def render_markdown(
    comparison: dict,
    baseline_stage: str,
    experiment_dirs: dict[str, Path],
    output_path: Path | None = None,
) -> str:
    """Render a Markdown comparison report.

    Args:
        comparison:      Output of compare_stages().
        baseline_stage:  Name of the baseline stage (shown as reference).
        experiment_dirs: Mapping of stage_name → directory (for artifact links).
        output_path:     If given, write the report to this path.

    Returns:
        The rendered Markdown string.
    """
    stages = comparison["stages"]
    deltas = comparison["deltas"]
    coverage = comparison["coverage"]
    ordered = list(stages.keys())

    lines: list[str] = []
    lines.append("# Stage Comparison Report\n")
    lines.append(f"Baseline: **{baseline_stage}**\n")

    # --- Per-label F1 table ---
    lines.append("## Per-Label F1\n")
    header = "| Label | " + " | ".join(ordered) + " |"
    sep = "| --- | " + " | ".join(["---"] * len(ordered)) + " |"
    lines.append(header)
    lines.append(sep)

    for label in LABEL_NAMES:
        key = f"f1_{label}"
        row = f"| {label} |"
        for stage in ordered:
            val = stages[stage].get(key, 0.0)
            if stage == baseline_stage:
                row += f" {val:.3f} |"
            else:
                delta = deltas[stage].get(key, 0.0)
                sign = "+" if delta >= 0 else ""
                row += f" {val:.3f} ({sign}{delta:.3f}) |"
        lines.append(row)

    # macro F1 row
    row = "| **macro_f1** |"
    for stage in ordered:
        val = stages[stage].get("macro_f1", 0.0)
        if stage == baseline_stage:
            row += f" **{val:.3f}** |"
        else:
            delta = deltas[stage].get("macro_f1", 0.0)
            sign = "+" if delta >= 0 else ""
            row += f" **{val:.3f}** ({sign}{delta:.3f}) |"
    lines.append(row)
    lines.append("")

    # --- Source coverage table ---
    lines.append("## Source Coverage\n")
    sources = ["rules", "biobert", "none"]
    header = "| Source | " + " | ".join(ordered) + " |"
    sep = "| --- | " + " | ".join(["---"] * len(ordered)) + " |"
    lines.append(header)
    lines.append(sep)
    for source in sources:
        row = f"| {source} |"
        for stage in ordered:
            count = coverage.get(stage, {}).get(source, 0)
            row += f" {count} |"
        lines.append(row)

    # abstention rate row
    row = "| abstention_rate |"
    for stage in ordered:
        rate = stages[stage].get("abstention_rate", 0.0)
        row += f" {rate:.3f} |"
    lines.append(row)
    lines.append("")

    # --- Artifact links ---
    lines.append("## Experiment Artifacts\n")
    for stage, path in experiment_dirs.items():
        lines.append(f"- **{stage}**: `{path}`")
    lines.append("")

    report = "\n".join(lines)

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(report)

    return report
