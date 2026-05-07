"""
CLI: summarize LLM call costs from the audit log.

Usage:
    uv run python -m ods_phenocontext.audit.summarize_costs --since 2026-01-01
    uv run python -m ods_phenocontext.audit.summarize_costs --by role
    uv run python -m ods_phenocontext.audit.summarize_costs --by model --since 2026-05-01
"""

from __future__ import annotations

from pathlib import Path

import click

from ods_phenocontext.audit.llm_calls import LLMCallRecord, load_records

_DEFAULT_LOG = Path("audits/teacher_outputs/llm_call_log.jsonl")


def _filter_since(records: list[LLMCallRecord], since: str | None) -> list[LLMCallRecord]:
    if not since:
        return records
    return [r for r in records if r.timestamp >= since]


def _summarize(records: list[LLMCallRecord], by: str) -> dict[str, dict[str, float | int]]:
    """Aggregate records by 'role' or 'model'. Returns keyed totals."""
    totals: dict[str, dict[str, float | int]] = {}
    for r in records:
        key = r.teacher_role if by == "role" else r.model_id
        if key not in totals:
            totals[key] = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "usd_cost": 0.0}
        totals[key]["calls"] = int(totals[key]["calls"]) + 1
        totals[key]["prompt_tokens"] = int(totals[key]["prompt_tokens"]) + r.prompt_tokens
        totals[key]["completion_tokens"] = (
            int(totals[key]["completion_tokens"]) + r.completion_tokens
        )
        totals[key]["usd_cost"] = float(totals[key]["usd_cost"]) + r.usd_cost
    return totals


@click.command()
@click.option(
    "--log",
    "log_path",
    default=str(_DEFAULT_LOG),
    show_default=True,
    help="Path to llm_call_log.jsonl.",
)
@click.option("--since", default=None, help="ISO date filter, e.g. 2026-01-01.")
@click.option(
    "--by",
    default="role",
    show_default=True,
    type=click.Choice(["role", "model"]),
    help="Group totals by teacher role or model ID.",
)
def main(log_path: str, since: str | None, by: str) -> None:
    """Print aggregated LLM call costs from the audit log."""
    records = load_records(Path(log_path))
    records = _filter_since(records, since)

    if not records:
        click.echo("No records found.")
        return

    totals = _summarize(records, by)
    grand_usd = sum(float(v["usd_cost"]) for v in totals.values())

    header = f"{'Key':<35} {'Calls':>6} {'Prompt tok':>12} {'Completion tok':>15} {'USD':>10}"
    click.echo(header)
    click.echo("-" * len(header))
    for key, v in sorted(totals.items()):
        click.echo(
            f"{key:<35} {v['calls']:>6} {v['prompt_tokens']:>12} {v['completion_tokens']:>15} "
            f"${float(v['usd_cost']):>9.4f}"
        )
    click.echo("-" * len(header))
    click.echo(f"{'TOTAL':<35} {'':>6} {'':>12} {'':>15} ${grand_usd:>9.4f}")


if __name__ == "__main__":
    main()
