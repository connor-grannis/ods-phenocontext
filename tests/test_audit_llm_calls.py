"""
Tests for M0: LLM call audit (pricing, LLMCostLogger, summarize_costs CLI).

Callback tests (TestLLMCostLogger) build a fake LLMResult without importing
langchain, so they always run.  The summarize_costs CLI test uses Click's
test runner and only needs the audit module.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock
from uuid import uuid4

import pytest
from click.testing import CliRunner

from ods_phenocontext.audit.llm_calls import LLMCallRecord, LLMCostLogger, load_records
from ods_phenocontext.audit.pricing import BEDROCK_PRICING, compute_usd
from ods_phenocontext.audit.summarize_costs import main as summarize_main

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

MODEL_ID = "us.anthropic.claude-sonnet-4-6"
ROLE = "generalist"


def _fake_llm_result(input_tokens: int = 100, output_tokens: int = 50) -> MagicMock:
    """Build a MagicMock that mimics langchain LLMResult with usage metadata."""
    result = MagicMock()
    result.llm_output = {
        "usage": {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
        }
    }
    result.generations = []
    return result


# ---------------------------------------------------------------------------
# Pricing
# ---------------------------------------------------------------------------


class TestPricing:
    def test_known_model_in_table(self):
        assert MODEL_ID in BEDROCK_PRICING

    def test_pricing_fields_present(self):
        p = BEDROCK_PRICING[MODEL_ID]
        assert "input_per_1k" in p
        assert "output_per_1k" in p
        assert "priced_on" in p

    def test_compute_usd_nonzero(self):
        cost = compute_usd(MODEL_ID, input_tokens=1000, output_tokens=500)
        assert cost > 0

    def test_compute_usd_zero_tokens(self):
        assert compute_usd(MODEL_ID, 0, 0) == 0.0

    def test_compute_usd_math(self):
        p = BEDROCK_PRICING[MODEL_ID]
        expected = 1000 / 1000 * float(p["input_per_1k"]) + 500 / 1000 * float(p["output_per_1k"])
        assert compute_usd(MODEL_ID, 1000, 500) == pytest.approx(expected)

    def test_unknown_model_raises(self):
        with pytest.raises(KeyError):
            compute_usd("non-existent-model", 100, 50)


# ---------------------------------------------------------------------------
# LLMCostLogger
# ---------------------------------------------------------------------------


class TestLLMCostLogger:
    def test_record_written_on_llm_end(self, tmp_path: Path):
        log = tmp_path / "calls.jsonl"
        logger = LLMCostLogger(log, model_id=MODEL_ID, region="us-east-2", teacher_role=ROLE)
        run_id = uuid4()

        logger.on_chat_model_start({}, [[]], run_id=run_id)
        logger.on_llm_end(_fake_llm_result(100, 50), run_id=run_id)

        lines = log.read_text().strip().splitlines()
        assert len(lines) == 1
        record = LLMCallRecord.model_validate_json(lines[0])
        assert record.model_id == MODEL_ID
        assert record.teacher_role == ROLE
        assert record.prompt_tokens == 100
        assert record.completion_tokens == 50
        assert record.total_tokens == 150

    def test_usd_cost_nonzero(self, tmp_path: Path):
        log = tmp_path / "calls.jsonl"
        logger = LLMCostLogger(log, model_id=MODEL_ID, region="us-east-2", teacher_role=ROLE)
        run_id = uuid4()

        logger.on_llm_start({}, [], run_id=run_id)
        logger.on_llm_end(_fake_llm_result(1000, 500), run_id=run_id)

        record = load_records(log)[0]
        assert record.usd_cost > 0

    def test_multiple_calls_append(self, tmp_path: Path):
        log = tmp_path / "calls.jsonl"
        logger = LLMCostLogger(log, model_id=MODEL_ID, region="us-east-2", teacher_role=ROLE)

        for _ in range(3):
            run_id = uuid4()
            logger.on_chat_model_start({}, [[]], run_id=run_id)
            logger.on_llm_end(_fake_llm_result(), run_id=run_id)

        assert len(load_records(log)) == 3

    def test_unknown_model_writes_zero_cost(self, tmp_path: Path):
        log = tmp_path / "calls.jsonl"
        logger = LLMCostLogger(log, model_id="unknown-model", region="us-east-2", teacher_role=ROLE)
        run_id = uuid4()
        logger.on_chat_model_start({}, [[]], run_id=run_id)
        logger.on_llm_end(_fake_llm_result(), run_id=run_id)

        record = load_records(log)[0]
        assert record.usd_cost == 0.0

    def test_record_id_matches_run_id(self, tmp_path: Path):
        log = tmp_path / "calls.jsonl"
        logger = LLMCostLogger(log, model_id=MODEL_ID, region="us-east-2", teacher_role=ROLE)
        run_id = uuid4()

        logger.on_chat_model_start({}, [[]], run_id=run_id)
        logger.on_llm_end(_fake_llm_result(), run_id=run_id)

        record = load_records(log)[0]
        assert record.request_id == str(run_id)

    def test_latency_recorded(self, tmp_path: Path):
        log = tmp_path / "calls.jsonl"
        logger = LLMCostLogger(log, model_id=MODEL_ID, region="us-east-2", teacher_role=ROLE)
        run_id = uuid4()

        logger.on_chat_model_start({}, [[]], run_id=run_id)
        logger.on_llm_end(_fake_llm_result(), run_id=run_id)

        record = load_records(log)[0]
        assert record.latency_ms >= 0.0


# ---------------------------------------------------------------------------
# load_records
# ---------------------------------------------------------------------------


class TestLoadRecords:
    def test_empty_file_returns_empty_list(self, tmp_path: Path):
        log = tmp_path / "empty.jsonl"
        log.touch()
        assert load_records(log) == []

    def test_missing_file_returns_empty_list(self, tmp_path: Path):
        assert load_records(tmp_path / "nonexistent.jsonl") == []


# ---------------------------------------------------------------------------
# summarize_costs CLI
# ---------------------------------------------------------------------------


def _write_records(path: Path, records: list[dict]) -> None:
    with path.open("w") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")


def _base_record(**overrides) -> dict:
    base = {
        "timestamp": "2026-05-07T12:00:00+00:00",
        "model_id": MODEL_ID,
        "region": "us-east-2",
        "teacher_role": "generalist",
        "prompt_tokens": 100,
        "completion_tokens": 50,
        "total_tokens": 150,
        "latency_ms": 200.0,
        "usd_cost": 0.0075,
        "request_id": str(uuid4()),
        "prompt_version": "v1",
        "instance_id": "",
    }
    base.update(overrides)
    return base


class TestSummarizeCosts:
    def test_no_records(self, tmp_path: Path):
        log = tmp_path / "empty.jsonl"
        log.touch()
        result = CliRunner().invoke(summarize_main, ["--log", str(log)])
        assert result.exit_code == 0
        assert "No records found" in result.output

    def test_totals_printed(self, tmp_path: Path):
        log = tmp_path / "calls.jsonl"
        _write_records(
            log,
            [
                _base_record(
                    teacher_role="generalist",
                    usd_cost=0.01,
                    prompt_tokens=200,
                    completion_tokens=100,
                ),
                _base_record(
                    teacher_role="generalist",
                    usd_cost=0.005,
                    prompt_tokens=100,
                    completion_tokens=50,
                ),
            ],
        )
        result = CliRunner().invoke(summarize_main, ["--log", str(log)])
        assert result.exit_code == 0
        assert "generalist" in result.output
        assert "TOTAL" in result.output

    def test_by_model_grouping(self, tmp_path: Path):
        log = tmp_path / "calls.jsonl"
        _write_records(
            log,
            [
                _base_record(model_id=MODEL_ID),
                _base_record(model_id="other-model", usd_cost=0.0),
            ],
        )
        result = CliRunner().invoke(summarize_main, ["--log", str(log), "--by", "model"])
        assert result.exit_code == 0
        assert MODEL_ID in result.output

    def test_since_filter(self, tmp_path: Path):
        log = tmp_path / "calls.jsonl"
        _write_records(
            log,
            [
                _base_record(timestamp="2026-01-01T00:00:00+00:00"),
                _base_record(timestamp="2026-06-01T00:00:00+00:00"),
            ],
        )
        result = CliRunner().invoke(summarize_main, ["--log", str(log), "--since", "2026-05-01"])
        assert result.exit_code == 0
        # Only 1 record passes the filter; calls column should show 1
        assert " 1 " in result.output

    def test_grand_total_matches_sum(self, tmp_path: Path):
        log = tmp_path / "calls.jsonl"
        costs = [0.01, 0.02, 0.005]
        _write_records(log, [_base_record(usd_cost=c) for c in costs])
        result = CliRunner().invoke(summarize_main, ["--log", str(log)])
        assert result.exit_code == 0
        # Grand total = 0.0350
        assert "0.0350" in result.output
