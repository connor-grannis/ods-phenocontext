"""
LLM call audit logging for Bedrock teacher calls.

LLMCostLogger is a LangChain BaseCallbackHandler that writes one JSONL
record per completed LLM/chat-model call to a configurable log file.
Attach it via the `callbacks=[...]` argument when constructing ChatBedrock.
"""

from __future__ import annotations

import time
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any
from uuid import UUID

from pydantic import BaseModel

from ods_phenocontext.audit.pricing import BEDROCK_PRICING, compute_usd

if TYPE_CHECKING:
    from langchain_core.outputs import LLMResult


class LLMCallRecord(BaseModel):
    """One audited Bedrock call."""

    timestamp: str  # ISO-8601 UTC
    model_id: str
    region: str
    teacher_role: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    latency_ms: float
    usd_cost: float
    request_id: str  # run_id from LangChain
    prompt_version: str
    instance_id: str  # empty string when not set


class LLMCostLogger:
    """LangChain callback handler that audits every Bedrock LLM/chat-model call.

    Writes one JSON line per call to `log_path`.  Appends to the file so
    records accumulate across runs.

    Conforms to the BaseCallbackHandler interface without inheriting from it —
    LangChain accepts any object with the expected hook methods.

    Args:
        log_path:       Destination JSONL file. Parent directory must exist.
        model_id:       Bedrock model ID (used for cost lookup).
        region:         AWS region of the Bedrock endpoint.
        teacher_role:   Role name (e.g. "generalist", "precision_biased").
        prompt_version: Version string for the active prompt template.
        instance_id:    Instance being annotated (set per-call if needed).
    """

    def __init__(
        self,
        log_path: Path,
        model_id: str,
        region: str,
        teacher_role: str,
        prompt_version: str = "unknown",
        instance_id: str = "",
    ) -> None:
        self.log_path = log_path
        self.model_id = model_id
        self.region = region
        self.teacher_role = teacher_role
        self.prompt_version = prompt_version
        self.instance_id = instance_id
        # Keyed by run_id; records start time at on_llm_start / on_chat_model_start
        self._start_times: dict[str, float] = {}

    # ------------------------------------------------------------------
    # LangChain callback hooks
    # ------------------------------------------------------------------

    def on_llm_start(
        self, serialized: dict[str, Any], prompts: list[str], run_id: UUID, **kwargs: Any
    ) -> None:
        self._start_times[str(run_id)] = time.monotonic()

    def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: list[list[Any]],
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        self._start_times[str(run_id)] = time.monotonic()

    def on_llm_end(self, response: LLMResult, run_id: UUID, **kwargs: Any) -> None:
        run_key = str(run_id)
        start = self._start_times.pop(run_key, time.monotonic())
        latency_ms = (time.monotonic() - start) * 1000

        usage = self._extract_usage(response)
        prompt_tokens = usage.get("input_tokens", usage.get("prompt_tokens", 0))
        completion_tokens = usage.get("output_tokens", usage.get("completion_tokens", 0))
        total_tokens = usage.get("total_tokens", prompt_tokens + completion_tokens)

        usd_cost = 0.0
        if self.model_id in BEDROCK_PRICING:
            usd_cost = compute_usd(self.model_id, prompt_tokens, completion_tokens)

        record = LLMCallRecord(
            timestamp=datetime.now(UTC).isoformat(),
            model_id=self.model_id,
            region=self.region,
            teacher_role=self.teacher_role,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            latency_ms=latency_ms,
            usd_cost=usd_cost,
            request_id=run_key,
            prompt_version=self.prompt_version,
            instance_id=self.instance_id,
        )
        with self.log_path.open("a") as fh:
            fh.write(record.model_dump_json() + "\n")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_usage(response: LLMResult) -> dict[str, int]:
        """Pull token counts from LLMResult regardless of nesting style."""
        # langchain-aws puts usage in llm_output or generation metadata
        if response.llm_output:
            meta = response.llm_output.get("usage", response.llm_output)
            if isinstance(meta, dict) and any(k in meta for k in ("input_tokens", "prompt_tokens")):
                return meta  # type: ignore[return-value]

        # Fall back to first generation's response_metadata
        for gen_list in response.generations:
            for gen in gen_list:
                rm = getattr(gen, "generation_info", None) or {}
                usage = rm.get("usage", rm.get("response_metadata", {}).get("usage", {}))
                if usage and isinstance(usage, dict):
                    return usage  # type: ignore[return-value]

        return {}


def load_records(log_path: Path) -> list[LLMCallRecord]:
    """Read all records from a JSONL log file."""
    if not log_path.exists():
        return []
    records = []
    with log_path.open() as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(LLMCallRecord.model_validate_json(line))
    return records
