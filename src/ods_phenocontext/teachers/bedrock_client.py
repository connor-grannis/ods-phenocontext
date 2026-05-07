"""
Thin wrapper around langchain_aws.ChatBedrock with structured output.

All imports from the teacher group (langchain_aws, boto3, etc.) are gated
behind a try/except so that production code — which does not install the
teacher group — can import this module without error.  Attempting to
*use* any function here without the teacher group installed will raise
a clear ImportError at call time, not at module load time.

Teacher roles and their Bedrock model IDs are passed in at construction;
the caller is responsible for building the role-specific system prompts.
See PROJECT_OVERVIEW.md §Teacher Committee for role descriptions.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

if TYPE_CHECKING:
    # These are only resolved during type-checking, not at runtime, so they
    # never cause an ImportError in production.
    from langchain_aws import ChatBedrock
    from langchain_core.runnables import Runnable

# Attempt the real runtime import; fail gracefully so production survives.
try:
    from langchain_aws import ChatBedrock as _ChatBedrock
    from langchain_core.messages import HumanMessage as _HumanMessage
    from langchain_core.messages import SystemMessage as _SystemMessage
    from langchain_core.runnables import RunnableLambda as _RunnableLambda

    _TEACHER_DEPS_AVAILABLE = True
except ImportError:
    _TEACHER_DEPS_AVAILABLE = False


# ---------------------------------------------------------------------------
# Structured output schema (mirrors teacher contract from PROJECT_OVERVIEW.md)
# ---------------------------------------------------------------------------


class TeacherOutput(BaseModel):
    """Pydantic schema for a single teacher's structured response."""

    # Multi-hot label vector matching the 4-class ontology:
    # [confirmed, negated, associated_with_someone_else, other_non_patient]
    labels: list[int]
    rationale: str
    evidence_spans: list[str]
    # Coarse confidence bucket — used downstream in aggregation weighting
    confidence_bin: str  # "high" | "medium" | "low"


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def build_teacher(
    model_id: str,
    system_prompt: str,
    region_name: str = "us-east-2",
    temperature: float = 0.0,
    **model_kwargs: Any,
) -> Runnable[Any, TeacherOutput]:
    """
    Return a LangChain runnable that sends a chat message to Bedrock and
    returns a structured TeacherOutput.

    Args:
        model_id:      Bedrock cross-region inference profile ID,
                       e.g. "us.anthropic.claude-sonnet-4-6".
        system_prompt: Role-specific instructions injected as a SystemMessage.
        region_name:   AWS region where Bedrock access is provisioned.
        temperature:   Sampling temperature; 0.0 for deterministic labels.
        **model_kwargs: Passed through to ChatBedrock (e.g. max_tokens).

    Returns:
        A runnable that accepts a HumanMessage (or string) and returns a
        TeacherOutput validated by Pydantic.

    Raises:
        ImportError: if the teacher dependency group is not installed.
    """
    if not _TEACHER_DEPS_AVAILABLE:
        raise ImportError("Teacher dependencies are not installed. Run: uv sync --group teacher")

    # langchain-aws stubs are incomplete; model_id/region_name are valid runtime
    # fields (confirmed via ChatBedrock.model_fields) but missing from type stubs.
    llm: ChatBedrock = _ChatBedrock(  # type: ignore[call-arg]
        model_id=model_id,
        region_name=region_name,
        temperature=temperature,
        **model_kwargs,
    )

    # with_structured_output wraps the LLM to parse/validate the response
    # into TeacherOutput via Pydantic.
    structured_llm: Runnable[Any, TeacherOutput] = llm.with_structured_output(TeacherOutput)  # type: ignore[assignment]

    # Prepend the system prompt so each teacher has its role baked in.
    # We use a RunnableLambda to inject the SystemMessage before the user
    # message rather than piping a raw SystemMessage (not supported in LC 1.x).
    system = _SystemMessage(content=system_prompt)

    def _prepend_system(message: Any) -> list[Any]:
        # Accept a HumanMessage, a string, or an already-built list.
        if isinstance(message, list):
            return [system] + message
        return [system, _HumanMessage(content=message) if isinstance(message, str) else message]

    return _RunnableLambda(_prepend_system) | structured_llm  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Default teacher committee
# ---------------------------------------------------------------------------

# Bedrock cross-region inference profile (us-east-2, enabled in AWS console)
_DEFAULT_MODEL_ID = "us.anthropic.claude-sonnet-4-6"
_DEFAULT_REGION = "us-east-2"

# Role-specific system prompts.  Temperatures follow PROJECT_OVERVIEW.md
# guidance: precision-biased is coldest, recall-biased is warmest.
TEACHER_CONFIGS: dict[str, dict[str, Any]] = {
    "generalist": {
        "model_id": _DEFAULT_MODEL_ID,
        "region_name": _DEFAULT_REGION,
        "temperature": 0.0,
        "system_prompt": (
            "You are a clinical NLP expert labeling phenotype mention context. "
            "For each input, assign the most accurate multi-hot labels from: "
            "[confirmed, negated, associated_with_someone_else, other_non_patient]. "
            "Be balanced — do not bias toward over- or under-labeling."
        ),
    },
    "precision_biased": {
        "model_id": _DEFAULT_MODEL_ID,
        "region_name": _DEFAULT_REGION,
        "temperature": 0.0,
        # Prefer fewer false positives: only label what is clearly supported
        "system_prompt": (
            "You are a conservative clinical NLP labeler. "
            "Only assign a label when the evidence is unambiguous. "
            "When in doubt, omit the label rather than risk a false positive. "
            "Labels: [confirmed, negated, associated_with_someone_else, other_non_patient]."
        ),
    },
    "recall_biased": {
        "model_id": _DEFAULT_MODEL_ID,
        "region_name": _DEFAULT_REGION,
        "temperature": 0.3,
        # Prefer fewer false negatives: flag any plausible label
        "system_prompt": (
            "You are a sensitive clinical NLP labeler. "
            "Assign a label whenever there is reasonable evidence, even if not "
            "fully explicit. Prefer not to miss a label. "
            "Labels: [confirmed, negated, associated_with_someone_else, other_non_patient]."
        ),
    },
}


def build_committee() -> dict[str, Runnable[Any, TeacherOutput]]:
    """
    Instantiate all teachers in TEACHER_CONFIGS.

    Returns a dict mapping role name → structured-output runnable.
    Raises ImportError if teacher deps are not installed.
    """
    return {
        role: build_teacher(
            model_id=cfg["model_id"],
            system_prompt=cfg["system_prompt"],
            region_name=cfg["region_name"],
            temperature=cfg["temperature"],
        )
        for role, cfg in TEACHER_CONFIGS.items()
    }
