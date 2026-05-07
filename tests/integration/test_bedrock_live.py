"""
Live Bedrock round-trip test.  Skipped unless RUN_BEDROCK_INTEGRATION=1 is set.

Run manually:
    RUN_BEDROCK_INTEGRATION=1 uv run --group teacher pytest tests/integration/ -v
"""

import os

import pytest

# Skip the entire module unless the integration flag is set.
# This prevents accidental network calls and AWS spend in CI.
pytestmark = pytest.mark.skipif(
    os.getenv("RUN_BEDROCK_INTEGRATION") != "1",
    reason="Set RUN_BEDROCK_INTEGRATION=1 to run live Bedrock tests",
)


def test_bedrock_committee_round_trip() -> None:
    """Each teacher in the committee should return a valid TeacherOutput."""
    from langchain_core.messages import HumanMessage

    from ods_phenocontext.teachers.bedrock_client import TeacherOutput, build_committee

    committee = build_committee()
    assert len(committee) >= 3, "Expected at least 3 teacher roles"

    # Synthetic note snippet — no PHI.
    test_input = HumanMessage(
        content=(
            "Context: 'The patient denies any history of asthma.' "
            "Entity: asthma"
        )
    )

    for role, teacher in committee.items():
        result = teacher.invoke(test_input)
        assert isinstance(result, TeacherOutput), f"{role} did not return TeacherOutput"
        assert isinstance(result.labels, list), f"{role}: labels must be a list"
        assert all(x in (0, 1) for x in result.labels), f"{role}: labels must be 0/1"
        assert result.confidence_bin in ("high", "medium", "low"), (
            f"{role}: invalid confidence_bin '{result.confidence_bin}'"
        )
