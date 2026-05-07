"""Tests for the rule-based phenotype context classifier."""

import pytest

from ods_phenocontext.rules import RuleClassifier
from ods_phenocontext.schema import NUM_LABELS, Instance


def _inst(context: str, entity: str = "diabetes") -> Instance:
    return Instance(
        instance_id="t-001",
        note_id="n-001",
        entity_text=entity,
        context_window=context,
        split="test",
    )


@pytest.fixture
def clf() -> RuleClassifier:
    return RuleClassifier()


# ── Confirmed (no triggers) ─────────────────────────────────────────────────


class TestConfirmed:
    def test_simple_affirmation(self, clf: RuleClassifier):
        result = clf(_inst("The patient has diabetes."))
        assert result["labels"] == [1, 0, 0, 0]
        assert not result["abstained"]

    def test_diagnosis_statement(self, clf: RuleClassifier):
        result = clf(_inst("Assessment: diabetes mellitus type 2."))
        assert result["labels"] == [1, 0, 0, 0]


# ── Negation ────────────────────────────────────────────────────────────────


class TestNegation:
    def test_denies(self, clf: RuleClassifier):
        result = clf(_inst("Patient denies diabetes."))
        assert result["labels"] == [0, 1, 0, 0]
        assert not result["abstained"]

    def test_marker_positions_define_entity_scope(self, clf: RuleClassifier):
        result = clf(_inst("Skin: No [ENT] rashes [/ENT] or lesions noted.", entity="rashes"))
        assert result["labels"] == [0, 1, 0, 0]
        assert not result["abstained"]
        assert any(rule_id.startswith("negation-pre-") for rule_id in result["rule_ids"])

    def test_no_evidence_of(self, clf: RuleClassifier):
        result = clf(_inst("No evidence of diabetes on labs."))
        assert result["labels"] == [0, 1, 0, 0]

    def test_negative_for(self, clf: RuleClassifier):
        result = clf(_inst("Screening negative for diabetes."))
        assert result["labels"] == [0, 1, 0, 0]

    def test_post_entity_negation(self, clf: RuleClassifier):
        result = clf(_inst("diabetes was ruled out."))
        assert result["labels"] == [0, 1, 0, 0]

    def test_without(self, clf: RuleClassifier):
        result = clf(_inst("Without diabetes or hypertension."))
        assert result["labels"] == [0, 1, 0, 0]

    def test_absent(self, clf: RuleClassifier):
        result = clf(_inst("diabetes absent on exam."))
        assert result["labels"] == [0, 1, 0, 0]


# ── Implicit negation (entity-internal) ─────────────────────────────────────


class TestImplicitNegation:
    def test_normal_in_entity(self, clf: RuleClassifier):
        result = clf(_inst("Patient has normal gait.", entity="normal gait"))
        assert result["labels"] == [0, 1, 0, 0]

    def test_full_term_in_entity(self, clf: RuleClassifier):
        result = clf(_inst("Infant is full-term.", entity="full-term"))
        assert result["labels"] == [0, 1, 0, 0]


# ── Family / associated_with_someone_else ───────────────────────────────────


class TestFamily:
    def test_family_history(self, clf: RuleClassifier):
        result = clf(_inst("Family history of diabetes."))
        assert result["labels"] == [0, 0, 1, 0]
        assert not result["abstained"]

    def test_mother(self, clf: RuleClassifier):
        result = clf(_inst("Mother has diabetes."))
        assert result["labels"] == [0, 0, 1, 0]

    def test_paternal(self, clf: RuleClassifier):
        result = clf(_inst("Paternal grandfather had diabetes."))
        assert result["labels"] == [0, 0, 1, 0]

    def test_fhx(self, clf: RuleClassifier):
        result = clf(_inst("FHx: diabetes, hypertension."))
        assert result["labels"] == [0, 0, 1, 0]


# ── Other person ────────────────────────────────────────────────────────────


class TestOtherPerson:
    def test_donor(self, clf: RuleClassifier):
        result = clf(_inst("Donor had diabetes."))
        assert result["labels"] == [0, 0, 1, 0]

    def test_recipient(self, clf: RuleClassifier):
        result = clf(_inst("Recipient with diabetes."))
        assert result["labels"] == [0, 0, 1, 0]


# ── Hypothetical / other_non_patient ────────────────────────────────────────


class TestHypothetical:
    def test_if(self, clf: RuleClassifier):
        result = clf(_inst("If diabetes develops, start metformin."))
        assert result["labels"] == [0, 0, 0, 1]
        assert not result["abstained"]

    def test_possible(self, clf: RuleClassifier):
        result = clf(_inst("Possible diabetes based on A1c."))
        assert result["labels"] == [0, 0, 0, 1]

    def test_risk_of(self, clf: RuleClassifier):
        result = clf(_inst("Risk of diabetes given BMI."))
        assert result["labels"] == [0, 0, 0, 1]

    def test_screening_for(self, clf: RuleClassifier):
        result = clf(_inst("Screening for diabetes."))
        assert result["labels"] == [0, 0, 0, 1]

    def test_post_hypothetical(self, clf: RuleClassifier):
        result = clf(_inst("diabetes is possible."))
        assert result["labels"] == [0, 0, 0, 1]


# ── Abstention (llm_review) ─────────────────────────────────────────────────


class TestAbstain:
    def test_can_triggers_abstain(self, clf: RuleClassifier):
        result = clf(_inst("Diabetes can cause neuropathy."))
        assert result["abstained"]
        assert result["labels"] == [0, 0, 0, 0]

    def test_cannot_does_not_abstain(self, clf: RuleClassifier):
        # "cannot" is a negation trigger, not llm_review
        result = clf(_inst("Cannot rule out diabetes."))
        assert not result["abstained"]


# ── Output shape ─────────────────────────────────────────────────────────────


class TestOutputShape:
    def test_labels_length(self, clf: RuleClassifier):
        result = clf(_inst("Patient has diabetes."))
        assert len(result["labels"]) == NUM_LABELS
        assert len(result["probs"]) == NUM_LABELS

    def test_rule_ids_present(self, clf: RuleClassifier):
        result = clf(_inst("Patient denies diabetes."))
        assert len(result["rule_ids"]) > 0

    def test_confirmed_has_rule_ids_empty(self, clf: RuleClassifier):
        result = clf(_inst("Patient has diabetes."))
        assert result["rule_ids"] == []
