"""
Regex patterns for phenotype context classification.

Ported from rule_based_binary_phenocontext and adapted for multi-label output.
Each pattern targets text relative to the phenotype mention span.

Categories and their label mappings:
  - negation          → negated
  - implicit_negation → negated
  - family            → associated_with_someone_else
  - other_person      → associated_with_someone_else
  - hypothetical      → other_non_patient
  - llm_review        → abstain (route to BioBERT)
"""

import re

_FLAGS = re.IGNORECASE

# ── Negation (pre-entity) ───────────────────────────────────────────────────
NEGATION_PRE = [
    r"\bno\b",
    r"\bnot\b",
    r"\bno\s+evidence\s+of\b",
    r"\bwithout\b",
    r"\bdenies\b",
    r"\bdenied\b",
    r"\bdeny\b",
    r"\bnegative\s+for\b",
    r"\bruled\s+out\b",
    r"\brule\s+out\b",
    r"\brules\s+out\b",
    r"\bfree\s+of\b",
    r"\babsence\s+of\b",
    r"\babsent\b",
    r"\bno\s+signs?\s+of\b",
    r"\bno\s+symptoms?\s+of\b",
    r"\bnever\s+had\b",
    r"\bnever\b",
    r"\bno\s+history\s+of\b",
    r"\bno\s+known\b",
    r"\bfailed\s+to\s+reveal\b",
    r"\bnot\s+demonstrate\b",
    r"\bnot\s+exhibit\b",
    r"\bnot\s+associated\s+with\b",
    r"\bwithout\s+evidence\s+of\b",
    r"\bno\s+definite\b",
    r"\bunlikely\b",
    r"\bnot\s+consistent\s+with\b",
    r"\bcannot\b",
    r"\bnormal\b",
    r"\bappropriate\b",
    r"\bnon(?=[\w\s-])",
    r"\bfull[\s-]?term\b",
    r"\bgestation(?:al)?\b",
]

# ── Negation (post-entity) ──────────────────────────────────────────────────
NEGATION_POST = [
    r"\bis\s+(?:not|unlikely)\b",
    r"\bwas\s+(?:not|unlikely|ruled\s+out|negative)\b",
    r"\bwere\s+(?:not|negative|absent|ruled\s+out)\b",
    r"\bnot\s+(?:found|seen|noted|observed|identified|present|detected)\b",
    r"\bruled\s+out\b",
    r"\bwas\s+negative\b",
    r"\bnegative\b",
    r"\babsent\b",
    r"\bnot\s+appreciated\b",
    r"\bcannot\b",
    r"\bnormal\b",
    r"\bappropriate\b",
    r"\bnon(?=[\w\s-])",
    r"\bfull[\s-]?term\b",
    r"\bgestation(?:al)?\b",
]

# ── Implicit negation (inside entity text) ──────────────────────────────────
IMPLICIT_NEGATION = [
    r"\bnormal\b",
    r"\bappropriate\b",
    r"\bnon(?=[\w\s-])",
    r"\bfull[\s-]?term\b",
    r"\bgestation(?:al)?\b",
]

# ── Family / experiencer attribution ────────────────────────────────────────
FAMILY = [
    r"\bfamily\s+history\b",
    r"\bfamilial\b",
    r"\bmother\b",
    r"\bfather\b",
    r"\bbrother\b",
    r"\bsister\b",
    r"\bsiblings?\b",
    r"\bdaughter\b",
    r"\bson\b",
    r"\bgrandmother\b",
    r"\bgrandfather\b",
    r"\bgrandparent\b",
    r"\baunt\b",
    r"\buncle\b",
    r"\bcousins?\b",
    r"\bparents?\b",
    r"\bfamily\s+members?\b",
    r"\bpregnancy\b",
    r"\bchild(?:ren)?\b",
    r"\bmaternal\b",
    r"\bpaternal\b",
    r"\bfh\s*:",
    r"\bfamily\s+hx\b",
    r"\bfhx\b",
]

# ── Hypothetical / uncertain / possible ─────────────────────────────────────
HYPOTHETICAL_PRE = [
    r"\bif\b",
    r"\bshould\b",
    r"\bcould\b",
    r"\bmay\b",
    r"\bmight\b",
    r"\bpossible\b",
    r"\bpossibly\b",
    r"\bprobable\b",
    r"\bprobably\b",
    r"\bperhaps\b",
    r"\bsuspect(?:ed|s)?\b",
    r"\bquestionable\b",
    r"\buncertain\b",
    r"\bequivocal\b",
    r"\bcannot\s+(?:be\s+)?(?:excluded|ruled\s+out)\b",
    r"\bconcern\s+for\b",
    r"\bworrisome\s+for\b",
    r"\bsuggestive\s+of\b",
    r"\bconsider\b",
    r"\bevaluate\s+for\b",
    r"\bwork\s*-?\s*up\s+for\b",
    r"\bto\s+rule\s+out\b",
    r"\bdifferential\b",
    r"\bpreventive\b",
    r"\bprophyla(?:ctic|xis)\b",
    r"\brisk\s+(?:of|for)\b",
    r"\bscreen(?:ing)?\s+for\b",
    r"\bborderline\b",
]

HYPOTHETICAL_POST = [
    r"\bis\s+(?:possible|probable|suspected|questionable|uncertain|unlikely)\b",
    r"\b(?:should|could|may|might)\s+be\b",
    r"\bcannot\s+be\s+(?:excluded|ruled\s+out)\b",
    r"\bhas\s+not\s+been\s+(?:confirmed|established)\b",
]

# ── Other-person attribution (non-family) ──────────────────────────────────
OTHER_PERSON = [
    r"\bdonor\b",
    r"\brecipient\b",
    r"\bunrelated\s+individual\b",
]

# ── Ambiguous — triggers abstention (route to BioBERT) ──────────────────────
LLM_REVIEW = [
    r"\bcan\b(?!not)",
]

# ── Compiled rule table ─────────────────────────────────────────────────────
# Each entry: scope (where to look), category (semantic label), compiled patterns.


def _compile(patterns: list[str]) -> list[re.Pattern[str]]:
    return [re.compile(p, _FLAGS) for p in patterns]


CONTEXT_RULES: list[dict] = [
    {"scope": "pre", "category": "negation", "patterns": _compile(NEGATION_PRE)},
    {"scope": "post", "category": "negation", "patterns": _compile(NEGATION_POST)},
    {"scope": "entity", "category": "implicit_negation", "patterns": _compile(IMPLICIT_NEGATION)},
    {"scope": "pre", "category": "hypothetical", "patterns": _compile(HYPOTHETICAL_PRE)},
    {"scope": "post", "category": "hypothetical", "patterns": _compile(HYPOTHETICAL_POST)},
    {"scope": "any", "category": "family", "patterns": _compile(FAMILY)},
    {"scope": "any", "category": "other_person", "patterns": _compile(OTHER_PERSON)},
    {"scope": "pre", "category": "llm_review", "patterns": _compile(LLM_REVIEW)},
    {"scope": "post", "category": "llm_review", "patterns": _compile(LLM_REVIEW)},
    {"scope": "entity", "category": "llm_review", "patterns": _compile(LLM_REVIEW)},
]
