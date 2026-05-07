# Rule Manifest

Documents every rule in `src/ods_phenocontext/rules/patterns.py` and
`src/ods_phenocontext/rules/engine.py`.  Update this file whenever
patterns are added, removed, or recalibrated.

Rules are implemented as compiled regex patterns in Python rather than a
YAML file — see `docs/decision_log.md` for rationale.

---

## Rule families

| Family | Target label | Label index | Confidence | Scope | Derivation |
|---|---|---|---|---|---|
| `negation` | `negated` | 1 | 0.95 | pre-entity | Heuristic v1.0 |
| `negation` (post) | `negated` | 1 | 0.95 | post-entity | Heuristic v1.0 |
| `implicit_negation` | `negated` | 1 | 0.90 | entity text | Heuristic v1.0 |
| `hypothetical` | `other_non_patient` | 3 | 0.85 | pre-entity | Heuristic v1.0 |
| `hypothetical` (post) | `other_non_patient` | 3 | 0.85 | post-entity | Heuristic v1.0 |
| `family` | `associated_with_someone_else` | 2 | 0.95 | any | Heuristic v1.0 |
| `other_person` | `associated_with_someone_else` | 2 | 0.90 | any | Heuristic v1.0 |
| `llm_review` | abstain | — | — | pre / post / entity | Heuristic v1.0 |

Confidence scores are **heuristic** defaults for iteration 0.  They will be
replaced with empirical PPV estimates once gold labels are available.

When no rule fires, the engine assigns `confirmed` (index 0) with confidence
0.95.  Multiple families can fire on the same instance (multi-hot output).

---

## Negation — pre-entity (`negation`, scope: `pre`)

Confidence: **0.95**

Patterns match text appearing before the phenotype mention.

| Pattern | Example trigger |
|---|---|
| `\bno\b` | "no asthma" |
| `\bnot\b` | "not diabetic" |
| `\bno\s+evidence\s+of\b` | "no evidence of seizure" |
| `\bwithout\b` | "without hypertension" |
| `\bdenies\b` | "denies fever" |
| `\bdenied\b` | "denied pain" |
| `\bdeny\b` | "deny reflux" |
| `\bnegative\s+for\b` | "negative for infection" |
| `\bruled\s+out\b` | "ruled out fracture" |
| `\brule\s+out\b` | "rule out malignancy" |
| `\brules\s+out\b` | "rules out anemia" |
| `\bfree\s+of\b` | "free of edema" |
| `\babsence\s+of\b` | "absence of tremor" |
| `\babsent\b` | "absent on exam" |
| `\bno\s+signs?\s+of\b` | "no signs of rash" |
| `\bno\s+symptoms?\s+of\b` | "no symptoms of cough" |
| `\bnever\s+had\b` | "never had seizures" |
| `\bnever\b` | "never smoked" |
| `\bno\s+history\s+of\b` | "no history of asthma" |
| `\bno\s+known\b` | "no known allergies" |
| `\bfailed\s+to\s+reveal\b` | "failed to reveal pathology" |
| `\bnot\s+demonstrate\b` | "did not demonstrate weakness" |
| `\bnot\s+exhibit\b` | "did not exhibit pain" |
| `\bnot\s+associated\s+with\b` | "not associated with fever" |
| `\bwithout\s+evidence\s+of\b` | "without evidence of bleeding" |
| `\bno\s+definite\b` | "no definite fracture" |
| `\bunlikely\b` | "unlikely to represent" |
| `\bnot\s+consistent\s+with\b` | "not consistent with infection" |
| `\bcannot\b` | "cannot rule out" |
| `\bnormal\b` | "normal development" |
| `\bappropriate\b` | "appropriate for age" |
| `\bnon(?=[\w\s-])` | "non-verbal" |
| `\bfull[\s-]?term\b` | "full-term infant" |
| `\bgestation(?:al)?\b` | "gestational age" |

---

## Negation — post-entity (`negation`, scope: `post`)

Confidence: **0.95**

Patterns match text appearing after the phenotype mention.

| Pattern | Example trigger |
|---|---|
| `\bis\s+(?:not\|unlikely)\b` | "asthma is not present" |
| `\bwas\s+(?:not\|unlikely\|ruled\s+out\|negative)\b` | "fever was ruled out" |
| `\bwere\s+(?:not\|negative\|absent\|ruled\s+out)\b` | "seizures were absent" |
| `\bnot\s+(?:found\|seen\|noted\|observed\|identified\|present\|detected)\b` | "not found on imaging" |
| `\bruled\s+out\b` | "diabetes ruled out" |
| `\bwas\s+negative\b` | "culture was negative" |
| `\bnegative\b` | "negative" |
| `\babsent\b` | "absent" |
| `\bnot\s+appreciated\b` | "not appreciated on exam" |
| `\bcannot\b` | "cannot" |
| `\bnormal\b` | "normal" |
| `\bappropriate\b` | "appropriate" |
| `\bnon(?=[\w\s-])` | "non-contributory" |
| `\bfull[\s-]?term\b` | "full-term" |
| `\bgestation(?:al)?\b` | "gestational" |

---

## Implicit negation — entity text (`implicit_negation`, scope: `entity`)

Confidence: **0.90**

Patterns match within the entity span itself (e.g., NER returned a phrase
that inherently encodes negation).

| Pattern | Example entity |
|---|---|
| `\bnormal\b` | "normal gait" |
| `\bappropriate\b` | "appropriate development" |
| `\bnon(?=[\w\s-])` | "non-verbal behavior" |
| `\bfull[\s-]?term\b` | "full-term birth" |
| `\bgestation(?:al)?\b` | "gestational diabetes" |

---

## Family / experiencer attribution (`family`, scope: `any`)

Confidence: **0.95**

Patterns match anywhere in the context window.  Assigns
`associated_with_someone_else` (index 2).

| Pattern | Example trigger |
|---|---|
| `\bfamily\s+history\b` | "family history of asthma" |
| `\bfamilial\b` | "familial hypercholesterolemia" |
| `\bmother\b` | "mother has diabetes" |
| `\bfather\b` | "father with hypertension" |
| `\bbrother\b` | "brother diagnosed with seizures" |
| `\bsister\b` | "sister has anemia" |
| `\bsiblings?\b` | "sibling with same condition" |
| `\bdaughter\b` | "daughter affected" |
| `\bson\b` | "son with obesity" |
| `\bgrandmother\b` | "grandmother had stroke" |
| `\bgrandfather\b` | "grandfather with COPD" |
| `\bgrandparent\b` | "grandparent history" |
| `\baunt\b` | "aunt with cancer" |
| `\buncle\b` | "uncle affected" |
| `\bcousins?\b` | "cousin with same diagnosis" |
| `\bparents?\b` | "parents both affected" |
| `\bfamily\s+members?\b` | "family members with condition" |
| `\bpregnancy\b` | "pregnancy-related" |
| `\bchild(?:ren)?\b` | "children unaffected" |
| `\bmaternal\b` | "maternal history" |
| `\bpaternal\b` | "paternal grandfather" |
| `\bfh\s*:` | "FH: diabetes" |
| `\bfamily\s+hx\b` | "family hx of seizures" |
| `\bfhx\b` | "FHx positive" |

---

## Other-person attribution (`other_person`, scope: `any`)

Confidence: **0.90**

Non-family third-party experiencers.  Assigns `associated_with_someone_else`
(index 2).

| Pattern | Example trigger |
|---|---|
| `\bdonor\b` | "donor had hepatitis" |
| `\brecipient\b` | "recipient with prior infection" |
| `\bunrelated\s+individual\b` | "unrelated individual affected" |

---

## Hypothetical / uncertain / screening — pre-entity (`hypothetical`, scope: `pre`)

Confidence: **0.85**

| Pattern | Example trigger |
|---|---|
| `\bif\b` | "if diabetes develops" |
| `\bshould\b` | "should seizures occur" |
| `\bcould\b` | "could represent infection" |
| `\bmay\b` | "may have anemia" |
| `\bmight\b` | "might be related to" |
| `\bpossible\b` | "possible asthma" |
| `\bpossibly\b` | "possibly related" |
| `\bprobable\b` | "probable diagnosis" |
| `\bprobably\b` | "probably asthma" |
| `\bperhaps\b` | "perhaps consistent with" |
| `\bsuspect(?:ed\|s)?\b` | "suspected seizure" |
| `\bquestionable\b` | "questionable significance" |
| `\buncertain\b` | "uncertain etiology" |
| `\bequivocal\b` | "equivocal findings" |
| `\bcannot\s+(?:be\s+)?(?:excluded\|ruled\s+out)\b` | "cannot be excluded" |
| `\bconcern\s+for\b` | "concern for malignancy" |
| `\bworrisome\s+for\b` | "worrisome for infection" |
| `\bsuggestive\s+of\b` | "suggestive of asthma" |
| `\bconsider\b` | "consider diabetes" |
| `\bevaluate\s+for\b` | "evaluate for seizures" |
| `\bwork\s*-?\s*up\s+for\b` | "work-up for fever" |
| `\bto\s+rule\s+out\b` | "to rule out infection" |
| `\bdifferential\b` | "on differential" |
| `\bpreventive\b` | "preventive screening" |
| `\bprophyla(?:ctic\|xis)\b` | "prophylactic treatment" |
| `\brisk\s+(?:of\|for)\b` | "risk of diabetes" |
| `\bscreen(?:ing)?\s+for\b` | "screening for hypertension" |
| `\bborderline\b` | "borderline values" |

---

## Hypothetical / uncertain — post-entity (`hypothetical`, scope: `post`)

Confidence: **0.85**

| Pattern | Example trigger |
|---|---|
| `\bis\s+(?:possible\|probable\|suspected\|questionable\|uncertain\|unlikely)\b` | "asthma is possible" |
| `\b(?:should\|could\|may\|might)\s+be\b` | "diabetes may be present" |
| `\bcannot\s+be\s+(?:excluded\|ruled\s+out)\b` | "cannot be ruled out" |
| `\bhas\s+not\s+been\s+(?:confirmed\|established)\b` | "has not been confirmed" |

---

## LLM review / abstention (`llm_review`, scope: `pre`, `post`, `entity`)

No target label — causes the engine to **abstain** and route to BioBERT.

| Pattern | Reason |
|---|---|
| `\bcan\b(?!not)` | "can" without "not" is highly ambiguous (e.g., "can cause", "can develop") and not reliably classifiable by rules. |

---

## Default: confirmed (no rules fire)

When no pattern fires on any scope, the engine returns `confirmed` (index 0)
with confidence **0.95**.  This reflects the base rate expectation that an
unqualified mention in a clinical note refers to a finding present in the
patient.

---

## Versioning

| Version | Date | Notes |
|---|---|---|
| v1.0 | 2026-05-07 | Initial rule set. Confidence scores are heuristic; no empirical PPV computed yet. |
