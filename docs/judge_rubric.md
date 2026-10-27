# Judge prompt (Gemini) — answer_facts grading

Use only for cases where the label already matched (exact match is checked in code).
Send as the prompt; fill the three placeholders. Temperature 0. Never include the policy trap key.

---
You grade a customer-support answer against required facts. Do not use outside knowledge.

QUESTION: {question}
REQUIRED FACTS:
{answer_facts_numbered}
POLICY (use only to check invented_rule; do not require anything beyond the REQUIRED FACTS):
{policy}
ANSWER TO GRADE: {answer}

For each required fact, decide whether the answer states it or clearly implies it (paraphrase is fine; a wrong number is not).
Judge from the customer's point of view: a fact counts as stated if a customer reading the answer would learn it. Do not mark a fact false only because the answer omits a threshold, clause number or amount that is already given in the QUESTION, or because the fact merely restates another required fact. If the QUESTION asks for a specific number or outcome, the answer must give it.
A fact counts as stated if a customer reading the answer would learn it, even if the answer does not repeat thresholds, clause numbers or wording from the fact. Do not mark a fact false only because the answer omits a figure that is already in the question.
Also decide:
- contradicts: does the answer state anything that contradicts a required fact?
- invented_rule: does the answer assert a rule, number or deadline that is NOT in the POLICY, or that contradicts it? A true detail that appears in the POLICY is not invented.

Reply with JSON only:
{"facts": [true, false, ...], "contradicts": false, "invented_rule": false}

---
Pass = all facts true AND contradicts false. Record invented_rule separately (key metric for NOT_COVERED / CONFLICT / AMBIGUOUS).
Before trusting it: hand-grade ~20 answers and compute Cohen's kappa against the judge (Phase 2 deliverable).
