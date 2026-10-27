import hashlib
import json
import re
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

CASES_DIR = Path(__file__).parent / "cases"
POLICY_PATH = CASES_DIR / "policy_v1_1.md"
CASES_PATH = CASES_DIR / "policy_sample.json"

LABELS = ("ANSWERABLE", "NOT_COVERED", "CONFLICT", "AMBIGUOUS")
CATEGORIES = (
    "direct",
    "boundary",
    "exception",
    "contradiction",
    "not_covered",
    "ambiguous",
    "multi_rule",
    "adversarial",
)
REQUIRED = {"id", "question", "category", "expected", "notes"}
REQUIRED_EXPECTED = {"label", "answer_facts", "clauses"}


class PolicyAnswer(BaseModel):
    # Field order = generation order: the model writes its answer before it commits to a label.
    answer: str
    clauses: list[str]
    label: Literal["ANSWERABLE", "NOT_COVERED", "CONFLICT", "AMBIGUOUS"]


def read_policy() -> str:
    return POLICY_PATH.read_text(encoding="utf-8")


def policy_clauses(policy: str) -> set[str]:
    return set(re.findall(r"^(\d+\.\d+)\s", policy, flags=re.M))


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:10]


def build_prompt(policy: str, question: str) -> str:
    return f"""You are a customer-support assistant. Answer ONLY from the policy below. Never use outside knowledge or invent rules.

Choose exactly one label:
- ANSWERABLE: the policy answers the question. Apply it literally, including thresholds, exceptions and arithmetic.
- NOT_COVERED: the policy says nothing about this. Say so and refer the customer to support.
- CONFLICT: two clauses give contradictory answers and the policy has no rule that settles it. Name both clauses; do not pick one.
- AMBIGUOUS: the wording supports more than one reading. State the readings or ask for clarification.

Reply with JSON: "answer" (short, customer-facing), "clauses" (clause numbers you relied on, e.g. ["3.2"]), "label".

POLICY:
{policy}

CUSTOMER QUESTION:
{question}"""


def load_cases(path: Path = CASES_PATH) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_cases(cases: list[dict], policy: str) -> list[str]:
    errors, seen, known = [], set(), policy_clauses(policy)
    for c in cases:
        cid = c.get("id", "<missing id>")
        if cid in seen:
            errors.append(f"{cid}: duplicate id")
        seen.add(cid)
        if " " in cid:
            errors.append(f"{cid}: id contains spaces")
        if not REQUIRED <= set(c):
            errors.append(f"{cid}: missing keys {REQUIRED - set(c)}")
            continue
        if c["category"] not in CATEGORIES:
            errors.append(f"{cid}: bad category {c['category']!r}")
        e = c["expected"]
        if not REQUIRED_EXPECTED <= set(e):
            errors.append(f"{cid}: expected missing {REQUIRED_EXPECTED - set(e)}")
            continue
        if e["label"] not in LABELS:
            errors.append(f"{cid}: bad label {e['label']!r}")
        if not e["answer_facts"]:
            errors.append(f"{cid}: empty answer_facts")
        for cl in e["clauses"]:
            if cl not in known:
                errors.append(f"{cid}: clause {cl} not in policy")
        if e["label"] == "NOT_COVERED" and e["clauses"]:
            errors.append(f"{cid}: NOT_COVERED should have clauses []")
        if e["label"] in ("ANSWERABLE", "CONFLICT") and not e["clauses"]:
            errors.append(f"{cid}: {e['label']} needs clauses")
        if "split" in c and c["split"] not in ("dev", "test"):
            errors.append(f"{cid}: bad split {c['split']!r}")
    return errors
