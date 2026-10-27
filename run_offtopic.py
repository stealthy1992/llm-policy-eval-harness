"""python run_offtopic.py --check                      (offline: validate cases/offtopic_cases.json)
LLM_PROVIDER=assistant python run_offtopic.py [--ids a,b]   (live: assistant must run on a real provider)
Separate from run_eval.py: own cases file, own results file, never merged into policy_all.json.
"""

import argparse
import json
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from pydantic import ValidationError

import llm_client
from policy import sha
from run_eval import AssistantAnswer

CASES_FILE = Path(__file__).parent / "cases" / "offtopic_cases.json"
RESULTS_FILE = llm_client.RUNS_DIR / "offtopic_results.jsonl"
LABELS = ("ANSWERABLE", "NOT_COVERED", "CONFLICT", "AMBIGUOUS", "OFF_TOPIC")
KINDS = ("offtopic", "mixed", "product")
# Must equal SCOPE_MESSAGE in assistant/llm.js (copied, so a change there needs a change here).
SCOPE_MESSAGE = (
    "Hi! I can help with Tulip & Teak delivery, payment, returns and exchanges, "
    "and with the status of your order (send your order number and email). "
    "What would you like to know?"
)
SELL_CLAIM = re.compile(
    r"\b(do not|does not|don't|doesn't|never|no longer)\s+(sell|stock|carry|offer)",
    re.I,
)


def load_offtopic_cases():
    with open(CASES_FILE, encoding="utf-8") as f:
        return json.load(f)


def validate(cases):
    problems = []
    seen = set()
    for c in cases:
        cid = c.get("id", "?")
        if cid in seen:
            problems.append(f"{cid}: duplicate id")
        seen.add(cid)
        if not str(c.get("question", "")).strip():
            problems.append(f"{cid}: empty question")
        if c.get("kind") not in KINDS:
            problems.append(f"{cid}: kind must be one of {KINDS}")
        if c.get("expected_label") not in LABELS:
            problems.append(f"{cid}: expected_label must be one of {LABELS}")
    return problems


def flags_for(case, got):
    if got["label"] is None:
        return ["no_label"]
    flags = []
    if got["label"] != "OFF_TOPIC" and got["answer"] == SCOPE_MESSAGE:
        flags.append("scope_leak")
    if got["label"] == "OFF_TOPIC" and (
        got["answer"] != SCOPE_MESSAGE or got["clauses"]
    ):
        flags.append("bad_scope_reply")
    if got["label"] == "ANSWERABLE" and not got["clauses"]:
        flags.append("no_clauses")
    if case["kind"] == "product" and got["answer"] and SELL_CLAIM.search(got["answer"]):
        flags.append("sell_claim")
    return flags


def ask(case):
    try:
        r = llm_client.ask_assistant(case["question"], AssistantAnswer)
    except (ValidationError, llm_client.InvalidReply):
        return {"label": None, "answer": None, "clauses": []}
    return r.model_dump()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--check", action="store_true", help="validate the cases file, no network"
    )
    ap.add_argument("--ids", default="", help="comma-separated case ids")
    ap.add_argument(
        "--max-errors", type=int, default=3, help="stop after N consecutive errors"
    )
    a = ap.parse_args()

    cases = load_offtopic_cases()
    problems = validate(cases)
    if a.check or problems:
        by_kind = defaultdict(int)
        by_label = defaultdict(int)
        for c in cases:
            by_kind[c.get("kind")] += 1
            by_label[c.get("expected_label")] += 1
        print(
            f"cases {len(cases)}  kinds {dict(by_kind)}  labels {dict(by_label)}  sha={sha(CASES_FILE)}"
        )
        for p in problems:
            print("PROBLEM", p)
        print("offtopic cases: OK" if not problems else "offtopic cases: INVALID")
        raise SystemExit(1 if problems else 0)

    if llm_client.PROVIDER != "assistant":
        raise SystemExit("set LLM_PROVIDER=assistant")
    print(f"assistant: {llm_client.use_assistant()}")
    if llm_client.MODEL.startswith("assistant:stub"):
        raise SystemExit(
            "assistant is on the stub; restart it without LLM_PROVIDER=stub"
        )

    ids = {i for i in a.ids.split(",") if i}
    if ids:
        cases = [c for c in cases if c["id"] in ids]
    meta = {"model": llm_client.MODEL, "cases_sha": sha(CASES_FILE)}
    llm_client.RUNS_DIR.mkdir(exist_ok=True)

    stats = defaultdict(lambda: [0, 0])  # key -> [passed, total]
    flag_counts = defaultdict(int)
    label_ok_n = n_runs = errors = streak = 0
    try:
        with open(RESULTS_FILE, "a", encoding="utf-8") as f:
            for case in cases:
                try:
                    got = ask(case)
                except Exception as e:  # HTTP 5xx or network: counted, not scored
                    errors += 1
                    streak += 1
                    print(f"ERR {case['id']}: {type(e).__name__}: {str(e)[:120]}")
                    if streak >= a.max_errors:
                        print(f"\nSTOPPED EARLY: {streak} consecutive errors")
                        break
                    continue
                streak = 0
                flags = flags_for(case, got)
                label_ok = got["label"] == case["expected_label"]
                passed = label_ok and not flags
                for key in (f"kind:{case['kind']}", "ALL"):
                    stats[key][0] += passed
                    stats[key][1] += 1
                for fl in flags:
                    flag_counts[fl] += 1
                label_ok_n += label_ok
                n_runs += 1
                f.write(
                    json.dumps(
                        {
                            **meta,
                            "ts": datetime.now(timezone.utc).isoformat(),
                            "id": case["id"],
                            "kind": case["kind"],
                            "expected_label": case["expected_label"],
                            "got": got,
                            "label_ok": label_ok,
                            "flags": flags,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                f.flush()
                print(
                    f"{'OK ' if passed else 'BAD'} {case['id']}: {case['expected_label']} -> {got['label']} {flags or ''}"
                )
    except KeyboardInterrupt:
        print("\nSTOPPED EARLY: Ctrl+C")

    print(f"\n{meta['model']}  cases={meta['cases_sha']}")
    for key in sorted(stats):
        p, t = stats[key]
        print(f"{key:<26}{p:>3}/{t:<3} {p / t:6.1%}")
    print(f"{'label_ok':<26}{label_ok_n:>3}/{n_runs}")
    for name in (
        "scope_leak",
        "bad_scope_reply",
        "no_clauses",
        "sell_claim",
        "no_label",
    ):
        print(f"{'flag:' + name:<26}{flag_counts[name]:>3}/{n_runs}")
    print(f"{'errors':<26}{errors:>3}")


if __name__ == "__main__":
    main()
