"""python run_eval.py --check | [--cases PATH] [--split all] [--repeats 1] [--temperature 0.0]
Model comes from LLM_MODEL (same env var as the pytest suite)."""

import hashlib
import argparse
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from pydantic import ValidationError

import llm_client
from policy import (
    CASES_PATH,
    POLICY_PATH,
    PolicyAnswer,
    build_prompt,
    load_cases,
    policy_clauses,
    read_policy,
    sha,
    validate_cases,
)

ABSTAIN = ("NOT_COVERED", "AMBIGUOUS", "CONFLICT")


class AssistantAnswer(PolicyAnswer):
    """Assistant replies may carry OFF_TOPIC (prompt v2). Direct runs keep PolicyAnswer,
    so the schema sent to direct models (and their baselines) does not change."""

    label: Literal["ANSWERABLE", "NOT_COVERED", "CONFLICT", "AMBIGUOUS", "OFF_TOPIC"]


def run_case(policy, case, temperature):
    try:
        if llm_client.PROVIDER == "assistant":
            r = llm_client.ask_assistant(case["question"], AssistantAnswer)
        else:
            prompt = build_prompt(policy, case["question"])
            r = llm_client.ask_structured(prompt, PolicyAnswer, temperature=temperature)
    except (ValidationError, llm_client.InvalidReply):
        return {"label": None, "answer": None, "clauses": []}
    return r.model_dump()


class Stop(Exception):
    pass


def q_sha(case):
    return hashlib.sha256(case["question"].encode("utf-8")).hexdigest()[:10]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["dev", "test", "all"], default="all")
    ap.add_argument("--cases", default=str(CASES_PATH), help="cases JSON file")
    ap.add_argument(
        "--check", action="store_true", help="validate cases offline, no model call"
    )
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument(
        "--ids", default="", help="comma-separated case ids; overrides --split"
    )
    ap.add_argument(
        "--skip-done",
        action="store_true",
        help="model/temp/policy hash and question text",
    )
    ap.add_argument(
        "--max-errors", type=int, default=3, help="stop after N consecutive API errors"
    )
    a = ap.parse_args()
    cases_path = Path(a.cases)
    if a.check:
        policy, cases = read_policy(), load_cases(cases_path)
        problems = validate_cases(cases, policy)
        labels = defaultdict(int)
        for c in cases:
            labels[c.get("expected", {}).get("label")] += 1
        print(f"cases {len(cases)}  labels {dict(labels)}  sha={sha(cases_path)}")
        for p in problems:
            print("PROBLEM", p)
        print("policy cases: OK" if not problems else "policy cases: INVALID")
        raise SystemExit(1 if problems else 0)
    if llm_client.PROVIDER == "assistant":
        if a.temperature:
            raise SystemExit(
                "--temperature is set by the assistant server (LLM_TEMPERATURE); leave it at 0"
            )
        print(f"assistant: {llm_client.use_assistant()}")

    policy, cases = read_policy(), load_cases(cases_path)
    known = policy_clauses(policy)
    ids = {i for i in a.ids.split(",") if i}
    cases = [
        c
        for c in cases
        if (c["id"] in ids if ids else (a.split == "all" or c.get("split") == a.split))
    ]
    meta = {
        "model": llm_client.MODEL,
        "temperature": a.temperature,
        "split": "ids" if ids else a.split,
        "policy_sha": sha(POLICY_PATH),
        "cases_sha": sha(cases_path),
    }
    out = llm_client.RUNS_DIR / "policy_results.jsonl"
    llm_client.RUNS_DIR.mkdir(exist_ok=True)
    stats = defaultdict(lambda: [0, 0])  # key -> [correct, total]
    flags = defaultdict(int)
    n_runs = errors = skipped = streak = 0

    done = set()
    if a.skip_done and out.exists():
        want_meta = (meta["model"], a.temperature, meta["policy_sha"])
        with open(out, encoding="utf-8") as g:
            for line in g:
                r = json.loads(line)
                if (
                    r["model"],
                    r["temperature"],
                    r["policy_sha"],
                ) == want_meta and "q_sha" in r:
                    done.add((r["id"], r["rep"], r["q_sha"]))

    try:
        with open(out, "a", encoding="utf-8") as f:
            for case in cases:
                exp = case["expected"]
                for rep in range(a.repeats):
                    if (case["id"], rep, q_sha(case)) in done:
                        skipped += 1
                        continue
                    try:
                        got = run_case(policy, case, a.temperature)
                    except Exception as e:  # API/network error, not bad JSON
                        errors += 1
                        streak += 1
                        print(f"ERR {case['id']}: {type(e).__name__}: {str(e)[:120]}")
                        if streak >= a.max_errors:
                            raise Stop(f"{streak} consecutive errors")
                        continue
                    streak = 0
                    ok = got["label"] == exp["label"]
                    cited = set(got["clauses"])
                    want = set(exp["clauses"])
                    recall = (len(cited & want) / len(want)) if want else None
                    invalid_clause = bool(cited - known)
                    inconsistent = (
                        got["label"] in ("ANSWERABLE", "CONFLICT") and not cited
                    )
                    abstained = got["label"] in ABSTAIN
                    for key in (
                        f"category:{case['category']}",
                        f"label:{exp['label']}",
                        "ALL",
                    ):
                        stats[key][0] += ok
                        stats[key][1] += 1
                    akey = (
                        "abstain:not_needed"
                        if exp["label"] == "ANSWERABLE"
                        else "abstain:needed"
                    )
                    stats[akey][0] += abstained
                    stats[akey][1] += 1
                    flags["invalid_clause"] += invalid_clause
                    flags["inconsistent"] += inconsistent
                    flags["off_topic"] += got["label"] == "OFF_TOPIC"
                    flags["no_label"] += got["label"] is None
                    n_runs += 1
                    f.write(
                        json.dumps(
                            {
                                **meta,
                                "ts": datetime.now(timezone.utc).isoformat(),
                                "id": case["id"],
                                "q_sha": q_sha(case),
                                "rep": rep,
                                "category": case["category"],
                                "expected_label": exp["label"],
                                "got": got,
                                "label_ok": ok,
                                "clause_recall": recall,
                                "invalid_clause": invalid_clause,
                                "inconsistent": inconsistent,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    f.flush()
                    print(
                        f"{'OK ' if ok else 'BAD'} {case['id']}: {exp['label']} -> {got['label']}"
                    )
    except (Stop, KeyboardInterrupt) as e:
        print(f"\nSTOPPED EARLY: {e or 'Ctrl+C'}")

    print(
        f"\n{meta['model']}  split={meta['split']}  policy={meta['policy_sha']}  cases={meta['cases_sha']}"
    )
    for key in sorted(stats):
        c, t = stats[key]
        print(f"{key:<26}{c:>3}/{t:<3} {c / t:6.1%}")
    for name in ("invalid_clause", "inconsistent", "off_topic", "no_label"):
        print(f"{'flag:' + name:<26}{flags[name]:>3}/{n_runs}")
    print(f"{'errors':<26}{errors:>3}   skipped(done) {skipped}")


if __name__ == "__main__":
    main()
