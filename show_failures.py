"""python show_failures.py [--label prompt-v0] [--rep 0] [--all] [--cases PATH] [--results runs/agent_results.jsonl]
Prints, for each failed case of one repeat: the user messages, the expected calls, the attempts the model made
(tool, args, loop status, short result) and the final reply. Offline: reads the results file only."""

import argparse
import json
import os
from pathlib import Path

DEFAULT_CASES = os.environ.get(
    "AGENT_CASES", r"D:\saleor-e2e-automation\agent\cases\dev.json"
)


def short(v, n=90):
    s = json.dumps(v, ensure_ascii=False)
    return s if len(s) <= n else s[:n] + "..."


def fmt_call(c):
    return f"{c['tool']}{short(c.get('args', {}), 120)}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="runs/agent_results.jsonl")
    ap.add_argument("--cases", default=DEFAULT_CASES)
    ap.add_argument("--label", default=None, help="only lines with this label (default: the label of the last line)")
    ap.add_argument("--model", default="", help="only lines whose model contains this text, e.g. llama3.1 or gemini")
    ap.add_argument("--rep", type=int, default=0)
    ap.add_argument("--all", action="store_true", help="also show passed cases")
    a = ap.parse_args()
    cases = {c["id"]: c for c in json.loads(Path(a.cases).read_text(encoding="utf-8"))}
    lines = [json.loads(x) for x in Path(a.results).read_text(encoding="utf-8").splitlines() if x.strip()]
    label = a.label if a.label is not None else lines[-1]["label"]
    # newest run of that label: the last N lines whose label matches and whose rep matches
    rows = [r for r in lines if r["label"] == label and r["rep"] == a.rep and a.model in r["model"]]
    latest = {}
    for r in rows:
        latest[r["id"]] = r  # a later line for the same case replaces an earlier one
    shown = 0
    for cid, r in latest.items():
        s = r["score"]
        if s["passed"] and not a.all:
            continue
        shown += 1
        print(f"\n== {cid} ({r['category']}) model={r['model']} rep={a.rep} traj={s['trajectory_ok']} safety={s['safety_ok']} state={s['state_ok']} actions={s['actions']}/{s['expected_actions']}")
        for i, turn in enumerate(cases[cid]["turns"]):
            print(f"  turn {i + 1} user: {turn['user']}")
            print(f"    expected: " + (", ".join(fmt_call(c) for c in turn["expected_calls"]) or "(none)")
                  + ("   [expect_no_calls]" if turn.get("expect_no_calls") else ""))
            if turn.get("forbidden_calls"):
                print("    forbidden: " + ", ".join(fmt_call(c) for c in turn["forbidden_calls"]))
            for t in r["attempts"][i]:
                print(f"    attempt:  {fmt_call(t)} -> {t['status']} {short(t['result'], 80)}")
            if not r["attempts"][i]:
                print("    attempt:  (no tool call)")
            print(f"    reply:    {r['replies'][i]}")
    print(f"\n{shown} case(s) shown (label={label!r}, model~{a.model!r}, rep={a.rep})")


if __name__ == "__main__":
    main()
