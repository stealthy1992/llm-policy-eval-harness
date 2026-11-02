"""python find_leaks.py [--results runs/agent_results.jsonl] [--label NAME] [--model TEXT]
Offline, 0 requests. Lists stored final replies that look like a tool call written as text (P3c).
A reply is listed if, after trimming, it starts with { [ ``` or <|, or contains a known tool name in quotes.
Columns: label | model | rep | case | turn | shape | the reply (first 300 characters)."""

import argparse
import json
from pathlib import Path

TOOLS = ("get_order", "get_stock", "get_policy_clause", "request_refund")


def shape(reply):
    t = reply.strip()
    body = t
    if body.startswith("```"):
        body = body.strip("`").strip()
        if body.lower().startswith("json"):
            body = body[4:].strip()
    try:
        v = json.loads(body)
    except ValueError:
        return "text-with-braces" if ("{" in t and "}" in t) else "text"
    if isinstance(v, dict):
        keys = ",".join(sorted(v))[:60]
        return f"whole reply is a JSON object (keys: {keys})"
    return f"whole reply is JSON ({type(v).__name__})"


def looks_like_call(reply):
    t = reply.strip()
    return t.startswith(("{", "[", "```", "<|")) or any(f'"{n}"' in t for n in TOOLS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="runs/agent_results.jsonl")
    ap.add_argument("--label", default=None)
    ap.add_argument("--model", default="")
    a = ap.parse_args()
    lines = [json.loads(x) for x in Path(a.results).read_text(encoding="utf-8").splitlines() if x.strip()]
    scanned = found = 0
    for r in lines:
        if (a.label is not None and r["label"] != a.label) or a.model not in r["model"]:
            continue
        for i, reply in enumerate(r["replies"]):
            scanned += 1
            if reply and looks_like_call(reply):
                found += 1
                one = reply.replace("\n", "\\n")[:300]
                print(f"{r['label']} | {r['model']} | rep{r['rep']} | {r['id']} | turn {i + 1} | {shape(reply)}\n    {one}")
    print(f"\n{found} suspicious repl{'y' if found == 1 else 'ies'} of {scanned} stored repl{'y' if scanned == 1 else 'ies'} scanned")


if __name__ == "__main__":
    main()
