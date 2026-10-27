import json
import sys

from llm_client import RUNS_DIR
from policy import CASES_PATH

split = sys.argv[1] if len(sys.argv) > 1 else "all"
cases = json.loads(CASES_PATH.read_text(encoding="utf-8"))
n = len(cases) if split == "all" else sum(c["split"] == split for c in cases)
rows = [
    json.loads(line)
    for line in open(RUNS_DIR / "policy_results.jsonl", encoding="utf-8")
]
last = [r for r in rows if r["split"] == split][-n:]
bad = [r for r in last if not r["label_ok"]]
print(f"rows {len(last)}  bad {len(bad)}")
for r in bad:
    print(
        f"{r['id']:<42}{r['category']:<14}{r['expected_label']:<12}-> {r['got']['label']}"
    )
