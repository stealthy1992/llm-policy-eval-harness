"""Load runs/agent_results.jsonl into Langfuse v4 (OTLP traces + /api/public/scores).
Dry run by default.
Usage: python load_langfuse.py | --send --limit 1 | --send | --send --scores-only --limit 1
Scores go in batches of 50 via /api/public/ingestion (score-create), with a one-by-one
fallback on /api/public/scores. --scores-only re-sends scores only (upsert, no duplicates).
Held-out rows: category-level aggregates only (no id, replies, attempts)."""

import argparse, collections, json, os, sys, time, uuid
from datetime import datetime, timezone
from pathlib import Path
import requests

ROOT = Path(__file__).parent
RUNS = ROOT / "runs" / "agent_results.jsonl"
NS = uuid.UUID("6f1c0f2e-5b1a-4c55-9a0e-3d8f2b7a1c01")


def load_env(path):
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"'))


def uid(*parts):
    return str(uuid.uuid5(NS, "|".join(str(p) for p in parts)))


def nanos(ts):
    return str(int(datetime.fromisoformat(ts).timestamp() * 1e9))


def sv(k, v):
    return {
        "key": k,
        "value": {
            "stringValue": v if isinstance(v, str) else json.dumps(v, default=str)
        },
    }


def av(k, items):
    return {
        "key": k,
        "value": {"arrayValue": {"values": [{"stringValue": str(i)} for i in items]}},
    }


def span_id(tid):
    return uid("span", tid).replace("-", "")[:16]


MS = (
    1_000_000  # the jsonl has no per-call timing: every step gets a synthetic 1 ms slot
)


def span(tid, name, start, end, attrs, sid=None, parent=None):
    d = {
        "traceId": tid.replace("-", ""),
        "spanId": sid or span_id(tid),
        "name": name,
        "kind": 1,
        "startTimeUnixNano": str(start),
        "endTimeUnixNano": str(end),
        "attributes": attrs,
        "status": {"code": 1},
    }
    if parent:
        d["parentSpanId"] = parent
    return d


def root_span(tid, name, ts, attrs, end=None):
    start = int(nanos(ts))
    return span(tid, name, start, end or start + MS, attrs)


def child_spans(tid, start, attempts, replies, turn_scores):
    """One span per turn, one tool span per call. Returns (spans, end_ns)."""
    root = span_id(tid)
    n_turns = max(len(attempts), len(replies), len(turn_scores))
    spans, cursor = [], start
    for i in range(n_turns):
        calls = (
            attempts[i] if i < len(attempts) and isinstance(attempts[i], list) else []
        )
        t_sid = uid("span", tid, "turn", i).replace("-", "")[:16]
        t_start = cursor
        for j, c in enumerate(calls):
            c = c if isinstance(c, dict) else {"tool": str(c)}
            status = c.get("status")
            a = [
                sv("langfuse.observation.type", "tool"),
                sv("langfuse.observation.input", c.get("args")),
                sv("langfuse.observation.output", c.get("result")),
                sv("langfuse.observation.metadata.status", status),
            ]
            if status != "ok":
                a += [
                    sv("langfuse.observation.level", "WARNING"),
                    sv("langfuse.observation.status_message", str(status)),
                ]
            c_sid = uid("span", tid, "turn", i, "call", j).replace("-", "")[:16]
            spans.append(
                span(
                    tid,
                    str(c.get("tool", "tool")),
                    cursor,
                    cursor + MS,
                    a,
                    c_sid,
                    t_sid,
                )
            )
            cursor += MS
        cursor += MS  # slot for the turn's final reply
        a = [sv("langfuse.observation.type", "span")]
        if i < len(replies):
            a.append(sv("langfuse.observation.output", replies[i]))
        ts_ = (
            turn_scores[i]
            if i < len(turn_scores) and isinstance(turn_scores[i], dict)
            else {}
        )
        a += [sv(f"langfuse.observation.metadata.{k}", v) for k, v in ts_.items()]
        spans.append(span(tid, f"turn {i + 1}", t_start, cursor, a, t_sid, root))
    return spans, cursor


def score(tid, name, value):
    if isinstance(value, bool):
        v, dt = (1 if value else 0), "BOOLEAN"
    elif isinstance(value, (int, float)):
        v, dt = float(value), "NUMERIC"
    else:
        return None
    return {
        "id": uid("score", "v2", tid, name),
        "traceId": tid.replace("-", ""),
        "observationId": span_id(tid),
        "name": name,
        "value": v,
        "dataType": dt,
    }


def dev_unit(r):
    tid = uid("trace", r["label"], r["id"], r["rep"], r["ts"])
    replies = r.get("replies") or [None]
    meta = {
        "rep": r["rep"],
        "model_calls": r["model_calls"],
        "error": r["error"],
        "eval_session_id": r["session_id"],
    }
    attrs = [
        sv("langfuse.trace.name", r["id"]),
        sv("langfuse.session.id", r["label"]),
        av("langfuse.trace.tags", [r["label"], r["category"], r["model"]]),
        sv("langfuse.observation.input", {"case": r["id"], "category": r["category"]}),
        sv("langfuse.observation.output", replies[-1]),
    ]
    attrs += [sv(f"langfuse.observation.metadata.{k}", v) for k, v in meta.items()]
    scores = [s for k, v in (r.get("score") or {}).items() if (s := score(tid, k, v))]
    start = int(nanos(r["ts"]))
    kids, end = child_spans(
        tid,
        start,
        r.get("attempts") or [],
        r.get("replies") or [],
        (r.get("score") or {}).get("turns") or [],
    )
    root = root_span(tid, r["id"], r["ts"], attrs, max(end, start + MS))
    return [root] + kids, scores


def heldout_units(rows):
    groups = collections.defaultdict(list)
    for r in rows:
        groups[(r["label"], r["category"])].append(r)
    units = []
    for (label, cat), rs in groups.items():
        tid = uid("trace-ho", label, cat)
        keys = sorted(
            {
                k
                for r in rs
                for k, v in (r.get("score") or {}).items()
                if isinstance(v, (bool, int, float))
            }
        )
        rates = {
            k: sum(float(r["score"][k]) for r in rs if k in r["score"])
            / max(1, sum(1 for r in rs if k in r["score"]))
            for k in keys
        }
        out = {
            "rows": len(rs),
            "pass_rate": rates,
            "errors": sum(1 for r in rs if r["error"]),
            "model_calls": sum(r["model_calls"] for r in rs),
        }
        attrs = [
            sv("langfuse.trace.name", f"{label}/{cat}"),
            sv("langfuse.session.id", label),
            av("langfuse.trace.tags", [label, cat, rs[0]["model"], "heldout"]),
            sv("langfuse.observation.input", {"category": cat}),
            sv("langfuse.observation.output", out),
        ]
        units.append(
            (
                [root_span(tid, f"{label}/{cat}", max(r["ts"] for r in rs), attrs)],
                [score(tid, k, v) for k, v in rates.items()],
            )
        )
    return units


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--send", action="store_true")
    ap.add_argument(
        "--scores-only",
        action="store_true",
        help="send scores only, no traces (scores upsert by id; spans do not)",
    )
    ap.add_argument(
        "--limit", type=int, default=0, help="max dev traces (0 = all dev + held-out)"
    )
    a = ap.parse_args()
    load_env(ROOT / ".env")
    rows = [
        json.loads(l)
        for l in RUNS.read_text(encoding="utf-8").splitlines()
        if l.strip()
    ]
    dev = [r for r in rows if "heldout" not in r["label"]]
    ho = [r for r in rows if "heldout" in r["label"]]
    units = [dev_unit(r) for r in dev]
    units = units[: a.limit] if a.limit else units + heldout_units(ho)
    n_scores = sum(len(u[1]) for u in units)
    n_spans = sum(len(u[0]) for u in units)
    print(
        f"rows {len(rows)} (dev {len(dev)}, heldout {len(ho)}); traces {len(units)}; spans {n_spans}; scores {n_scores}"
    )
    if not a.send:
        print("dry run: nothing sent")
        return
    host = os.environ["LANGFUSE_HOST"].rstrip("/")
    auth = (os.environ["LANGFUSE_PUBLIC_KEY"], os.environ["LANGFUSE_SECRET_KEY"])
    if not a.scores_only:
        spans = [s for u in units for s in u[0]]
        for i in range(0, len(spans), 50):
            body = {
                "resourceSpans": [
                    {
                        "resource": {
                            "attributes": [sv("service.name", "agent-gate-loader")]
                        },
                        "scopeSpans": [
                            {
                                "scope": {"name": "load_langfuse"},
                                "spans": spans[i : i + 50],
                            }
                        ],
                    }
                ]
            }
            r = requests.post(
                f"{host}/api/public/otel/v1/traces",
                json=body,
                auth=auth,
                headers={"x-langfuse-ingestion-version": "4"},
                timeout=60,
            )
            if r.status_code != 200:
                print("traces HTTP", r.status_code, r.text[:200])
                sys.exit(1)
        print(f"traces sent: {len(units)} ({len(spans)} spans)")
    t0 = time.time()
    ok, bad, fallback = send_scores(host, auth, [s for u in units for s in u[1]])
    print(
        f"scores sent: {ok} ok, {bad} errors, {fallback} via fallback, {time.time() - t0:.1f} s"
    )


def send_one(host, auth, s):
    r = requests.post(f"{host}/api/public/scores", json=s, auth=auth, timeout=30)
    if r.status_code in (200, 201):
        return True
    print("score HTTP", r.status_code, r.text[:150])
    return False


def send_scores(host, auth, scores, size=50):
    """Batches of score-create events; anything the batch rejects is sent one by one."""
    ok = bad = fallback = 0
    for i in range(0, len(scores), size):
        chunk = scores[i : i + size]
        now = datetime.now(timezone.utc).isoformat()
        events = {
            str(uuid.uuid4()): s for s in chunk
        }  # event id -> score (fresh event ids, score ids stay deterministic)
        batch = [
            {"id": eid, "type": "score-create", "timestamp": now, "body": s}
            for eid, s in events.items()
        ]
        retry = list(chunk)
        try:
            r = requests.post(
                f"{host}/api/public/ingestion",
                json={"batch": batch},
                auth=auth,
                timeout=120,
            )
            if r.status_code in (200, 207):
                j = r.json()
                failed = {e.get("id") for e in j.get("errors", [])}
                retry = [s for eid, s in events.items() if eid in failed]
                ok += len(chunk) - len(retry)
            else:
                print("batch HTTP", r.status_code, r.text[:150], "-> fallback")
        except (requests.RequestException, ValueError) as e:
            print("batch error", type(e).__name__, "-> fallback")
        for s in retry:
            fallback += 1
            if send_one(host, auth, s):
                ok += 1
            else:
                bad += 1
    return ok, bad, fallback


if __name__ == "__main__":
    main()
