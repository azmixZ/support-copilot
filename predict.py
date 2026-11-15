"""Batch prediction: python predict.py IN=tickets.jsonl OUT=pred.jsonl [WORKERS=12] [META=1]

Each output line is {"ticket_id": ..., <POST /triage response>}.
META=1 also keeps per-ticket latency and cost in a "_meta" field (used by scripts/evaluate.py).
"""
import json
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent / ".env")

from app.bootstrap import ensure_models  # noqa: E402
from app.pipeline import Triage  # noqa: E402

# Returned when a ticket can't be processed at all, so one broken line never breaks the whole file.
SAFE_FALLBACK = {
    "lang": "und", "category": "report.offtopic", "priority": "p3", "is_spam_or_duplicate": False,
    "extracted": {"nickname": None, "rule_refs": [], "has_evidence": False, "server": None},
    "draft_reply": {"lang": "ru", "text": "", "cited": []},
    "needs_human": True, "confidence": 0.0, "reasons": ["не удалось разобрать обращение"],
}


def read_tickets(src: str) -> list[dict]:
    tickets = []
    for n, line in enumerate(open(src, encoding="utf-8"), 1):
        if not line.strip():
            continue
        try:
            t = json.loads(line)
        except json.JSONDecodeError:
            print(f"line {n}: invalid JSON, answered with fallback", file=sys.stderr)
            t = {"ticket_id": None, "_broken": True}
        tickets.append(t if isinstance(t, dict) else {"_broken": True})
    return tickets


def main():
    args = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
    pos = [a for a in sys.argv[1:] if "=" not in a]
    src = args.get("IN") or (pos[0] if pos else "tickets.jsonl")
    dst = args.get("OUT") or (pos[1] if len(pos) > 1 else "pred.jsonl")
    workers = int(args.get("WORKERS", 12))
    keep_meta = args.get("META") == "1"

    tickets = read_tickets(src)
    ensure_models()  # builds the models once if `make build` was skipped
    tr = Triage()
    # Dedup depends on order: run the fast stage chronologically first, then the LLM calls in parallel.
    order = sorted(range(len(tickets)), key=lambda i: str(tickets[i].get("created_at") or ""))
    ctxs, prep_ms = {}, {}
    for i in order:
        if tickets[i].get("_broken"):
            continue
        start = time.perf_counter()
        try:
            ctxs[i] = tr.prepare(tickets[i])
        except Exception as e:  # noqa: BLE001 — never lose the whole batch on one bad ticket
            print(f"ticket {tickets[i].get('ticket_id')}: {type(e).__name__}: {e}", file=sys.stderr)
        prep_ms[i] = (time.perf_counter() - start) * 1000

    def run(i):
        t = tickets[i]
        if i not in ctxs:
            return {**SAFE_FALLBACK, "_meta": {"latency_ms": 0.0, "cost_usd": 0.0, "llm": False}}
        ctx = ctxs[i]
        start = time.perf_counter()
        try:
            prompt = tr.drafter.build_prompt({**t, "kind": ctx["kind"], "messages": t.get("messages") or []},
                                             tr._prompt_ctx(ctx))
            llm = tr.drafter.draft(prompt, ctx["cites"])
            return tr.finish(t, ctx, llm, prep_ms[i] + (time.perf_counter() - start) * 1000)
        except Exception as e:  # noqa: BLE001
            print(f"ticket {t.get('ticket_id')}: {type(e).__name__}: {e}", file=sys.stderr)
            return {**SAFE_FALLBACK, "_meta": {"latency_ms": 0.0, "cost_usd": 0.0, "llm": False}}

    t0 = time.time()
    with ThreadPoolExecutor(workers) as ex:
        results = list(ex.map(run, range(len(tickets))))

    with open(dst, "w", encoding="utf-8") as f:
        for t, r in zip(tickets, results):
            out = {"ticket_id": t.get("ticket_id"), **r}
            if not keep_meta:
                out.pop("_meta", None)
            f.write(json.dumps(out, ensure_ascii=False) + "\n")

    print(f"done {len(results)} tickets in {time.time() - t0:.1f}s -> {dst}")
    if results:
        lat = sorted(r["_meta"]["latency_ms"] for r in results)
        cost = [r["_meta"]["cost_usd"] for r in results]
        print(f"latency p50={statistics.median(lat):.0f}ms p95={lat[max(int(len(lat) * 0.95) - 1, 0)]:.0f}ms; "
              f"cost avg=${sum(cost) / len(cost):.6f}/ticket; llm used {sum(r['_meta']['llm'] for r in results)}/{len(results)}; "
              f"needs_human {sum(r['needs_human'] for r in results) / len(results):.0%}")


if __name__ == "__main__":
    main()
