"""Cheap bulk silver labels with Claude Haiku 5.5 (development only), same prompt as scripts/label_llm.py.

  python scripts/label_haiku.py check N      # agreement with the Sonnet labels on N validation tickets
  python scripts/label_haiku.py label N      # label N more train tickets -> data/labels_haiku.jsonl
"""
import gzip
import json
import random
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import anthropic  # noqa: E402
from dotenv import load_dotenv  # noqa: E402

import label_llm as L  # noqa: E402

load_dotenv(ROOT / ".env")
MODEL = "claude-haiku-5-5"
OUT = ROOT / "data" / "labels_haiku.jsonl"
client = anthropic.Anthropic(max_retries=4)
cost = {"usd": 0.0}


def one(t):
    try:
        msg = client.messages.create(
            model=MODEL, max_tokens=200, thinking={"type": "disabled"},
            output_config={"effort": "low", "format": {"type": "json_schema", "schema": L.SCHEMA}},
            system=[{"type": "text", "text": L.SYSTEM, "cache_control": {"type": "ephemeral"}}],
            messages=[{"role": "user", "content": L.ticket_prompt(t)}])
        u = msg.usage
        cost["usd"] += ((u.input_tokens or 0) * 0.10 + (u.output_tokens or 0) * 0.50
                        + (u.cache_read_input_tokens or 0) * 0.01 + (u.cache_creation_input_tokens or 0) * 0.125) / 1e6
        return {"ticket_id": t["ticket_id"], **json.loads(next(b.text for b in msg.content if b.type == "text"))}
    except Exception as e:  # noqa: BLE001
        print("fail", type(e).__name__, flush=True)
        return None


def check(n):
    sonnet = {r["ticket_id"]: r for r in map(json.loads, open(ROOT / "data/labels_llm.jsonl", encoding="utf-8"))}
    valid = [json.loads(l) for l in open(ROOT / "data/handout/tickets_valid.jsonl", encoding="utf-8")]
    random.Random(3).shuffle(valid)
    sample = valid[:n]
    with ThreadPoolExecutor(16) as ex:
        res = [r for r in ex.map(one, sample) if r]
    agree = sum(sonnet[r["ticket_id"]]["category"] == r["category"] for r in res)
    pagree = sum(sonnet[r["ticket_id"]]["priority"] == r["priority"] for r in res)
    print(f"n={len(res)} category agreement {agree / len(res):.3f}, priority {pagree / len(res):.3f}, cost ${cost['usd']:.3f}")


def label(n):
    taken = {json.loads(l)["ticket_id"] for l in open(ROOT / "data/label_sample.jsonl", encoding="utf-8")}
    if OUT.exists():
        taken |= {json.loads(l)["ticket_id"] for l in open(OUT, encoding="utf-8")}
    train = [json.loads(l) for l in gzip.open(ROOT / "data/handout/tickets_train.jsonl.gz", "rt", encoding="utf-8")]
    pool = [t for t in train if t["ticket_id"] not in taken and t["messages"] and len(t["messages"][0]["text"]) >= 2]
    random.Random(7).shuffle(pool)
    done = 0
    with ThreadPoolExecutor(24) as ex, open(OUT, "a", encoding="utf-8") as f:
        for r in ex.map(one, pool[:n]):
            if r:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
                done += 1
                if done % 1000 == 0:
                    print("labelled", done, f"${cost['usd']:.3f}", flush=True)
    print("done", done, f"${cost['usd']:.3f}")


if __name__ == "__main__":
    {"check": check, "label": label}[sys.argv[1]](int(sys.argv[2]))
