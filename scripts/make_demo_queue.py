"""Triage a sample of validation tickets into data/demo_queue.jsonl for the admin UI."""
import json
import random
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
N = int(sys.argv[1]) if len(sys.argv) > 1 else 60
valid = [json.loads(l) for l in open(ROOT / "data/handout/tickets_valid.jsonl", encoding="utf-8")]
random.Random(1).shuffle(valid)
sample = valid[:N]
# Make sure the demo shows urgent tickets (account / payment) and a duplicate series, rare in a random sample.
urgent_words = ("взлом", "аккаунт", "оплат", "донат", "пропал")
extra = [t for t in valid[N:] if t["kind"] == "report" and any(w in " ".join(m["text"] for m in t["messages"]).lower() for w in urgent_words)][:4]
sample += extra
tmp_in, tmp_out = ROOT / "data/demo_in.jsonl", ROOT / "data/demo_pred.jsonl"
tmp_in.write_text("\n".join(json.dumps(t, ensure_ascii=False) for t in sample), encoding="utf-8")
subprocess.run([sys.executable, str(ROOT / "predict.py"), f"IN={tmp_in}", f"OUT={tmp_out}"], check=True, cwd=ROOT)
preds = {}
for l in open(tmp_out, encoding="utf-8"):
    r = json.loads(l)
    r.pop("_meta", None)
    preds[r.pop("ticket_id")] = r
with open(ROOT / "data/demo_queue.jsonl", "w", encoding="utf-8") as f:
    for t in sample:
        f.write(json.dumps({"ticket": t, "pred": preds[t["ticket_id"]]}, ensure_ascii=False) + "\n")
print("demo queue:", len(sample))
