"""Dedup check on train: tickets are replayed in time order through the same DedupStore the service uses.

Admin verdict "duplicate" is the only label we have. It is incomplete (admins often answer an obvious repeat
as a normal complaint), so recall against it is meaningful and precision is a lower bound.
Writes reports/dedup.json.
"""
import gzip
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.pipeline import DedupStore  # noqa: E402
from app.text import normalize, player_text  # noqa: E402


def main():
    rows = [json.loads(l) for l in gzip.open(ROOT / "data/handout/tickets_train.jsonl.gz", "rt", encoding="utf-8")]
    rows.sort(key=lambda t: t["created_at"])
    store = DedupStore()
    tp = fp = fn = flagged = 0
    for t in rows:
        text = normalize(player_text(t["messages"]), t.get("region"))
        ts = datetime.fromisoformat(t["created_at"]).timestamp()
        dup = store.check(t.get("reporter_hash"), t.get("reported_hash"), t["kind"], text, ts)
        if t["kind"] != "complaint":  # organisers check against the admin verdict, which only complaints have
            continue
        gold = t.get("verdict") == "duplicate"
        flagged += dup
        tp += dup and gold
        fp += dup and not gold
        fn += gold and not dup
    n = sum(t["kind"] == "complaint" for t in rows)
    prec, rec = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    rep = {"complaints": n, "flagged_share": round(flagged / n, 4), "admin_duplicates": tp + fn,
           "recall_vs_admin_duplicates": round(rec, 4), "precision_vs_admin_duplicates": round(prec, 4),
           "f1_vs_admin_duplicates": round(2 * prec * rec / max(prec + rec, 1e-9), 4)}
    rep.update(valid_service_only())
    (ROOT / "reports").mkdir(exist_ok=True)
    (ROOT / "reports/dedup.json").write_text(json.dumps(rep, indent=1), encoding="utf-8")
    print(json.dumps(rep, indent=1))


def valid_service_only() -> dict:
    """Validation scored like the hidden test: inputs without the admin answer; the only history is what the
    service itself has seen during the run (train / valid must not be used as history, organisers)."""
    from app.pipeline import Triage
    tr = Triage()
    valid = sorted((json.loads(l) for l in open(ROOT / "data/handout/tickets_valid.jsonl", encoding="utf-8")),
                   key=lambda t: t["created_at"])
    tp = fp = fn = 0
    for t in valid:
        inp = {k: v for k, v in t.items() if k not in ("verdict", "violation", "punishment", "status")}
        inp["messages"] = [m for m in t["messages"] if m.get("role") != "admin"]
        dup = tr.prepare(inp)["is_dup"]
        if t["kind"] != "complaint":
            continue
        gold = t.get("verdict") == "duplicate"
        tp += dup and gold
        fp += dup and not gold
        fn += gold and not dup
    p, r = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    return {"valid_service_history_only": {"tp": tp, "fp": fp, "fn": fn, "precision": round(p, 4), "recall": round(r, 4),
                                           "f1": round(2 * p * r / max(p + r, 1e-9), 4)}}

if __name__ == "__main__":
    main()
