"""Three ways to measure category accuracy, side by side (writes reports/categories.json).

1. agreement with the category implied by the admin's own answer (appeal verdict -> complaint.admin,
   "оформлено неверно" -> complaint.invalid, named violation "2.1 ОП" / "DM" -> insult / dm ...), on validation
   complaints, with the admin answer removed from the input. No LLM involved; the organisers said the hidden labels
   take the admin answer into account.
2. silver labels on a held-out sample (from models/clf_valid_report.json, written by build_models.py);
3. silver labels the model was trained on (in-sample, an upper bound only — needs data/labels_llm*.jsonl).
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.pipeline import ADMIN_VERDICT_CATEGORY, Triage, category_of_cite, cite_of_violation  # noqa: E402

ADMIN = ("verdict", "violation", "punishment", "status")


def blind(t):
    inp = {k: v for k, v in t.items() if k not in ADMIN}
    inp["messages"] = [m for m in t.get("messages", []) if m.get("role") != "admin"]
    return inp


def main():
    tr = Triage()
    rep = {}
    ok = n = 0
    for line in open(ROOT / "data/handout/tickets_valid.jsonl", encoding="utf-8"):
        t = json.loads(line)
        if t["kind"] != "complaint":
            continue
        gold = ADMIN_VERDICT_CATEGORY.get(t.get("verdict"))
        if not gold and t.get("violation"):
            gold = category_of_cite(cite_of_violation(t["violation"], tr.rules), tr.rules)
        if not gold:
            continue
        n += 1
        ok += tr.prepare(blind(t))["category"] == gold
    rep["admin_answer_agreement"] = {"n": n, "accuracy": round(ok / max(n, 1), 4)}
    hold = ROOT / "models/clf_valid_report.json"
    if hold.exists():
        rep["silver_holdout"] = json.loads(hold.read_text(encoding="utf-8"))["category_accuracy"]
    labels, rows = {}, []
    for lf, sf in (("labels_llm.jsonl", "label_sample.jsonl"), ("labels_llm2.jsonl", "label_sample2.jsonl")):
        if (ROOT / "data" / lf).exists() and (ROOT / "data" / sf).exists():
            labels.update({r["ticket_id"]: r for r in map(json.loads, open(ROOT / "data" / lf, encoding="utf-8"))})
            rows += [json.loads(l) for l in open(ROOT / "data" / sf, encoding="utf-8")]
    if rows:
        rows = [r for r in rows if r["ticket_id"] in labels]
        ok = sum(tr.prepare(blind({k: v for k, v in r.items() if k != "split"}))["category"] == labels[r["ticket_id"]]["category"]
                 for r in rows)
        rep["silver_in_sample"] = {"n": len(rows), "accuracy": round(ok / len(rows), 4)}
    (ROOT / "reports/categories.json").write_text(json.dumps(rep, indent=1), encoding="utf-8")
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
