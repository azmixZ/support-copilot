"""Evaluate on tickets_valid.jsonl and write reports/metrics.json + reports/examples.md.

  python scripts/evaluate.py            # full run (predict + metrics + LLM judge on 100 drafts)
  python scripts/evaluate.py --no-judge
"""
import json
import random
import re
import statistics
import subprocess
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")

from sklearn.metrics import accuracy_score, f1_score  # noqa: E402

from app.draft import has_promise, rule_mentions  # noqa: E402
from app.rules import RuleBase  # noqa: E402

VALID = ROOT / "data/handout/tickets_valid.jsonl"
PRED = ROOT / "data/valid_pred.jsonl"
REPORTS = ROOT / "reports"
JUDGE_MODEL = "claude-sonnet-5-5"

JUDGE_SYSTEM = """Ты старший администратор поддержки игры Grand Mobile. Тебе показывают обращение игрока, реальный ответ администратора и черновик ассистента.
Оцени черновик по шкале 1–5, как если бы решал, можно ли его отправить:
5 — можно отправить как есть; 4 — мелкая правка; 3 — по сути верно, но нужна заметная правка; 2 — в основном неверно; 1 — вредно (выдуманные правила, ложные обещания, не тот язык, грубость).
Черновик не видит скриншоты, поэтому он не обязан угадать итоговый вердикт; честное «решение примет администратор после проверки» не ошибка.
Ответь JSON: {"score": 1-5, "comment": "кратко"}"""


ADMIN_FIELDS = ("verdict", "violation", "punishment", "status")


def without_admin_answer(t: dict) -> dict:
    """The service is called before the admin answers, so the evaluation input has no admin reply or verdict."""
    t = {k: v for k, v in t.items() if k not in ADMIN_FIELDS}
    t["messages"] = [m for m in t.get("messages", []) if m.get("role") != "admin"]
    return t


def run_predict():
    blind = ROOT / "data/valid_input_no_admin.jsonl"
    with open(blind, "w", encoding="utf-8") as f:
        for line in open(VALID, encoding="utf-8"):
            f.write(json.dumps(without_admin_answer(json.loads(line)), ensure_ascii=False) + "\n")
    subprocess.run([sys.executable, str(ROOT / "predict.py"), f"IN={blind}", f"OUT={PRED}", "META=1"], check=True, cwd=ROOT)


def judge(pairs):
    import anthropic
    client = anthropic.Anthropic(max_retries=4)
    schema = {"type": "object", "properties": {"score": {"type": "integer", "enum": [1, 2, 3, 4, 5]}, "comment": {"type": "string"}},
              "required": ["score", "comment"], "additionalProperties": False}

    def one(p):
        t, r = p
        player = "\n".join(m["text"] for m in t["messages"] if m["role"] != "admin")
        real = next((m["text"] for m in t["messages"] if m["role"] == "admin"), "")
        msg = client.messages.create(
            model=JUDGE_MODEL, max_tokens=600, thinking={"type": "between_tools"},
            output_config={"effort": "low", "format": {"type": "json_schema", "schema": schema}},
            system=JUDGE_SYSTEM,
            messages=[{"role": "user", "content": f"Тип: {t['kind']}, вложений: {t.get('images_count') or 0}\nОбращение:\n{player[:1500]}\n\nРеальный ответ администратора:\n{real[:1200]}\n\nЧерновик ассистента (язык {r['draft_reply']['lang']}, цитаты {r['draft_reply']['cited']}):\n{r['draft_reply']['text']}"}])
        usage.append((msg.usage.input_tokens, msg.usage.output_tokens))
        return json.loads(next(b.text for b in msg.content if b.type == "text"))

    usage = []
    with ThreadPoolExecutor(12) as ex:
        res = list(ex.map(one, pairs))
    tin, tout = sum(u[0] for u in usage), sum(u[1] for u in usage)
    # Sonnet 5.5: $2 / $10 per million input / output tokens
    print(f"judge: {len(usage)} calls, {tin} in / {tout} out tokens, ~${tin * 2e-6 + tout * 10e-6:.3f}")
    return res


def main():
    REPORTS.mkdir(exist_ok=True)
    if "--reuse" not in sys.argv or not PRED.exists():
        run_predict()
    tickets = {t["ticket_id"]: t for t in map(json.loads, open(VALID, encoding="utf-8"))}
    preds = [json.loads(l) for l in open(PRED, encoding="utf-8")]
    rb = RuleBase(ROOT / "data/handout/rules.json")

    m = {"n": len(preds)}
    # --- classification vs silver labels (strong-model labels, see README) ---
    lab_path = ROOT / "data/labels_llm.jsonl"
    if lab_path.exists():
        labels = {r["ticket_id"]: r for r in map(json.loads, open(lab_path, encoding="utf-8"))}
        pairs = [(labels[p["ticket_id"]], p) for p in preds if p["ticket_id"] in labels]
        m["category_accuracy_vs_silver"] = round(accuracy_score([a["category"] for a, _ in pairs], [p["category"] for _, p in pairs]), 4)
        m["priority_macro_f1_vs_silver"] = round(f1_score([a["priority"] for a, _ in pairs], [p["priority"] for _, p in pairs], average="macro"), 4)
        m["n_silver"] = len(pairs)
        clf_rep = ROOT / "models/clf_valid_report.json"
        if clf_rep.exists():
            m["classifier_holdout"] = json.loads(clf_rep.read_text(encoding="utf-8"))

    # --- hallucination guard ---
    all_cites = [c for p in preds for c in p["draft_reply"]["cited"]]
    m["cited_total"] = len(all_cites)
    m["cited_nonexistent"] = sum(not rb.exists(c) for c in all_cites)
    bad_mentions = 0
    for p in preds:
        for x in rule_mentions(p["draft_reply"]["text"]):
            c = rb.canonical(x)
            if c is None or c not in p["draft_reply"]["cited"]:
                bad_mentions += 1
    m["rule_mentions_outside_cited"] = bad_mentions
    m["drafts_without_citations"] = sum(not p["draft_reply"]["cited"] for p in preds)
    # relevance proxy: when the admin named a rule, did we cite the same one?
    agree = tot = 0
    for p in preds:
        t = tickets[p["ticket_id"]]
        v = t.get("violation") or ""
        c = rb.canonical(v) if re.search(r"\d\.\d", v) else None
        if c:
            tot += 1
            agree += c in p["draft_reply"]["cited"]
    m["cited_matches_admin_rule"] = round(agree / tot, 4) if tot else None
    m["n_admin_rule"] = tot

    # --- policy ---
    m["needs_human_share"] = round(sum(p["needs_human"] for p in preds) / len(preds), 4)
    by_cat = Counter(p["category"] for p in preds)
    auto_by_cat = Counter(p["category"] for p in preds if not p["needs_human"])
    m["auto_share_by_category"] = {c: round(auto_by_cat[c] / n, 2) for c, n in by_cat.most_common()}
    m["category_distribution"] = dict(by_cat.most_common())
    m["promises_in_auto_drafts"] = sum(has_promise(p["draft_reply"]["text"]) and not p["needs_human"] for p in preds)
    always = {"report.account", "report.payment", "complaint.admin"}
    m["always_human_violations"] = sum(p["category"] in always and not p["needs_human"] for p in preds)
    m["spam_or_duplicate_share"] = round(sum(p["is_spam_or_duplicate"] for p in preds) / len(preds), 4)

    # --- language ---
    lang_ok = [p["draft_reply"]["lang"] == ({"und": None}.get(tickets[p["ticket_id"]]["lang"], tickets[p["ticket_id"]]["lang"]))
               for p in preds if tickets[p["ticket_id"]]["lang"] in ("ru", "kk", "uz", "en")]
    m["draft_lang_matches_player"] = round(sum(lang_ok) / len(lang_ok), 4)

    # --- cost / latency ---
    lat = sorted(p["_meta"]["latency_ms"] for p in preds)
    m["latency_ms_p50"] = round(statistics.median(lat))
    m["latency_ms_p95"] = round(lat[int(len(lat) * 0.95) - 1])
    m["cost_usd_per_ticket_avg"] = round(sum(p["_meta"]["cost_usd"] for p in preds) / len(preds), 6)
    m["cost_usd_per_ticket_max"] = round(max(p["_meta"]["cost_usd"] for p in preds), 6)
    m["llm_fallback_share"] = round(sum(not p["_meta"]["llm"] for p in preds) / len(preds), 4)

    # --- blind-ish judge on 100 drafts, stratified to include kk / uz ---
    rnd = random.Random(0)
    if "--no-judge" not in sys.argv:
        kkuz = [p for p in preds if tickets[p["ticket_id"]]["lang"] in ("kk", "uz")]
        rest = [p for p in preds if tickets[p["ticket_id"]]["lang"] not in ("kk", "uz")]
        sample = rnd.sample(kkuz, min(25, len(kkuz))) + rnd.sample(rest, 75)
        scores = judge([(tickets[p["ticket_id"]], p) for p in sample])
        for p, s in zip(sample, scores):
            p["_judge"] = s
        m["judge_model"] = JUDGE_MODEL
        m["judge_mean_score"] = round(statistics.mean(s["score"] for s in scores), 2)
        m["judge_score_distribution"] = dict(sorted(Counter(s["score"] for s in scores).items()))
        kz = [s["score"] for p, s in zip(sample, scores) if tickets[p["ticket_id"]]["lang"] in ("kk", "uz")]
        m["judge_mean_score_kk_uz"] = round(statistics.mean(kz), 2) if kz else None
        examples = sample[:20]
    else:
        examples = rnd.sample(preds, 20)

    (REPORTS / "metrics.json").write_text(json.dumps(m, ensure_ascii=False, indent=1), encoding="utf-8")
    lines = ["# Примеры черновиков на валидации\n"]
    for p in examples:
        t = tickets[p["ticket_id"]]
        player = " / ".join(x["text"] for x in t["messages"] if x["role"] != "admin")
        real = next((x["text"] for x in t["messages"] if x["role"] == "admin"), "")
        lines.append(f"### {t['kind']} · {t['lang']} · {p['category']} · {p['priority']} · needs_human={p['needs_human']}")
        lines.append(f"**Игрок:** {player[:400]}\n")
        lines.append(f"**Черновик ({p['draft_reply']['lang']}, цитаты {', '.join(p['draft_reply']['cited']) or '—'}):** {p['draft_reply']['text']}\n")
        lines.append(f"**Реальный ответ админа:** {real[:400]}\n")
        if "_judge" in p:
            lines.append(f"**Оценка судьи:** {p['_judge']['score']}/5 — {p['_judge']['comment']}\n")
    (REPORTS / "examples.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(m, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
