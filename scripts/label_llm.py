"""One-time silver labelling of category + priority with a strong model (Message Batches, 50% cheaper).

Usage:
  python scripts/label_llm.py submit   # builds sample, submits batch, saves batch id
  python scripts/label_llm.py collect  # waits for the batch and writes data/labels_llm.jsonl
"""
import gzip
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import anthropic  # noqa: E402
from dotenv import load_dotenv  # noqa: E402

from app.labels import CATEGORIES, complaint_category, report_category  # noqa: E402
from app.text import normalize, player_text  # noqa: E402

load_dotenv(ROOT / ".env")
DATA = ROOT / "data" / "handout"
STATE = ROOT / "data" / "label_batch.json"
OUT = ROOT / "data" / "labels_llm.jsonl"
MODEL = "claude-sonnet-5-5"
N_TRAIN = 2500

TAXONOMY = json.loads((DATA / "taxonomy.json").read_text(encoding="utf-8"))

SYSTEM = f"""Ты руководитель поддержки мобильной ролевой игры Grand Mobile. Размечаешь обращения игроков по таксономии.

Таксономия (JSON):
{json.dumps(TAXONOMY, ensure_ascii=False, indent=1)}

Как размечать:
- kind=complaint — жалоба из лаунчера, выбирай только complaint.*; kind=report — репорт из игры, выбирай только report.*.
- complaint.admin: жалоба на действия администратора, в том числе несогласие с выданным самому заявителю наказанием (апелляция).
- complaint.invalid: не жалоба (вопрос, болтовня), пустая жалоба вроде «жб», «вот он», только ссылка без описания, оскорбления в тексте самой жалобы.
- Если есть описание нарушения и оно подходит под категорию, выбирай категорию нарушения, даже если доказательства не видно.
- report.offtopic: пустые и бессодержательные «помогите», «ау», «хелп», «сос», приветствия, флуд. Если сказано, что именно случилось (застрял, перевернулась машина), это report.help.
- report.complaint: игрок сообщает о нарушении другого игрока через репорт (ДМ, оскорбления, читы).
- Приоритет ставь по правилам priorities. Наличие доказательств: images_count > 0 или ссылка <url> в тексте.
- Тексты короткие, с опечатками, на русском, казахском, узбекском и в транслите. Казахские буквы могут быть заменены: ќ=қ, ґ=ғ, ѕ=ң, є=ә, ј=ө, ў=ұ, ї=ү.
Ответь только JSON по схеме."""

SCHEMA = {
    "type": "object",
    "properties": {
        "category": {"type": "string", "enum": CATEGORIES},
        "priority": {"type": "string", "enum": ["p0", "p1", "p2", "p3"]},
    },
    "required": ["category", "priority"],
    "additionalProperties": False,
}


def ticket_prompt(t: dict) -> str:
    msgs = [m for m in t["messages"] if m["role"] != "admin"]
    lines = [f"kind: {t['kind']}"]
    if t["kind"] == "complaint":
        lines.append(f"rule_refs_raw: {t.get('rule_refs_raw')!r}")
        lines.append(f"images_count: {t.get('images_count') or 0}")
    for m in msgs:
        lines.append(f"{m['role']}: {normalize(m['text'], t.get('region'))[:1500]}")
    return "\n".join(lines)


def weak(t):
    txt = normalize(player_text(t["messages"]), t.get("region"))
    if t["kind"] == "complaint":
        return complaint_category(txt, t.get("rule_refs_raw") or "", "", None)
    return report_category(txt)


def build_sample():
    valid = [json.loads(l) for l in open(DATA / "tickets_valid.jsonl", encoding="utf-8")]
    train = [json.loads(l) for l in gzip.open(DATA / "tickets_train.jsonl.gz", "rt", encoding="utf-8")]
    rnd = random.Random(42)
    # Stratify by weak label so rare classes (payment, account, rmt) get enough examples.
    groups = defaultdict(list)
    for t in train:
        groups[weak(t)].append(t)
    per = N_TRAIN // len(groups)
    picked = []
    for g in groups.values():
        rnd.shuffle(g)
        picked += g[:per]
    rest = list(train)
    rnd.shuffle(rest)
    seen = {t["ticket_id"] for t in picked}
    for t in rest:
        if len(picked) >= N_TRAIN:
            break
        if t["ticket_id"] not in seen:
            picked.append(t)
            seen.add(t["ticket_id"])
    return [("valid", t) for t in valid] + [("train", t) for t in picked]


def submit():
    client = anthropic.Anthropic()
    sample = build_sample()
    (ROOT / "data" / "label_sample.jsonl").write_text(
        "\n".join(json.dumps({"split": s, **t}, ensure_ascii=False) for s, t in sample), encoding="utf-8")
    requests = []
    for _, t in sample:
        requests.append({
            "custom_id": t["ticket_id"][:64],
            "params": {
                "model": MODEL,
                "max_tokens": 1024,
                "thinking": {"type": "between_tools"},
                "output_config": {"effort": "low", "format": {"type": "json_schema", "schema": SCHEMA}},
                "system": [{"type": "text", "text": SYSTEM, "cache_control": {"type": "ephemeral"}}],
                "messages": [{"role": "user", "content": ticket_prompt(t)}],
            },
        })
    batch = client.messages.batches.create(requests=requests)
    STATE.write_text(json.dumps({"id": batch.id, "n": len(requests)}), encoding="utf-8")
    print("submitted", batch.id, len(requests))


def collect():
    client = anthropic.Anthropic()
    st = json.loads(STATE.read_text(encoding="utf-8"))
    while True:
        b = client.messages.batches.retrieve(st["id"])
        print(b.processing_status, b.request_counts, flush=True)
        if b.processing_status == "ended":
            break
        time.sleep(30)
    ok = bad = 0
    in_tok = out_tok = 0
    with open(OUT, "w", encoding="utf-8") as f:
        for r in client.messages.batches.results(st["id"]):
            if r.result.type != "succeeded":
                bad += 1
                continue
            msg = r.result.message
            text = next((b.text for b in msg.content if b.type == "text"), None)
            try:
                lab = json.loads(text)
            except (TypeError, json.JSONDecodeError):
                bad += 1
                continue
            in_tok += msg.usage.input_tokens + (msg.usage.cache_read_input_tokens or 0) + (msg.usage.cache_creation_input_tokens or 0)
            out_tok += msg.usage.output_tokens
            f.write(json.dumps({"ticket_id": r.custom_id, **lab}, ensure_ascii=False) + "\n")
            ok += 1
    print(f"ok={ok} failed={bad} input_tokens={in_tok} output_tokens={out_tok}")


def sync(workers: int = 16):
    """Same labelling with parallel direct calls (used when the batch queue is slow)."""
    from concurrent.futures import ThreadPoolExecutor
    client = anthropic.Anthropic(max_retries=4)
    sample_path = ROOT / "data" / "label_sample.jsonl"
    if sample_path.exists():
        sample = [(r["split"], r) for r in map(json.loads, open(sample_path, encoding="utf-8"))]
    else:
        sample = build_sample()
        sample_path.write_text("\n".join(json.dumps({"split": s, **t}, ensure_ascii=False) for s, t in sample), encoding="utf-8")
    done = set()
    if OUT.exists():
        done = {json.loads(l)["ticket_id"] for l in open(OUT, encoding="utf-8")}
    todo = [t for _, t in sample if t["ticket_id"] not in done]
    print("to label:", len(todo), flush=True)

    def one(t):
        try:
            msg = client.messages.create(
                model=MODEL, max_tokens=1024, thinking={"type": "between_tools"},
                output_config={"effort": "low", "format": {"type": "json_schema", "schema": SCHEMA}},
                system=[{"type": "text", "text": SYSTEM, "cache_control": {"type": "ephemeral"}}],
                messages=[{"role": "user", "content": ticket_prompt(t)}])
            lab = json.loads(next(b.text for b in msg.content if b.type == "text"))
            return {"ticket_id": t["ticket_id"], **lab}
        except Exception as e:  # keep going; failures are re-tried on the next run
            print("fail", t["ticket_id"], type(e).__name__, flush=True)
            return None

    n = 0
    with ThreadPoolExecutor(workers) as ex, open(OUT, "a", encoding="utf-8") as f:
        for r in ex.map(one, todo):
            if r:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
                n += 1
                if n % 250 == 0:
                    f.flush()
                    print("labelled", n, flush=True)
    print("done", n)


def submit_more(n: int = 2500):
    """Second batch: n more train tickets, stratified by the current classifier's prediction so the
    confusable classes (admin / invalid / other / dm) get more labels. Results -> data/labels_llm2.jsonl."""
    from app.labels import has_evidence_of
    from app.pipeline import MODELS, Classifier
    client = anthropic.Anthropic()
    taken = {json.loads(l)["ticket_id"] for l in open(ROOT / "data/label_sample.jsonl", encoding="utf-8")}
    train = [json.loads(l) for l in gzip.open(DATA / "tickets_train.jsonl.gz", "rt", encoding="utf-8")]
    rnd = random.Random(11)
    rnd.shuffle(train)
    clf = Classifier(MODELS / "clf.pkl")
    groups = defaultdict(list)
    for t in train[:40000]:
        if t["ticket_id"] in taken or not t["messages"]:
            continue
        text = normalize(player_text(t["messages"]), t.get("region"))
        cat, _ = clf.predict(t["kind"], text, t.get("rule_refs_raw") or "", has_evidence_of(text, t.get("images_count")))
        groups[cat].append(t)
    per = n // len(groups)
    picked = []
    for g in groups.values():
        picked += g[:per]
    rest = [t for g in groups.values() for t in g[per:]]
    rnd.shuffle(rest)
    picked += rest[: n - len(picked)]
    (ROOT / "data" / "label_sample2.jsonl").write_text(
        "\n".join(json.dumps({"split": "train", **t}, ensure_ascii=False) for t in picked), encoding="utf-8")
    requests = [{"custom_id": t["ticket_id"][:64], "params": {
        "model": MODEL, "max_tokens": 1024, "thinking": {"type": "between_tools"},
        "output_config": {"effort": "low", "format": {"type": "json_schema", "schema": SCHEMA}},
        "system": [{"type": "text", "text": SYSTEM, "cache_control": {"type": "ephemeral"}}],
        "messages": [{"role": "user", "content": ticket_prompt(t)}]}} for t in picked]
    batch = client.messages.batches.create(requests=requests)
    (ROOT / "data" / "label_batch2.json").write_text(json.dumps({"id": batch.id, "n": len(requests)}), encoding="utf-8")
    print("submitted", batch.id, len(requests))


def collect_more():
    global STATE, OUT
    STATE = ROOT / "data" / "label_batch2.json"
    OUT = ROOT / "data" / "labels_llm2.jsonl"
    collect()


if __name__ == "__main__":
    {"submit": submit, "collect": collect, "sync": sync, "submit_more": submit_more,
     "collect_more": collect_more}[sys.argv[1]]()
