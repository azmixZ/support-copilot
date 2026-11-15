"""Fast tests. The LLM is switched off, so they cost nothing and need no API key.

They need the case data in data/handout (not in the repository: the hackathon terms forbid publishing it).
A few tests check what the trained category classifier does; its weights are not published either, so without
models/clf.pkl those tests are skipped.
"""
import os
from pathlib import Path

os.environ["DRAFT_LLM"] = "0"

import pytest  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
if not (ROOT / "data" / "handout" / "rules.json").exists():
    pytest.skip("case data (data/handout) is not in the repository", allow_module_level=True)
needs_clf = pytest.mark.skipif(not (ROOT / "models" / "clf.pkl").exists(),
                               reason="needs the trained category classifier (weights are not published)")

from app.pipeline import Triage  # noqa: E402
from app.text import detect_lang, normalize  # noqa: E402

KEYS = {"lang", "category", "priority", "is_spam_or_duplicate", "extracted", "draft_reply", "needs_human",
        "confidence", "reasons"}


@pytest.fixture(scope="module")
def tr():
    return Triage()


def ticket(text, kind="complaint", **kw):
    return {"kind": kind, "messages": [{"role": "player", "text": text, **({"attachments": kw.pop("att")} if "att" in kw else {})}], **kw}


# --- language ---
def test_cp1251_kazakh_is_restored():
    assert normalize("Сєлем, кµмек ќажет", "kz") == "Сәлем, кµмек қажет"


def test_ukrainian_is_not_kazakh():
    assert detect_lang("переверніть машину") == "ru"
    assert detect_lang("Мен ойынға кіре алмай жүрмін") == "kk"
    assert detect_lang("iltimos yordam bering") == "uz"


# --- rules ---
def test_canonical_refs(tr):
    rb = tr.rules
    assert rb.canonical("2.1 оп") == "2.1"
    assert rb.canonical("ЖБ 1.2.7") == "ЖБ 1.2.7"
    assert rb.canonical("gz 1.1") == "GZ 1.1"
    assert rb.canonical("9.99") is None
    assert rb.validate(["2.1", "9.99", "ЖБ 7.7"]) == ["2.1"]


# --- contract and policy ---
def test_contract_example(tr):
    r = tr.triage(ticket("Arman_Petrov оскорблял меня и мою семью в чате, скрины есть, 2.1 ОП", att=2))
    assert KEYS <= r.keys()
    assert r["category"] == "complaint.insult"
    assert r["extracted"]["nickname"] == "Arman_Petrov"
    assert r["extracted"]["has_evidence"] is True
    assert r["needs_human"] is True


@pytest.mark.parametrize("text,kind", [
    ("оплатил донат, деньги списали, а гк не пришло", "report"),
    ("взломали аккаунт, пропала машина", "report"),
    ("не согласен с наказанием, админ выдал бан просто так, прошу разбан", "complaint"),
])
def test_always_human(tr, text, kind):
    assert tr.triage(ticket(text, kind))["needs_human"] is True


def test_every_citation_exists(tr):
    for text, kind in [("помогите", "report"), ("убил в зелёной зоне у больницы", "complaint"),
                       ("летает читер аимбот", "complaint"), ("жб", "complaint")]:
        r = tr.triage(ticket(text, kind))
        assert all(tr.rules.exists(c) for c in r["draft_reply"]["cited"])


def test_invented_rule_in_draft_is_replaced(tr):
    t = ticket("оскорбил в чате", att=1)
    ctx = tr.prepare(t)
    fake = {"draft_text": "Нарушение пункта 9.99 ОП, игрок получит бан.", "cited": ["9.99"], "nickname": None,
            "server": None, "needs_human": False, "confidence": 0.9, "reasons": []}
    r = tr.finish(t, ctx, fake, 0)
    assert "9.99" not in r["draft_reply"]["text"]
    assert all(tr.rules.exists(c) for c in r["draft_reply"]["cited"])


def test_duplicates_same_reporter_and_target(tr):
    base = dict(reporter_hash="r1", reported_hash="x1")
    a = tr.triage(ticket("оскорбил меня", created_at="2026-10-01T10:00:00", **base))
    b = tr.triage(ticket("оскорбил меня опять", created_at="2026-10-01T10:20:00", **base))
    c = tr.triage(ticket("оскорбил меня", created_at="2026-10-01T16:00:00", **base))
    assert (a["is_spam_or_duplicate"], b["is_spam_or_duplicate"], c["is_spam_or_duplicate"]) == (False, True, False)


def test_http_endpoint():
    from fastapi.testclient import TestClient
    from app.main import app
    r = TestClient(app).post("/triage", json={"kind": "report", "messages": [{"role": "player", "text": "застрял в текстурах"}]})
    assert r.status_code == 200
    assert KEYS <= r.json().keys()


@pytest.mark.parametrize("body", [
    {"kind": "report", "messages": []},
    {"kind": "report", "messages": [{"role": "player", "text": None}]},
    {"messages": [{"text": "застрял"}]},
    {"kind": "complaint", "messages": [{"role": "player", "text": "оскорбил " * 20000}]},
    {"kind": "report", "messages": [{"role": "player", "text": "😭 помогите 🙏"}]},
])
def test_messy_input_does_not_crash(body):
    from fastapi.testclient import TestClient
    from app.main import app
    r = TestClient(app).post("/triage", json=body)
    assert r.status_code == 200
    assert KEYS <= r.json().keys()


def test_negative_attachments_are_not_evidence(tr):
    r = tr.triage({"kind": "complaint", "messages": [{"role": "player", "text": "дм", "attachments": -3}]})
    assert r["extracted"]["has_evidence"] is False


def test_predict_survives_broken_lines(tmp_path):
    import subprocess
    import sys
    src, dst = tmp_path / "in.jsonl", tmp_path / "out.jsonl"
    src.write_text('{"ticket_id":"a","kind":"report","messages":[{"role":"player","text":"застрял"}]}\nnot json\n[1]\n',
                   encoding="utf-8")
    subprocess.run([sys.executable, "predict.py", f"IN={src}", f"OUT={dst}"], check=True,
                   env={**os.environ, "DRAFT_LLM": "0"})
    import json
    out = [json.loads(l) for l in dst.read_text(encoding="utf-8").splitlines()]
    assert len(out) == 3 and out[0]["ticket_id"] == "a" and "_meta" not in out[0]


@pytest.mark.parametrize("extra,text,expected", [
    ({"region": "kz"}, "оскорбил меня", "kz"),
    ({"region": "CIS"}, "оскорбил меня", "cis"),
    ({}, "на 15 сервере убил без причины", None),   # a server number is not a region
    ({}, "оскорбил на kz сервере", "kz"),
    ({}, "Мен ойыншыға шағым жазамын", None),       # language alone doesn't define the region
])
def test_extracted_server_is_region(tr, extra, text, expected):
    r = tr.triage({"kind": "complaint", "messages": [{"role": "player", "text": text}], **extra})
    assert r["extracted"]["server"] == expected


def test_p0_by_taxonomy(tr):
    # account / payment / threats outside the game -> p0, as in taxonomy.json
    for text in ("оплатил донат, не пришло", "взломали аккаунт", "найду тебя в реале, знаю где ты живешь"):
        assert tr.triage({"kind": "report", "messages": [{"role": "player", "text": text}]})["priority"] == "p0"
    # a mass bug can't be recognised from a single ticket
    assert tr.triage({"kind": "report", "messages": [{"role": "player", "text": "у всех сервер упал"}]})["priority"] != "p0"

def test_account_without_access_problem_is_not_p0():
    from app.labels import priority_for
    assert priority_for("report.account", "хочу поменять ник на нормальный", False) == "p1"
    assert priority_for("report.account", "взломали аккаунт", False) == "p0"


def test_other_complaint_does_not_cite_insult_rule(tr):
    r = tr.triage({"kind": "complaint", "messages": [{"role": "player", "text": "1.6 оп, скрин прикрепил, прошу проверить"}],
                   "images_count": 1})
    if r["category"] == "complaint.other":
        assert "2.1" not in r["draft_reply"]["cited"]
        assert r["draft_reply"]["cited"][:1] == ["1.6"]


def test_reports_about_a_player_by_id_go_to_human(tr):
    for text in ("435 дб", "138 софт", "347ид задрал"):
        r = tr.triage({"kind": "report", "messages": [{"role": "player", "text": text}]})
        assert r["category"] == "report.complaint" and r["needs_human"] is True


def test_draft_naming_a_punishment_is_replaced(tr):
    t = ticket("оскорбил в чате", att=1)
    ctx = tr.prepare(t)
    llm = {"draft_text": "Здравствуйте! Нарушитель получит мут на 30 минут по пункту 2.1 ОП.", "cited": ["2.1"],
           "nickname": None, "needs_human": True, "confidence": 0.9, "reasons": []}
    r = tr.finish(t, ctx, llm, 0)
    assert "мут на 30" not in r["draft_reply"]["text"]


def test_uncoded_sections_cited_by_full_title(tr):
    assert tr.rules.canonical("Правила для ОПГ 1.3") == "Правила для ОПГ 1.3"
    assert tr.rules.canonical("правила войны за бизнес 1.1.1") == "Правила войны за бизнес 1.1.1"
    assert not any(c.startswith(("ОПГ ", "ВБ ", "ЛИД ")) for c in tr.rules.by_cite)


@pytest.mark.parametrize("text,region,lang", [
    ("osk rodni", "cis", "ru"), ("ne soflasem", "cis", "ru"),
    ("Meni haqorat qildi, jazo berilishini so'rayman", "uz", "uz"), ("мен учун жазо керак эмас", "uz", "uz"),
    ("я не согласен с наказанием", "uz", "ru"), ("Salem, maǵan kómek kerek", None, "kk"),
])
def test_translit_and_scripts(text, region, lang):
    assert detect_lang(text, region) == lang


def test_implicit_duplicates(tr):
    # Same reporter, same accused, different wording -> duplicate.
    a = tr.triage(ticket("он меня оскорбил", created_at="2026-10-02T10:00:00", reporter_hash="r9", reported_hash="x9"))
    b = tr.triage(ticket("вот ещё скрины, родню трогал", created_at="2026-10-02T10:05:00", reporter_hash="r9",
                         reported_hash="x9"))
    # Different accused player -> not a duplicate even from the same reporter.
    c = tr.triage(ticket("а этот дм у банка", created_at="2026-10-02T10:10:00", reporter_hash="r9", reported_hash="y9"))
    assert (a["is_spam_or_duplicate"], b["is_spam_or_duplicate"], c["is_spam_or_duplicate"]) == (False, True, False)


def test_history_lookup_and_six_month_retention(tmp_path):
    import json
    from datetime import datetime
    from app.history import History
    src = tmp_path / "t.jsonl"
    rows = [
        {"ticket_id": "old", "kind": "complaint", "created_at": "2025-01-01T10:00:00", "reporter_hash": "r", "reported_hash": "x",
         "messages": [{"role": "player", "text": "давно"}]},
        {"ticket_id": "new", "kind": "complaint", "created_at": "2026-10-01T10:00:00", "reporter_hash": "r", "reported_hash": "x",
         "messages": [{"role": "player", "text": "недавно"}]},
    ]
    src.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")
    History.build([src]).save(tmp_path / "h.pkl")
    h = History.load(tmp_path / "h.pkl", now=datetime(2026, 10, 31))
    got = h.lookup("r", "x")
    assert [r["ticket_id"] for r in got["against_accused"]] == ["new"]   # older than 6 months is dropped
    assert got["from_reporter_total"] == 1


def test_spam_definition(tr):
    def spam(text, kind="report"):
        return tr.triage({"kind": kind, "messages": [{"role": "player", "text": text}]})["is_spam_or_duplicate"]
    assert spam("ау") and spam("") and spam("пж пж пж пж пж") and spam("ааааааааа")
    assert not spam("админ ты сука")          # abuse towards admins is not spam
    assert not spam("помогите застрял")


@pytest.mark.parametrize("text,expected", [
    ("дал дмг ни за что, я просто стоял", "complaint.admin"),
    ("меня посадили в деморган просто так", "complaint.admin"),
    ("убил без причины, требую посадить его в деморган", "complaint.dm"),
])
def test_dmg_is_admin_jail_not_dm(tr, text, expected):
    assert tr.triage(ticket(text, att=1))["category"] == expected


@needs_clf
def test_cases_found_in_admin_ui(tr):
    def run(kind, text, **kw):
        return tr.triage({"kind": kind, "messages": [{"role": "player", "text": text}], **kw})
    # a random code in the rule field must not become a violation in the draft
    assert "DB" not in run("complaint", "Разбаньте его, он нормальный игрок", rule_refs_raw="дб", images_count=1)["draft_reply"]["text"]
    # tow-truck tip only for vehicles
    assert "эвакуатор" not in run("report", "Тп ко мне, надо поговорить")["draft_reply"]["text"]
    # "osk" with a screenshot is an insult complaint
    assert run("complaint", "osk", images_count=1)["category"] == "complaint.insult"
    # quoting the accused's messages is evidence, not the reporter's own abuse
    quoted = run("complaint", "Test_Player7 лох\n\nTest_Player7 чмо\n\nTest_Player7 черт", images_count=1)
    assert quoted["category"] == "complaint.insult" and "ЖБ 1.2.7" not in quoted["draft_reply"]["cited"]


def test_bare_rule_number_and_how_to_complain(tr):
    r = tr.triage({"kind": "complaint", "images_count": 1, "messages": [{"role": "player", "text": "5.2 оп"}]})
    assert r["category"] == "complaint.cheat"
    r = tr.triage({"kind": "report", "messages": [{"role": "player", "text": "куда кинуть жалобу на админа"}]})
    assert "лаунчер" in r["draft_reply"]["text"] and r["draft_reply"]["cited"] == ["ЖБ 1.2.1"]


def test_kazakh_reply_is_always_cyrillic(tr):
    r = tr.triage({"kind": "complaint", "images_count": 1, "region": "kz",
                   "messages": [{"role": "player", "text": "Salem, Arman_Petrov meni balaǵattap sókti"}]})
    assert r["draft_reply"]["lang"] == "kk" and any(c in r["draft_reply"]["text"] for c in "әғқңөұүі")


def test_categories_with_rules_cite_them(tr):
    for text, cite in (("взломали аккаунт", "1.4"), ("перевернулась машина", "3.1"), ("помогите", "3.1")):
        assert cite in tr.triage({"kind": "report", "messages": [{"role": "player", "text": text}]})["draft_reply"]["cited"]


def test_duplicates_use_only_the_service_own_history(tr):
    # train / valid are not a history for new tickets (organisers): the first complaint is never a duplicate,
    # a repeat of the same complaint the service has already seen within an hour is.
    base = {"kind": "complaint", "reporter_hash": "rep-own", "reported_hash": "acc-own", "images_count": 1,
            "messages": [{"role": "player", "text": "он опять оскорбляет"}]}
    first = tr.triage({**base, "ticket_id": "own-1", "created_at": "2031-01-01T10:00:00"})
    second = tr.triage({**base, "ticket_id": "own-2", "created_at": "2031-01-01T10:20:00"})
    assert first["is_spam_or_duplicate"] is False
    assert second["is_spam_or_duplicate"] is True


def test_priority_does_not_depend_on_screenshots(tr):
    for att in (0, 2):
        assert tr.triage(ticket("оск родни", att=att))["priority"] == "p2"


@needs_clf
@pytest.mark.parametrize("text", ["опру", "не согл", "Где доки мои?", "так кого я убил?", "без читов я играю"])
def test_appeal_slang_is_complaint_about_admin(tr, text):
    assert tr.triage(ticket(text, att=1))["category"] == "complaint.admin"

def test_flipped_vehicle_in_kk_uz_is_help(tr):
    for region, text in (("kz", "көлігім аударылып қалды"), ("uz", "mashinam ag'darilib qoldi"), (None, "переверните")):
        r = tr.triage({"kind": "report", "region": region, "messages": [{"role": "player", "text": text}]})
        assert r["category"] == "report.help"


def test_uzbek_complaint_draft_follows_admin_format(tr):
    r = tr.triage({"kind": "complaint", "region": "uz", "reported_nick": "Jasur_Karim", "images_count": 1,
                   "messages": [{"role": "player", "text": "onamni so'kdi haqorat qildi"}]})
    text = r["draft_reply"]["text"]
    assert text.startswith("Hurmatli") and text.rstrip().endswith("Ma'muriyati")


def test_kazakh_complaint_is_about_the_player(tr):
    r = tr.triage({"kind": "complaint", "region": "kz", "reported_nick": "Arman_Petrov", "images_count": 1,
                   "messages": [{"role": "player", "text": "анамды сөкті"}]})
    assert "Arman_Petrov ойыншысына қатысты шағым" in r["draft_reply"]["text"]

def test_categories_with_taxonomy_rules_always_cite_a_rule(tr):
    # taxonomy.json lists 3.1 for report.offtopic: even a greeting cites it ("ответ без цитат только где правила не нужны")
    for text in ("Добрый вечер:)", "хорошего настроения", "сос", "ау привет"):
        r = tr.triage(ticket(text, kind="report"))
        if r["category"] in ("report.offtopic", "report.help", "report.account", "complaint.admin", "complaint.invalid"):
            assert r["draft_reply"]["cited"], text

@pytest.mark.parametrize("text", ["OSK ROD", "osk rod", "Osk.rod", "2.1 оп мут дайте", "5.2 оп", "1.1 gz"])
def test_named_violation_is_not_an_empty_complaint(tr, text):
    # organisers: category by the essence; invalid only when the text names no violation
    assert tr.triage(ticket(text, att=1))["category"] != "complaint.invalid"


@pytest.mark.parametrize("text", ["жб", ".", "вот он", "как купить дом"])
def test_empty_complaints_stay_invalid(tr, text):
    assert tr.triage(ticket(text, att=1))["category"] == "complaint.invalid"


def test_appeal_with_rule_number_stays_appeal(tr):
    assert tr.triage(ticket("не согласен с мутом 2.1", att=1))["category"] == "complaint.admin"

def test_no_template_promises_actions_the_service_cannot_do():
    # "слежу за игроком", "проверим", "администратор поможет": the service can't do it, so the draft only says
    # what is required ("требуется проверка администратора"), in every language
    from app.draft import PROMISE
    from app.local_draft import T
    for lang, table in T.items():
        for key, text in table.items():
            if isinstance(text, str):
                assert not PROMISE.search(text), (lang, key, text)


@pytest.mark.parametrize("kind,text", [("report", "393 дм"), ("report", "помогите"), ("report", "как продать машину?"),
                                       ("report", "оплатил донат, не пришло"), ("complaint", "не согласен с мутом, админ выдал просто так")])
def test_drafts_do_not_promise_actions(tr, kind, text):
    from app.draft import PROMISE
    r = tr.triage(ticket(text, kind=kind, att=1) if kind == "complaint" else ticket(text, kind=kind))
    assert not PROMISE.search(r["draft_reply"]["text"]), r["draft_reply"]["text"]
