"""Triage pipeline: normalise -> classify -> dedup/spam -> extract facts -> RAG -> draft -> guardrails."""
import json
import os
import pickle
import re
import threading
import time
from collections import defaultdict, deque
from datetime import datetime
from pathlib import Path

from .draft import Drafter, has_promise, names_punishment, rule_mentions
from .knowledge import KnowledgeBase
from .labels import (NEEDS_HUMAN_ALWAYS, complaint_category, has_evidence_of, has_threat, is_jail_appeal, is_report_on_player,
                     STUCK, priority_for, report_category, urgent_report_category)
from .local_draft import GREETING, local_draft
from .rulepick import CATEGORY_RULES, RulePicker
from .rules import RuleBase
from .text import detect_lang, normalize, player_text, script_of
from .verdict import VerdictModel

# An appeal verdict means the player disputed a punishment. "Оформлено неверно" is not mapped: the category follows
# the essence of the violation ("оск родни" is an insult even if the admin closed it for the format, organisers).
ADMIN_VERDICT_CATEGORY = {"appeal_approved": "complaint.admin", "appeal_rejected": "complaint.admin"}
MAIN_RULE_MIN_CONF = 0.5
# Main rule when the rule model isn't sure: what admins cite most for the category.
CATEGORY_MAIN_RULE = {"complaint.insult": "2.1", "complaint.dm": "дм", "complaint.cheat": "5.2",
                      "complaint.scam": "4.1", "complaint.rmt_ads": "4.4"}
APPEAL_VERDICT_CONF = 0.5  # the verdict model says "appeal" (approved + rejected) at least this likely
VERDICT_HINT_CONF = 0.8  # verdict right ~85% of the time at this confidence on validation
VERDICT_RU = {"approved": "одобрена, нарушитель наказан", "need_evidence": "нужны доказательства",
              "rejected": "нарушений нет", "invalid_format": "оформлена неверно", "duplicate": "дубль",
              "already_punished": "игрок уже наказан", "appeal_rejected": "наказание выдано верно",
              "appeal_approved": "наказание снято"}

ROOT = Path(__file__).resolve().parent.parent
DATA = Path(os.getenv("DATA_DIR", ROOT / "data" / "handout"))
MODELS = Path(os.getenv("MODELS_DIR", ROOT / "models"))

NICK = re.compile(r"\b[A-Z][a-z]+_[A-Z][a-z]+\d{0,3}\b")
# extracted.server is the server REGION (as in the dataset field `region`), not the server number.
# Taken from the request when given; otherwise only from an explicit mention in the text — never guessed
# from the language (a Kazakh-speaking player may well play on a CIS server).
REGIONS = {"cis", "kz", "uz", "eu", "latam"}
REGION_MENTIONS = [
    ("kz", r"\b(kz|кз)\b[\s-]*(сервер|server|серв)|(сервер|server|серв)\w*[\s-]*(kz|кз|казах)|казахстанск\w* (сервер|серв)"),
    ("uz", r"\b(uz|уз)\b[\s-]*(сервер|server|серв)|(сервер|server|серв)\w*[\s-]*(uz|уз|узб)|узбекск\w* (сервер|серв)"),
    ("eu", r"\b(eu|европ\w*)\b[\s-]*(сервер|server|серв)|(сервер|server|серв)\w*[\s-]*(eu|европ)"),
    ("latam", r"\b(latam|латам|br|brasil|brazil)\b[\s-]*(сервер|server|servidor)|servidor"),
    ("cis", r"\b(снг|cis)\b[\s-]*(сервер|server|серв)|(сервер|server|серв)\w*[\s-]*(снг|cis)"),
]


def region_of(t: dict, text: str) -> str | None:
    r = (t.get("region") or "").strip().lower()
    if r in REGIONS:
        return r
    for region, rx in REGION_MENTIONS:
        if re.search(rx, text or "", re.IGNORECASE):
            return region
    return None
ALWAYS_HUMAN_TOPICS = re.compile(
    r"разбан|unban|сним\w* (бан|мут|наказ|блок)|верн\w* (деньг|акк|машин|имущ|дом|вещ)|возврат|компенсац|"
    r"апелляц|не согласен|взлом|hacked|украл\w* акк|оплат|донат|платеж|платёж|списал", re.IGNORECASE)

# Rules cited by the taxonomy for each category; "GZ" means the whole green-zone section.
EXTRA_CITES = {
    "complaint.invalid": ["ЖБ 1.1.2", "ЖБ 1.2.3", "ЖБ 1.2.7", "ЖБ 1.3.4"],
    "complaint.admin": ["3.4", "ЖБ 1.2.6", "ЖБ 1.5.2"],
    "report.offtopic": ["3.1", "3.3"],
    "report.help": ["3.1"],
    "report.complaint": ["ЖБ 1.1.1"],
    "report.account": ["1.4", "1.7"],
}


SHORT_INSULT = re.compile(r"^\W*(оск\w*|osk\w*|оскорб\w*)\.?([\s.,:;-]+(родн\w*|род|rodn\w*|rod|мат\w*|семь\w*|в чате|"
                          r"chat\w*)\.?)?\W*$", re.IGNORECASE)  # "оск", "OSK ROD", "Osk.rod", "оск родни"
# Kazakh / Uzbek words for "insulted / swore at": the classifier saw few of them in the labelled sample.
KK_UZ_INSULT = re.compile(r"балағат|боқтад|сөкті|сөгіп|қорлад|қорлап|balaǵat|balagat|sókti|sokti|qorlad|"
                          r"haqorat|xaqorat|so'?k(di|yapti|kan|ib)|"
                          r"sokdi|sokyap|onamni|onamga|ота-анам", re.IGNORECASE)
# Player slang for disputing their own punishment: "опру" (= оспорю), "не согл", "где доки?", "за что?", "без читов я
# играю". Admins close such tickets as appeals (complaint.admin); the classifier, trained on text-only labels, often
# took them for empty complaints. On validation this lifts agreement with the admin's own verdicts from 76% to 81%.
APPEAL_SLANG = re.compile(r"(?<!\w)опру(?!\w)|опровер|не\s*согл|ne\s*sogl|где\s+(мои\s+)?док|(?<!\w)докв|за\s+что(?!\w)|"
                          r"кого\s+(же\s+)?я\s+убил|где\s+я\s+\w+|нет\w*\s+(у\s+меня\s+)?софт|без\s+читов|"
                          r"я\s+не\s+(нарушал|читер|читерил|софтер|убивал)|в\s+каком\s+месте|проверьте\s+(мои\s+)?логи|"
                          r"по\s+логам", re.IGNORECASE)
UNBAN_REQUEST = re.compile(r"разбан|razban|сним\w* (бан|наказ)|unban", re.IGNORECASE)
QUOTED_LINE = re.compile(r"^\s*[A-Z][a-z]+_[A-Z][a-z]+\d{0,3}\s*[:\-—]?\s+\S", re.MULTILINE)


def cite_of_violation(violation: str, rules) -> str | None:
    from .verdict import violation_key
    key = violation_key(violation, rules.canonical)
    if not key:
        return None
    return key if key in rules.by_cite else rules.term(key)


def category_of_cite(cite: str | None, rules) -> str | None:
    """Rule citation -> complaint category (by the taxonomy's rules for each category)."""
    if not cite:
        return None
    if cite.startswith(("GZ", "RP ")) or cite in ("5.3", "5.4"):
        return "complaint.dm"
    if cite in ("2.1", "2.4", "2.6"):
        return "complaint.insult"
    if cite == "5.2":
        return "complaint.cheat"
    if cite in ("4.1", "4.2"):
        return "complaint.scam"
    if cite in ("2.3", "4.4"):
        return "complaint.rmt_ads"
    return None


NUM_REF = re.compile(r"(?<![\d.])\d{1,2}[.,]\d{1,2}(?:[.,]\d{1,2})?\s*(?:опг|оп|op|гз|gz|зз|пг)?", re.IGNORECASE)


def category_of_rule_in_text(text: str, rules) -> str | None:
    """The first rule number in the text that names a specific violation ("2.1", "5.2 оп", "1.1 gz", "2.1 оп мут
    дайте"): category from the rule."""
    for m in NUM_REF.finditer(re.sub(r"<\w+>", " ", text or "")):
        cite = rules.canonical(m.group(0))
        cat = category_of_cite(cite, rules) if cite else None
        if cat:
            return cat
    return None


def quotes_insults(text: str) -> bool:
    """Two or more lines starting with a nickname: the reporter quotes the accused's chat messages."""
    from .local_draft import ABUSE
    return len(QUOTED_LINE.findall(text or "")) >= 2 and ABUSE.search(text or "") is not None


def is_gibberish(text: str) -> bool:
    t = re.sub(r"<\w+>", "", text or "").strip().lower()
    letters = re.sub(r"[^\w]", "", t)
    if len(letters) < 12:
        return False
    words = t.split()
    longest = max((len(w) for w in words), default=0)
    vowels = sum(c in "аеёиоуыэюяaeiouәөұүі" for c in letters)
    return longest >= 25 or vowels / max(len(letters), 1) < 0.18


# Spam as the organisers define it: "ау"-style pings, flood, empty tickets. Abuse towards admins is NOT spam.
PING = re.compile(r"^\W*(ау+|алл?о+|эй+|э+|а+|ы+|ку+|ауу+|хм+|ok|ок|\d{1,2}|au+|alo+|hello\?*|hey)\W*$", re.IGNORECASE)


def is_flood(text: str) -> bool:
    t = (text or "").lower()
    if re.search(r"(.)\1{5,}", t) or re.search(r"(\w{2,4})\1{3,}", t):
        return True
    words = re.findall(r"\w+", t)
    return len(words) >= 4 and max(words.count(w) for w in set(words)) / len(words) >= 0.6


def is_spam(kind: str, text: str) -> bool:
    content = re.sub(r"<\w+>|\W", "", text or "")
    if kind == "report" and (not content or PING.match(text or "")):
        return True
    return is_flood(text) or is_gibberish(text)


class DedupStore:
    """1 player, 1 problem, many tickets -> duplicates (implicit ones too: the wording may differ completely).

    Organisers evaluate is_spam_or_duplicate for complaints against the admin verdict "duplicate". Tuned on the
    full train (scripts/eval_dedup.py) for F1 against that verdict:
      * complaint: the same reporter already complained about the same accused player within 60 minutes;
      * report: the same reporter sent a similar report within 60 minutes ("помогите" x5).
    Text cues like "пишу повторно" were dropped: 93% of them are not admin-marked duplicates.
    """

    def __init__(self, window_min: int = 60, report_sim: float = 0.6):
        self.window = window_min * 60
        self.report_sim = report_sim
        self.recent = defaultdict(deque)
        self.lock = threading.Lock()

    @staticmethod
    def _shingles(text: str) -> set:
        t = re.sub(r"\W+", " ", (text or "").lower()).strip()
        return {t[i:i + 3] for i in range(max(len(t) - 2, 1))}

    def check(self, reporter: str | None, reported: str | None, kind: str, text: str, ts: float) -> bool:
        if not reporter:
            return False
        sh = self._shingles(text)
        with self.lock:
            q = self.recent[reporter]
            while q and ts - q[0][0] > self.window:
                q.popleft()
            dup = False
            for (t0, k0, rep0, sh0) in q:
                if abs(ts - t0) > self.window or k0 != kind:
                    continue
                if kind == "complaint" and reported and rep0 == reported:
                    dup = True
                elif kind == "report" and len(sh & sh0) / max(len(sh | sh0), 1) >= self.report_sim:
                    dup = True
                if dup:
                    break
            q.append((ts, kind, reported, sh))
            return dup


class Classifier:
    def __init__(self, path: Path):
        self.model = None
        if path.exists():
            with open(path, "rb") as f:
                self.model = pickle.load(f)

    @staticmethod
    def features(kind: str, text: str, refs_raw: str, has_ev: bool) -> str:
        return f"[{kind}] [ev:{int(has_ev)}] [refs:{refs_raw or ''}] {text}"

    def predict(self, kind: str, text: str, refs_raw: str, has_ev: bool) -> tuple[str, float]:
        rule_cat = complaint_category(text, refs_raw) if kind == "complaint" else report_category(text)
        if self.model is None:
            return rule_cat, 0.6
        proba = self.model.predict_proba([self.features(kind, text, refs_raw, has_ev)])[0]
        classes = list(self.model.classes_)
        best, best_p = rule_cat, 0.0
        for c, p in zip(classes, proba):
            if c.startswith(kind + ".") and p > best_p:
                best, best_p = c, p
        total = sum(p for c, p in zip(classes, proba) if c.startswith(kind + "."))
        return best, float(best_p / total) if total else 0.5


class Triage:
    def __init__(self):
        self.rules = RuleBase(DATA / "rules.json")
        self.kb = KnowledgeBase.load(MODELS / "kb.pkl") if (MODELS / "kb.pkl").exists() else None
        self.clf = Classifier(MODELS / "clf.pkl")
        self.dedup = DedupStore()
        self.drafter = Drafter()
        self.verdicts = VerdictModel.load(MODELS)
        self.rulepick = RulePicker.load(MODELS)
        tax = json.loads((DATA / "taxonomy.json").read_text(encoding="utf-8"))
        self.tax_rules = {c["code"]: c["rules"] for c in tax["categories"]}
        self.gz_all = [p.cite for p in self.rules.points if p.cite.startswith("GZ 1.")]

    # ---------------- helpers ----------------
    def candidate_cites(self, category: str, text: str, player_refs: list[str]) -> list[str]:
        out: list[str] = []

        def add(c):
            c = self.rules.canonical(c) if c not in self.rules.by_cite else c
            if c and c in self.rules.by_cite and c not in out:
                out.append(c)

        for r in self.tax_rules.get(category, []):
            if r == "GZ":
                for c in self.gz_all[:3]:
                    add(c)
            else:
                add(r)
        for c in EXTRA_CITES.get(category, []):
            add(c)
        for c in player_refs[:3]:
            # Players often type irrelevant numbers; only accept refs from the violation sections.
            if category.startswith("complaint.") and (c.split(".")[0] in ("2", "4", "5") or c.startswith(("GZ", "RP "))
                                                       or c == "1.6"):
                add(c)
        # Retrieval adds rules from the core sections only: violations (ОП 2/4/5, GZ) for complaints,
        # report/complaint procedure (ОП 1/3, ЖБ) for everything else.
        if category.startswith("complaint.") and category not in ("complaint.invalid", "complaint.admin"):
            add("ЖБ 1.2.2")
            allowed = {"2", "4", "5", "GZ"}
        else:
            allowed = {"1", "3", "ЖБ"}
        for p in self.rules.search(text, k=12):
            if len(out) >= 10:
                break
            head = p.cite.split(" ")[0] if " " in p.cite else p.cite.split(".")[0]
            if p.section_id in (30, 31, 32) and head in allowed:
                add(p.cite)
        return out[:10]

    def resolve_mention(self, mention: str, allowed: list[str]) -> str | None:
        """A rule number written in the draft must be one of the verified candidate citations.
        Sections without a code are written as "Правила для ОПГ 1.3", so a bare number may match
        the end of a titled citation."""
        c = self.rules.canonical(mention)
        if c in allowed:
            return c
        num = re.search(r"\d+(?:\.\d+)+", mention).group(0)
        titled = [a for a in allowed if a.endswith(" " + num) and not a.startswith(("ЖБ ", "GZ "))]
        return titled[0] if len(titled) == 1 else None

    @staticmethod
    def nothing_to_do(ctx: dict) -> bool:
        """True only when an admin clearly has nothing to do: spam (ping, flood, empty) or a greeting."""
        return ctx["is_spam"] or GREETING.match(ctx["text"]) is not None

    @staticmethod
    def _ts(t: dict) -> float:
        try:
            return datetime.fromisoformat(t["created_at"]).timestamp()
        except (KeyError, ValueError, TypeError):
            return time.time()

    # ---------------- main ----------------
    def prepare(self, t: dict) -> dict:
        """Everything except the LLM call (fast, deterministic)."""
        kind = t.get("kind") or "report"
        region = t.get("region")
        messages = t.get("messages") or []
        raw = player_text(messages)
        text = normalize(raw, region)
        attachments = t.get("images_count")
        if attachments is None:
            attachments = sum(max(int(m.get("attachments") or 0), 0) for m in messages
                              if (m.get("role") or "player") == "player")
        attachments = max(int(attachments or 0), 0)
        has_ev = has_evidence_of(text, attachments)
        refs_raw = t.get("rule_refs_raw") or ""
        lang = detect_lang(raw, region)
        if lang == "und":
            lang_reply = {"uz": "uz", "kz": "kk"}.get(region or "", "ru")
        else:
            lang_reply = lang

        category, cat_conf = self.clf.predict(kind, text, refs_raw, has_ev)
        if kind == "report" and category in ("report.offtopic", "report.question") and is_report_on_player(text):
            category = "report.complaint"  # "435 дб": a report about a player by in-game ID
        if kind == "report" and category in ("report.offtopic", "report.question") and STUCK.search(text):
            category = "report.help"  # "переверните", "көлігім аударылып қалды": stuck or flipped, needs help
        verdict = t.get("verdict")
        if kind == "complaint" and verdict in ADMIN_VERDICT_CATEGORY:
            # The service is normally called before the admin answers; if a record does carry an appeal verdict,
            # the player disputed a punishment.
            category = ADMIN_VERDICT_CATEGORY[verdict]
        elif kind == "complaint" and t.get("violation"):
            # The admin already named the violation ("2.1 ОП", "DM", "1.2 GZ"): the category follows from it.
            by_violation = category_of_cite(cite_of_violation(t["violation"], self.rules), self.rules)
            category = by_violation or category
        if kind == "complaint" and category == "complaint.invalid" and (
                SHORT_INSULT.match(re.sub(r"<\w+>", " ", text)) or quotes_insults(raw) or KK_UZ_INSULT.search(text)):
            # "оск" / "osk" / "оск родни" with a screenshot is an insult complaint (85% in the labelled sample);
            # "Nick лох / Nick чмо / ..." line by line quotes what the accused wrote, it is not the reporter's abuse.
            category = "complaint.insult"
        if kind == "complaint" and category == "complaint.invalid" and APPEAL_SLANG.search(text):
            category = "complaint.admin"  # "опру", "не согл", "где доки?": the player disputes their punishment
        if kind == "complaint" and category != "complaint.admin" and is_jail_appeal(text):
            category = "complaint.admin"  # "дал дмг ни за что": DMG is the admin jail (demorgan), not DM
        verdict_p = self.verdicts.proba({**t, "images_count": attachments}) if kind == "complaint" else {}
        if (kind == "complaint" and category != "complaint.admin" and verdict not in ADMIN_VERDICT_CATEGORY
                and not t.get("violation")
                and verdict_p.get("appeal_rejected", 0) + verdict_p.get("appeal_approved", 0) >= APPEAL_VERDICT_CONF):
            # Past admin answers to texts like this one were appeal verdicts: the player disputes a punishment.
            # On validation: agreement with the admin-answer category 85.8% -> 88.3% (silver labels -0.3%).
            category = "complaint.admin"
        if kind == "complaint" and category == "complaint.invalid" and (by_rule := category_of_rule_in_text(text, self.rules)):
            # "2.1", "5.2 оп", "2.1 оп мут дайте": the player named a violation, so it is not an empty complaint
            # (organisers: invalid only when the text names no violation). Train: 92.1% -> 92.3% vs admin answers.
            category = by_rule
        if kind == "report" and (urgent := urgent_report_category(text)) and category not in (urgent,):
            category = urgent  # hacking / payment: never miss these
        priority = priority_for(category, text, has_ev)

        player_refs = self.rules.parse_player_refs(f"{refs_raw} {text}")
        nick = t.get("reported_nick") or (NICK.findall(text) or [None])[0]
        server = region_of(t, text)

        ts = self._ts(t)
        is_dup = self.dedup.check(t.get("reporter_hash"), t.get("reported_hash"), kind, text, ts)
        # The admin's own verdict, when the record already has it, is the ground truth for duplicates.
        if kind == "complaint" and verdict == "duplicate":
            is_dup = True
        spam = is_spam(kind, text)

        cites = self.candidate_cites(category, text, player_refs)
        # Main rule of a violation complaint: a local model trained on the rules admins named (90% on valid).
        main_cite = None
        if category.startswith("complaint.") and category not in ("complaint.invalid", "complaint.admin"):
            rule, p = self.rulepick.predict(t, category)
            named = self.rules.term(f"{refs_raw} {text}") if category == "complaint.dm" else None
            admin_cite = cite_of_violation(t["violation"], self.rules) if t.get("violation") else None
            own = next((r for r in player_refs if r in self.rules.by_cite and CATEGORY_RULES["complaint.other"](r)), None)
            if admin_cite:
                main_cite = admin_cite  # the admin already named the violation
            elif category == "complaint.other" and own:
                main_cite = own  # "1.6 оп": an "other" violation the player named, and the rule exists
            elif rule in self.rules.by_cite and p >= MAIN_RULE_MIN_CONF:
                main_cite = rule
            elif named:
                main_cite = named  # the player named the violation: "дм", "db", "нрп"
            else:
                main_cite = self.rules.canonical(CATEGORY_MAIN_RULE.get(category, "")) or None
            if main_cite:
                cites = [main_cite] + [c for c in cites if c != main_cite]
        elif category == "complaint.invalid" and has_ev:
            # "NRP KOP: <url>", "фаст кулак": short, but it does name a violation and has evidence. The category
            # stays (that's the classifier's call); only the draft answers it as a normal complaint.
            rule, p = self.rulepick.predict(t)
            # Only the text counts here: players often type a random code ("дб") into the rule field.
            named = None if UNBAN_REQUEST.search(text) else self.rules.term(text)
            # 2.1 is the most frequent rule overall, so on a near-empty text it is a guess, not a signal.
            if named or (rule in self.rules.by_cite and p >= 0.8 and rule != "2.1" and not UNBAN_REQUEST.search(text)):
                main_cite = named or rule
                cites = [main_cite] + [c for c in cites if c != main_cite]
        # Few-shot examples are only for the optional API backend; the local drafts search the KB by themselves
        # (questions only), so the search is skipped here — it is the slowest step of the pipeline.
        examples = (self.kb.similar(kind, text, k=4, lang=t.get("lang") or lang)
                    if self.kb and self.drafter.enabled else [])

        # Likely admin verdict: shown to the admin as a hint only. Putting it into the draft text lowered
        # blind-judge scores on validation (3.25 -> 2.9), so the draft stays neutral (see REPORT.md).
        verdict_hint = None
        if verdict_p:
            v = max(verdict_p, key=verdict_p.get)
            if verdict_p[v] >= VERDICT_HINT_CONF:
                verdict_hint = (v, verdict_p[v])

        return dict(kind=kind, text=text, lang=lang, lang_reply=lang_reply, script=script_of(text),
                    category=category, cat_conf=cat_conf,
                    priority=priority, has_evidence=has_ev, attachments=attachments, player_refs=player_refs,
                    nick=nick, server=server, is_dup=is_dup, is_spam=spam, cites=cites, main_cite=main_cite,
                    examples=examples,
                    verdict_hint=verdict_hint)

    def finish(self, t: dict, ctx: dict, llm: dict | None, latency_ms: float) -> dict:
        category = ctx["category"]
        reasons: list[str] = []
        cost = 0.0
        if llm and llm.get("draft_text"):
            draft_text = llm["draft_text"].strip()
            cited = self.rules.validate(llm.get("cited") or [])
            # Any rule number mentioned in the text must be a validated citation.
            if names_punishment(draft_text):
                llm = None
                reasons.append("черновик называл наказание, заменён нейтральным шаблоном")
        if llm and llm.get("draft_text"):
            allowed = list(dict.fromkeys(cited + ctx["cites"]))
            resolved = [self.resolve_mention(x, allowed) for x in rule_mentions(draft_text)]
            if all(resolved):
                cited = self.rules.validate(cited + [c for c in resolved if c not in cited])
            else:
                llm = None
                reasons.append("черновик ссылался на непроверенный пункт, заменён шаблоном")
            cost = llm.get("_cost_usd", 0.0) if llm else 0.0
        if not llm or not llm.get("draft_text"):
            reply_lang, draft_text, cited, _ = local_draft(ctx, self.rules, self.kb)
            cited = self.rules.validate(cited)
            # Local drafts come from verified templates, so the confidence is the classifier's probability.
            llm_conf, llm_human, llm_reasons = 1.0, False, []
        else:
            reply_lang = ctx["lang_reply"]
            llm_conf = float(llm.get("confidence") or 0.5)
            llm_human = bool(llm.get("needs_human"))
            llm_reasons = [r for r in (llm.get("reasons") or []) if r][:3]

        # ---- needs_human policy (rules first, LLM can only add caution) ----
        needs_human = False
        if category in NEEDS_HUMAN_ALWAYS:
            needs_human = True
            reasons.append({"report.account": "аккаунт или пропажа имущества — всегда человек",
                            "report.payment": "платёж — всегда человек",
                            "complaint.admin": "жалоба на администратора или апелляция — всегда человек"}[category])
        if ALWAYS_HUMAN_TOPICS.search(ctx["text"]):
            needs_human = True
            reasons.append("тема разбана, возврата, апелляции, платежа или взлома")
        if has_threat(ctx["text"]):
            needs_human = True
            reasons.append("угрозы вне игры")
        if category.startswith("complaint.") and category != "complaint.invalid" and ctx["has_evidence"]:
            needs_human = True
            reasons.append("решение о наказании подтверждает администратор")
        if category in ("report.help", "report.complaint"):
            needs_human = True
            reasons.append("нужно действие администратора в игре")
        if has_promise(draft_text):
            needs_human = True
            reasons.append("черновик содержит решение, которое подтверждает администратор")
        confidence = round(min(ctx["cat_conf"], llm_conf), 2)
        if confidence < 0.45:
            needs_human = True
            reasons.append("низкая уверенность")
        if not needs_human and not self.nothing_to_do(ctx):
            needs_human = True
            reasons.append("без человека закрываются только спам и приветствия")
        if llm_human and not needs_human and category not in ("report.offtopic",):
            needs_human = True
        reasons += [r for r in llm_reasons if r not in reasons]
        if not needs_human and not reasons:
            reasons.append("типовой ответ по правилам и базе знаний")
        if ctx.get("verdict_hint"):
            v, p = ctx["verdict_hint"]
            reasons.insert(0, f"подсказка: похожие жалобы обычно закрывались как «{VERDICT_RU.get(v, v)}» ({p:.0%})")
        if ctx["is_dup"]:
            reasons.append("похоже на дубль: тот же игрок на того же нарушителя в течение часа")
        if ctx["is_spam"]:
            reasons.append("спам: пустое обращение, «ау» или флуд")

        nick = ctx["nick"] or (llm or {}).get("nickname") or None
        server = ctx["server"]  # region from the request or an explicit mention; the LLM's guess is not used
        return {
            # The case uses four languages; others (es, pt in the data) are reported as "und".
            "lang": ctx["lang"] if ctx["lang"] in ("ru", "kk", "uz", "en") else "und",
            "category": category,
            "priority": ctx["priority"],
            "is_spam_or_duplicate": bool(ctx["is_dup"] or ctx["is_spam"]),
            "extracted": {"nickname": nick, "rule_refs": ctx["player_refs"], "has_evidence": ctx["has_evidence"],
                          "server": server},
            "draft_reply": {"lang": reply_lang, "text": draft_text, "cited": cited},
            "needs_human": needs_human,
            "confidence": confidence,
            "reasons": reasons[:5],
            "_meta": {"latency_ms": round(latency_ms, 1), "cost_usd": round(cost, 6), "llm": bool(llm)},
        }

    def _prompt_ctx(self, ctx):
        return {**ctx, "lang": ctx["lang_reply"], "norm": normalize,
                "rules_text": self.rules.context(ctx["cites"]), "reported_nick": ctx["nick"]}

    def triage(self, t: dict) -> dict:
        start = time.perf_counter()
        ctx = self.prepare(t)
        prompt = self.drafter.build_prompt({**t, "kind": ctx["kind"], "messages": t.get("messages") or []},
                                           self._prompt_ctx(ctx))
        llm = self.drafter.draft(prompt, ctx["cites"])
        return self.finish(t, ctx, llm, (time.perf_counter() - start) * 1000)
