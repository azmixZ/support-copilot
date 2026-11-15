"""Category / priority rules.

Used in two places:
  * to build weak training labels from the admin outcome (verdict, violation) in train data;
  * as a deterministic layer on top of the ML classifier (priority, hard overrides).
"""
import os
import re

CATEGORIES = [
    "complaint.insult", "complaint.cheat", "complaint.dm", "complaint.scam", "complaint.rmt_ads",
    "complaint.admin", "complaint.other", "complaint.invalid",
    "report.question", "report.help", "report.complaint", "report.account", "report.payment", "report.offtopic",
]
NEEDS_HUMAN_ALWAYS = {"report.account", "report.payment", "complaint.admin"}


def _has(text: str, pattern: str) -> bool:
    return re.search(pattern, text, re.IGNORECASE) is not None


# --- keyword groups (ru + translit + kk + uz) ---
KW = {
    "account": r"взлом|взломал|украл[иа]? акк|пропал[аио]? (акк|машин|тачк|дом|бизнес|деньг|имуществ|вещ)|"
               r"не могу (зайти|войти)|парол|почт[ауы] привяз|акк(аунт)?\s*(угнал|взлом|увел)|"
               r"hacked|vzlom|akkaunt\w* (buz|o'g'irla|ogirla)|аккаунт\w* (бұз|ұрла)|бұзып алды|akkauntim|buzib|"
               r"көлігім жоғал|машинам жоғал|mashinam yo'qol|mashinam yoqol",
    "payment": r"донат|оплатил|оплата|пополнил|пополнение|задонат|не (пришл|начисл|зачисл)|списал[иао]|"
               r"купил за (реал|руб|тенге|сум)|\bчек\b|транзакц|платеж|платёж|kaspi|каспи|\bclick\b|\bpayme\b|"
               r"\bгк\b.*(не пришл|оплат)|donat|to'lov|tolov|төле",
    "threat": r"(найду|вычислю|приеду|приду) (тебя|к тебе)|твой адрес|деанон|сват|дос(ну|ить)|ddos|"
              r"убью тебя в реал|знаю где ты жив",
    "massbug": r"у всех|всех (кикнул|выкинул)|сервер (упал|лежит|не работает)|массов",
    "help": r"застрял|текстур|переверн|перевер|вытащ|утон|утан|тонет|в воду|упал[аи]? в воду|наручник|выпуст|"
            r"не могу (выйти|выбраться|двигат|сдвинуть)|бага?\b|баг |забаг|застр|провалил|"
            r"телепорт|тпш|\bтп\b|почин|эваку|тачк|машин[ау] (перевер|застр)|"
            r"qotib|qoldim|stuck|тұрып қалдым|кептеліп|аударылып|ag'darilib|agdarilib",
    "complaint": r"\bдм\b|\bdm\b|\bдб\b|\bdb\b|\bск\b|\bsk\b|\bтк\b|оскорб|оск\b|матер|убил|убива|"
                 r"чит|софт|аим|aim|wh\b|нарушает|мешает|нрп|nrp|пг\b|haqorat|sokyapti|ұрды|балағат",
    "question": r"^(как|где|что|почему|зачем|когда|сколько|можно ли|можно|подскажите|скажите|кто)\b|\?|"
                r"\b(qanday|qayerda|nima|nega|қалай|қайда|неге|how|where|what|why)\b",
}

INSULT = r"(?<!\d)2[.,m][146](?!\d)|оск|оскорб|мат\b|расизм|нацизм|родн|мать|мам[уа]|семь|токси|haqorat|sokyapti|балағат|insult"
CHEAT = r"(?<!\d)5[.,]2(?!\d)|чит|софт|аим|aim|wh\b|вх\b|фаст|макрос|кликер|бот\b|скрипт|летает|сквозь стен|через стен|спидхак|cheat"
DM = r"\bдм\b|\bdm\b|\bдб\b|\bdb\b|\bск\b|\bsk\b|\bтк\b|\btk\b|gz|гз|зз|зел[её]н|5\.[34]|убил|убива|без причин|" \
     r"нрп|nrp|non ?rp|пг\b|pg\b|мг\b|mg\b|рп|deathmatch|урон|стрел|таран"
SCAM = r"(?<!\d)4[.,][12](?!\d)|обман|кинул|развел|развёл|мошен|скам|scam|не отдал|не вернул|вымога|долг"
RMT = r"(?<!\d)2[.,]3(?!\d)|(?<!\d)4[.,]4(?!\d)|реклам|за реал|реальные деньги|продает акк|продаёт акк|продажа акк|телеграм|\bтг\b|t\.me|сайт|другой сервер|rmt"
ADMIN = r"админ|администратор|модер|выдал(и)? (мне )?(мут|бан|варн|джайл|деморган|наказ)|не согласен|снимите (мут|бан|наказ)|" \
        r"наказание (выдано|неверно|просто так)|наказал[иа]? (просто так|без причин)|апелляц|разбан|admin|адм\b"


def complaint_category(text: str, rule_refs_raw: str = "", violation: str = "", verdict: str | None = None) -> str:
    """Rule-based complaint category; when admin outcome is known (train), it takes priority."""
    blob = " ".join(x for x in (violation, rule_refs_raw, text) if x)
    if verdict in ("appeal_approved", "appeal_rejected"):
        return "complaint.admin"
    if _has(text, r"не соглас|наказание (выдано|выдали)|снимите (мут|бан|варн|наказ|деморган|джайл)|выдал[аи]? мне|"
                  r"админ\w* (выдал|наказал|посадил|забанил)|апелляц|разбан"):
        return "complaint.admin"
    # violation code written by the admin is the most reliable signal
    for src in ([violation] if violation else []) + [rule_refs_raw or "", text]:
        if not src:
            continue
        if _has(src, CHEAT):
            return "complaint.cheat"
        if _has(src, RMT):
            return "complaint.rmt_ads"
        if _has(src, SCAM):
            return "complaint.scam"
        if _has(src, INSULT):
            return "complaint.insult"
        if _has(src, DM):
            return "complaint.dm"
    if _has(text, ADMIN) or _has(rule_refs_raw or "", r"не согласен|админ|наказ"):
        return "complaint.admin"
    letters = re.sub(r"<\w+>|[^\w]", "", text or "")
    if len(letters) < 4 or _has(text, r"^(жб|вот|вот он|он|смотрите|.)$"):
        return "complaint.invalid"
    if _has(blob, KW["question"]) and not _has(blob, r"\d"):
        return "complaint.invalid"
    return "complaint.other"


# "435 дб", "138 софт", "347ид задрал", "3.2 299": a report about another player by in-game ID.
# Admins answer these with "слежу за игроком", so they are report.complaint, not offtopic.
_VIOL = r"(дб|db|дм|dm|тк|tk|ск|sk|нрп|nrp|софт|soft|чит\w*|cheat|оск\w*|ос\d|пг|pg|мг|mg|афк|спек|spec)"
REPORT_ON_PLAYER = re.compile(
    rf"\b\d{{1,4}}\s*(ид|id|айди)?\s*{_VIOL}\b|\b{_VIOL}\s*\d{{1,4}}\b|\b\d{{1,4}}\s*(ид|id|айди)|"
    rf"(ид|id|айди)\s*\d{{1,4}}|\b\d\.\d\s+\d{{1,4}}\b", re.IGNORECASE)


# "переверните", "застрял", "көлігім аударылып қалды", "mashinam ag'darilib qoldi": the player needs help in game.
STUCK = re.compile(r"перевер|аударыл|ag'?daril|застр[ая]л|тұрып қал|qotib qol|кептел|stuck|провалил|текстур", re.IGNORECASE)


def is_report_on_player(text: str) -> bool:
    return REPORT_ON_PLAYER.search(text or "") is not None


def report_category(text: str) -> str:
    t = (text or "").strip()
    letters = re.sub(r"<\w+>|[^\w]", "", t)
    if _has(t, KW["account"]):
        return "report.account"
    if _has(t, KW["payment"]):
        return "report.payment"
    if is_report_on_player(t):
        return "report.complaint"
    if _has(t, KW["help"]):
        return "report.help"
    if _has(t, KW["complaint"]):
        return "report.complaint"
    if len(letters) >= 6 and _has(t, KW["question"]):
        return "report.question"
    return "report.offtopic"


def urgent_report_category(text: str) -> str | None:
    """Explicit hacking / payment words override the model: these tickets always go to a human."""
    if _has(text or "", KW["account"]):
        return "report.account"
    if _has(text or "", KW["payment"]):
        return "report.payment"
    return None


def has_threat(text: str) -> bool:
    return _has(text or "", KW["threat"])


# p0 follows the taxonomy: account access / hacking, payments, threats outside the game. A mass bug can't be
# recognised from one ticket (organisers), so it is not p0. ALLOW_P0=0 marks these p1 instead — for a test set
# without p0, where predicting an absent class would add a zero-F1 class to macro-F1.
ALLOW_P0 = os.getenv("ALLOW_P0", "1") == "1"

# Priority follows the type and severity of the violation, not the screenshots (organisers): evidence goes into
# extracted.has_evidence and the confidence. p1: cheats, scam, RMT ads, complaints about an admin;
# p2: insults, DM, RP violations, reports about a player; p3: mechanics questions, offtopic, empty complaints.
def priority_for(category: str, text: str, has_evidence: bool = False) -> str:
    # A mass bug can't be recognised from a single ticket (organisers), so it doesn't raise the priority.
    # An account ticket is p0 only when it is about access or hacking ("взломали", "не могу зайти");
    # "смена ника" or a missing insurance is an account matter, but not an emergency.
    if category == "report.payment" or has_threat(text) or (category == "report.account" and _has(text, KW["account"])):
        return "p0" if ALLOW_P0 else "p1"
    if category in ("report.account", "complaint.cheat", "complaint.scam", "complaint.rmt_ads", "complaint.admin"):
        return "p1"
    if category in ("complaint.insult", "complaint.dm", "complaint.other", "report.complaint"):
        return "p2"
    # complaint.invalid, report.help (stuck / flipped car is a mechanics issue), report.question, report.offtopic
    return "p3"


def has_evidence_of(text: str, attachments: int | None) -> bool:
    return (attachments or 0) > 0 or "<url>" in (text or "") or _has(text or "", r"https?://|youtu|imgur|yapx|disk\.yandex|drive\.google")


# "ДМГ" / "деморган" / "джайл" is the admin jail, not DM. "Дал дмг ни за что", "посадили в деморган" is a complaint
# about an admin's punishment (appeal), while "требую посадить его в деморган" asks to punish another player.
JAIL_MENTION = re.compile(r"(?<!\w)(дмг|dmg|деморган\w*|demorgan\w*|деморг\w*|джайл\w*|jail\w*)(?!\w)", re.IGNORECASE)
JAIL_APPEAL = re.compile(
    r"(дал\w*|выдал\w*|посадил\w*|посадили|кинул\w*|отправил\w*|закинул\w*|получил\w*)\s+(\S+\s+){0,2}?(в\s+)?"
    r"(дмг|dmg|деморган|demorgan|джайл|jail)|(?<!\w)(в|на)\s+(дмг|dmg|деморган\w*|demorgan|джайл\w*|jail)|"
    r"(дмг|dmg|деморган\w*|джайл\w*|jail)\s+(\S+\s+){0,2}?(ни\s+за\s+что|не\s+за\s+что|за\s+что|просто\s+так|"
    r"без\s+причин\w*|незачто)|(ни|не)\s+за\s+что|незачто|не\s*соглас", re.IGNORECASE)
JAIL_REQUEST = re.compile(r"(дайте|дай|выдайте|выдать|дать|накажите|посадить|посадите|требую|отправьте)\s+(\S+\s+){0,3}?"
                          r"(дмг|dmg|деморган|demorgan|джайл|jail)", re.IGNORECASE)


def is_jail_appeal(text: str) -> bool:
    t = text or ""
    return bool(JAIL_MENTION.search(t) and JAIL_APPEAL.search(t) and not JAIL_REQUEST.search(t))
