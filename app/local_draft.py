"""Fully local draft generation: situation-aware templates + answers from the knowledge base.

No external API: the organisers require that player texts never leave the server in production.
Drafts never name a punishment, never promise an unban or refund, and cite only verified rules.
"""
import re

from .draft import PROMISE, PUNISHMENT

# ---------------------------------------------------------------- texts
T = {
    "ru": {
        "question_pass": "Ваш вопрос передан администратору. Ответы на частые вопросы есть также в официальном Discord-сервере игры.",
        "how_to_complain": "Жалобу подают в лаунчере игры, в разделе жалоб: опишите, что произошло, укажите ник нарушителя "
                           "и приложите скриншоты или видео (пункт ЖБ 1.2.1).",
        "hello": "Приветствую!",
        "review": "Жалоба{on} принята и передана администратору. По описанию это похоже на нарушение {label}. "
                  "Решение примет администратор после проверки доказательств.",
        "on": " на игрока {nick}", "thanks": "Спасибо за обращение.",
        "review_nocite": "Жалоба{on} принята и передана администратору. Решение примет администратор после проверки доказательств.",
        "video": "Если есть видеозапись момента, приложите её: такие нарушения подтверждаются по видео.",
        "need_evidence": "Для рассмотрения жалобы нужны доказательства: скриншоты или видео без обрезки (пункт ЖБ 1.2.2). "
                         "Подайте, пожалуйста, жалобу повторно с доказательствами.",
        "invalid": "Жалоба оформлена с нарушением правил подачи: нет описания сути нарушения (пункт ЖБ 1.2.3). "
                   "Опишите, что произошло, укажите ник нарушителя и приложите доказательства.",
        "invalid_toxic": "Жалоба содержит оскорбления или ненормативную лексику и не может быть рассмотрена (пункт ЖБ 1.2.7). "
                         "Подайте, пожалуйста, жалобу повторно без оскорблений.",
        "invalid_question": "Раздел жалоб предназначен только для сообщений о нарушениях (пункт ЖБ 1.1.2). "
                            "С вопросами обращайтесь в репорт в игре или в Discord.",
        "admin": "Жалоба на действия администратора подана через раздел жалоб, как и предусмотрено пунктом 3.4. Требуется проверка старшей администрации: решение принимается после изучения доказательств.",
        "help": "Если транспорт перевернулся или застрял, воспользуйтесь кнопкой эвакуатора на спидометре: через репорт транспорт не чинят и не выдают (пункт 3.1). Если застряли вы сами, требуется помощь администратора: репорт передан ему.",
        "report_player": "Обращение передано администратору: требуется проверка {who}. Если есть доказательства нарушения, подайте жалобу в лаунчере игры.",
        "who_id": "игрока с ID {id}", "who": "игрока",
        "question": "Уточните, пожалуйста, ваш вопрос: так администратору будет проще на него ответить.",
        "need_help": "Репорт передан администратору. Если можете, коротко опишите, что случилось: репорт — для вопросов и проблем в игре (пункт 3.1).",
        "offtopic": "Репорт предназначен для вопросов и жалоб по игре (пункт 3.1). Опишите, пожалуйста, вашу проблему.",
        "abuse": "Пожалуйста, соблюдайте правила общения с администрацией (пункт 3.2). Если нужна помощь, опишите проблему.",
        "greeting": "Приятной игры на Grand Mobile! Если понадобится помощь, опишите вопрос: репорт предназначен для вопросов и проблем в игре (пункт 3.1).",
        "account": "Вопросы доступа к аккаунту и пропажи имущества решает техническая поддержка, обращение передано специалисту. "
                   "Чтобы защитить аккаунт, смените пароль и никому не передавайте данные: за его сохранность отвечает сам "
                   "игрок (пункт 1.4).",
        "payment": "Вопрос по платежу передан специалисту поддержки: требуется проверка операции.",
        # how a citation is named in the text
        "op": "пункта {ref} ОП", "code": "пункта {cite}", "titled": "пункта {ref} раздела «{title}»",
        "term": "{ref} (раздел «{title}»)",
    },
    "kk": {
        "question_pass": "Сұрағыңыз әкімшіге жіберілді. Жиі қойылатын сұрақтардың жауаптары ойынның ресми Discord-серверінде де бар.",
        "how_to_complain": "Шағым ойын лаунчеріндегі шағымдар бөлімінде беріледі: не болғанын жазып, бұзушының никін "
                           "көрсетіп, скриншот немесе видео қосыңыз (ЖБ 1.2.1 тармағы).",
        "hello": "Сәлеметсіз бе!",
        "on": "{nick} ойыншысына қатысты ", "thanks": "Өтінішіңіз үшін рахмет.",
        "review": "{on}Шағым қабылданып, әкімшіге жіберілді. Сипаттамаға қарағанда бұл {label} бұзуға ұқсайды. "
                  "Шешімді әкімші дәлелдерді тексергеннен кейін қабылдайды.",
        "review_nocite": "{on}Шағым қабылданып, әкімшіге жіберілді. Шешімді әкімші дәлелдерді тексергеннен кейін қабылдайды.",
        "video": "Видеожазбаңыз болса, оны да тіркеңіз: мұндай бұзушылықтар видеомен расталады.",
        "need_evidence": "Шағымды қарау үшін бұзушылықтың дәлелі керек: қиылмаған скриншот немесе видео (ЖБ 1.2.2 тармағы). "
                         "Шағымды дәлелдермен қайта жіберіңіз.",
        "invalid": "Шағым ережеге сай рәсімделмеген: бұзушылықтың мәні сипатталмаған (ЖБ 1.2.3 тармағы). "
                   "Не болғанын жазып, бұзушының никін көрсетіп, дәлел қосыңыз.",
        "invalid_toxic": "Шағымда балағат немесе қорлайтын сөздер бар, сондықтан ол қаралмайды (ЖБ 1.2.7 тармағы). "
                         "Шағымды балағатсыз қайта жіберіңіз.",
        "invalid_question": "Шағымдар бөлімі тек ережені бұзу туралы хабарламаларға арналған (ЖБ 1.1.2 тармағы). "
                            "Сұрақтарыңызды ойындағы репортқа немесе Discord-қа жазыңыз.",
        "admin": "Әкімшінің әрекетіне шағым 3.4 тармағына сәйкес шағымдар бөлімі арқылы берілді. Аға әкімшіліктің тексеруі қажет: шешім дәлелдер қаралғаннан кейін қабылданады.",
        "help": "Көлік аударылып немесе тұрып қалса, спидометрдегі эвакуатор батырмасын басыңыз: репорт арқылы көлік жөнделмейді және берілмейді (3.1 тармағы). Өзіңіз тұрып қалсаңыз, әкімшінің көмегі қажет: репорт әкімшіге жіберілді.",
        "report_player": "Өтініш әкімшіге жіберілді: {who} тексеру қажет. Дәлеліңіз болса, ойын лаунчері арқылы шағым жазыңыз.",
        "who_id": "ID {id} ойыншыны", "who": "ойыншыны",
        "question": "Сұрағыңызды нақтылап жазыңыз: әкімшіге жауап беру оңайырақ болады.",
        "need_help": "Репортыңыз әкімшіге жіберілді. Мүмкін болса, не болғанын қысқаша жазыңыз: репорт ойындағы сұрақтар мен мәселелерге арналған (3.1 тармағы).",
        "offtopic": "Репорт ойын бойынша сұрақтар мен шағымдарға арналған (3.1 тармағы). Мәселеңізді жазыңыз.",
        "abuse": "Әкімшілікпен сөйлескенде ережені сақтаңыз (3.2 тармағы). Көмек керек болса, мәселеңізді жазыңыз.",
        "greeting": "Grand Mobile-да жақсы ойын тілейміз! Көмек керек болса, сұрағыңызды жазыңыз: репорт ойындағы сұрақтар мен мәселелерге арналған (3.1 тармағы).",
        "account": "Аккаунтқа кіру және мүліктің жоғалуы мәселелерін техникалық қолдау шешеді, өтініш маманға жіберілді. "
                   "Аккаунтты қорғау үшін құпиясөзді ауыстырып, деректерді ешкімге бермеңіз: оның қауіпсіздігіне ойыншының "
                   "өзі жауап береді (1.4 тармағы).",
        "payment": "Төлем туралы сұрағыңыз қолдау маманына жіберілді: операцияны тексеру қажет.",
        "op": "{ref} тармағын (ОП)", "code": "{cite} тармағын", "titled": "«{title}» бөлімінің {ref} тармағын",
        "term": "{ref} ережесін («{title}» бөлімі)",
    },
    "uz": {
        "question_pass": "Savolingiz administratorga yuborildi. Ko'p beriladigan savollarga javoblar o'yinning rasmiy Discord serverida ham bor.",
        "how_to_complain": "Shikoyat o'yin launcheridagi shikoyatlar bo'limida beriladi: nima bo'lganini yozing, "
                           "qoidabuzarning nikini ko'rsating va skrinshot yoki video qo'shing (ЖБ 1.2.1-band).",
        "hello": "Assalomu alaykum!",
        # Uzbek admin answers to complaints: "Hurmatli ..." (85%) and "Hurmat bilan, GrandMobile Ma'muriyati" (86%).
        "hello_complaint": "Hurmatli o'yinchi!", "sign_complaint": "\n\nHurmat bilan,\nGrandMobile Ma'muriyati",
        "on": "{nick} o'yinchisi ustidan ", "thanks": "Murojaatingiz uchun rahmat.",
        "review": "{on}Shikoyat qabul qilindi va administratorga yuborildi. Tavsifga ko'ra bu {label} buzilishiga o'xshaydi. "
                  "Qarorni administrator dalillarni tekshirgandan so'ng qabul qiladi.",
        "review_nocite": "{on}Shikoyat qabul qilindi va administratorga yuborildi. Qarorni administrator dalillarni tekshirgandan so'ng qabul qiladi.",
        "video": "Agar video yozuvingiz bo'lsa, uni ham ilova qiling: bunday qoidabuzarliklar video orqali tasdiqlanadi.",
        "need_evidence": "Shikoyatni ko'rib chiqish uchun qirqilmagan skrinshot yoki video dalil kerak (ЖБ 1.2.2-band). "
                         "Iltimos, shikoyatni dalillar bilan qayta yuboring.",
        "invalid": "Shikoyat qoidalarga muvofiq rasmiylashtirilmagan: qoidabuzarlik mohiyati yozilmagan (ЖБ 1.2.3-band). "
                   "Nima bo'lganini yozing, qoidabuzarning nikini ko'rsating va dalil qo'shing.",
        "invalid_toxic": "Shikoyatda haqoratli yoki nojo'ya so'zlar bor, shuning uchun u ko'rib chiqilmaydi (ЖБ 1.2.7-band). "
                         "Shikoyatni haqoratsiz qayta yuboring.",
        "invalid_question": "Shikoyatlar bo'limi faqat qoidabuzarliklar haqida xabar berish uchun (ЖБ 1.1.2-band). "
                            "Savollaringizni o'yindagi reportga yoki Discord'ga yozing.",
        "admin": "Administrator harakatlari ustidan shikoyat 3.4-bandga muvofiq shikoyatlar bo'limi orqali berildi. Katta administratsiya tekshiruvi talab qilinadi: qaror dalillar ko'rib chiqilgandan so'ng qabul qilinadi.",
        "help": "Agar mashina ag'darilgan yoki qotib qolgan bo'lsa, spidometrdagi evakuator tugmasini bosing: report orqali transport tuzatilmaydi va berilmaydi (3.1-band). O'zingiz qotib qolgan bo'lsangiz, administrator yordami talab qilinadi: report administratorga yuborildi.",
        "report_player": "Murojaat administratorga yuborildi: {who} tekshirish talab qilinadi. Dalilingiz bo'lsa, o'yin launcherida shikoyat yozing.",
        "who_id": "ID {id} o'yinchini", "who": "o'yinchini",
        "question": "Iltimos, savolingizni aniqlashtirsangiz: administratorga javob berish osonroq bo'ladi.",
        "need_help": "Reportingiz administratorga yuborildi. Iloji bo'lsa, nima bo'lganini qisqacha yozing: report o'yindagi savol va muammolar uchun (3.1-band).",
        "offtopic": "Report o'yin bo'yicha savol va shikoyatlar uchun (3.1-band). Muammoingizni yozing.",
        "abuse": "Administratsiya bilan muloqotda qoidalarga rioya qiling (3.2-band). Yordam kerak bo'lsa, muammoingizni yozing.",
        "greeting": "Grand Mobile'da yoqimli o'yin tilaymiz! Yordam kerak bo'lsa, savolingizni yozing: report o'yindagi savol va muammolar uchun (3.1-band).",
        "account": "Akkauntga kirish va mulk yo'qolishi masalalarini texnik yordam hal qiladi, murojaatingiz mutaxassisga "
                   "yuborildi. Akkauntni himoya qilish uchun parolni almashtiring va ma'lumotlarni hech kimga bermang: uning "
                   "xavfsizligi uchun o'yinchining o'zi javobgar (1.4-band).",
        "payment": "To'lov bo'yicha savolingiz yordam mutaxassisiga yuborildi: operatsiyani tekshirish talab qilinadi.",
        "op": "{ref}-band (OP)", "code": "{cite}-band", "titled": "«{title}» bo'limining {ref}-bandi",
        "term": "{ref} qoidasi («{title}» bo'limi)",
    },
    "en": {
        "question_pass": "Your question has been passed to an administrator. Answers to common questions are also on the game's official Discord server.",
        "how_to_complain": "Complaints are filed in the game launcher, in the complaints section: describe what happened, "
                           "give the violator's nickname and attach screenshots or video (rule ЖБ 1.2.1).",
        "hello": "Hello!",
        "on": " about {nick}", "thanks": "Thank you for reporting.",
        "review": "Your complaint{on} has been received and passed to an administrator. From the description this looks like "
                  "a violation of {label}. The administrator will decide after checking the evidence.",
        "review_nocite": "Your complaint{on} has been received and passed to an administrator, who will decide after checking the evidence.",
        "video": "If you have a video of the moment, please attach it: such violations are confirmed by video.",
        "need_evidence": "To review the complaint we need evidence: uncut screenshots or video (rule ЖБ 1.2.2). "
                         "Please submit the complaint again with evidence.",
        "invalid": "The complaint does not describe what happened (rule ЖБ 1.2.3). Please describe the situation, "
                   "give the violator's nickname and attach evidence.",
        "invalid_toxic": "The complaint contains insults or profanity and cannot be reviewed (rule ЖБ 1.2.7). "
                         "Please submit it again without insults.",
        "invalid_question": "The complaints section is only for reporting rule violations (rule ЖБ 1.1.2). "
                            "Please ask questions through the in-game report or Discord.",
        "admin": "Your complaint about an administrator was filed through the complaints section, as rule 3.4 provides. It requires review by senior administration: the decision is made after the evidence is examined.",
        "help": "If your vehicle flipped or got stuck, use the tow truck button on the speedometer: vehicles are not repaired or handed out through reports (rule 3.1). If you are stuck yourself, an administrator's help is required: the report has been passed on.",
        "report_player": "Passed to an administrator: {who} requires a check. If you have evidence, please file a complaint in the game launcher.",
        "who_id": "the player with ID {id}", "who": "the player",
        "question": "Please clarify your question so that an administrator can answer it more easily.",
        "need_help": "Your report has been passed to an administrator. If you can, briefly describe what happened: reports are for game questions and problems (rule 3.1).",
        "offtopic": "Reports are for game questions and complaints (rule 3.1). Please describe your problem.",
        "abuse": "Please follow the rules of communication with the administration (rule 3.2). If you need help, describe the problem.",
        "greeting": "Have a nice game on Grand Mobile! If you need help, describe your question: reports are for game questions and problems (rule 3.1).",
        "account": "Account access and missing property are handled by technical support; your request has been passed to "
                   "a specialist. To protect your account, change the password and never share your details: players are "
                   "responsible for their account's safety (rule 1.4).",
        "payment": "Your payment question has been passed to a support specialist: the transaction requires a check.",
        "op": "rule {ref} (general rules)", "code": "rule {cite}", "titled": "rule {ref} of «{title}»",
        "term": "{ref} (section «{title}»)",
    },
}

# Situations -> which rule the template itself cites.
TEMPLATE_CITES = {"need_evidence": ["ЖБ 1.2.2"], "invalid": ["ЖБ 1.2.3"], "invalid_toxic": ["ЖБ 1.2.7"],
                  "invalid_question": ["ЖБ 1.1.2"], "admin": ["3.4"], "offtopic": ["3.1"], "greeting": ["3.1"],
                  "help": ["3.1"], "need_help": ["3.1"], "account": ["1.4"], "abuse": ["3.2"], "how_to_complain": ["ЖБ 1.2.1"]}
VIDEO_CATEGORIES = {"complaint.dm", "complaint.cheat"}

HELP_ONLY = re.compile(r"^\W*(помоги\w*|памаги\w*|помаги\w*|помги\w*|помогі\w*|памагі\w*|поможіть|хелп|help|sos|сос|"
                       r"спасите|помощь|yordam\w*|komek|көмек\w*|админ\w*|admin\w*)(\s+(пж|пожалуйста|плиз|pls|please|"
                       r"срочно|админ\w*|admin|e|bervorila|bering|керек))*\W*$", re.IGNORECASE)
GREETING = re.compile(r"^\W*(привет\w*|здравствуй\w*|добр\w+ (утро|день|вечер)|доброй ночи|спасибо\w*|благодар\w*|"
                      r"хорошего \w+|приятной игры|спокойной ночи|салам\w*|salom|assalomu alaykum|rahmat|сәлем\w*|рахмет|"
                      r"hi|hello|thanks?|thank you)\W*$", re.IGNORECASE)
ABUSE = re.compile(r"\b(сука|суки|сучк\w*|бля\w*|хуй\w*|нахуй|пизд\w*|ебал\w*|пидор\w*|пидар\w*|мраз\w*|чмо|даун\w*)\b",
                   re.IGNORECASE)
HOW_TO_COMPLAIN = re.compile(r"(как|где|куда)\s+(\S+\s+){0,3}?(залить|подать|написать|кинуть|отправить|оформить)?\s*"
                             r"(\S+\s+){0,2}?(жб|жалоб\w*)", re.IGNORECASE)
TRANSPORT = re.compile(r"машин|тачк|авто\b|транспорт|перевер|эвакуат|утон|утан|в воду|мото|байк|велик|фур|грузовик|"
                       r"mashina|moshina|машина|көлік|аударыл|ag'daril|agdaril", re.IGNORECASE)
QUESTIONISH = re.compile(r"\?|^(как|где|что|почему|зачем|когда|сколько|можно|кто)\b", re.IGNORECASE)
PLAYER_ID = re.compile(r"(?:\b(?:ид|id|айди)\s*(\d{1,4})\b|\b(\d{1,4})\s*(?:ид|id|айди)?\s*(?:дб|db|дм|dm|тк|tk|ск|sk|"
                       r"нрп|nrp|софт|чит|оск|пг|мг|афк|спек))", re.IGNORECASE)

# Answers in the knowledge base that carry no information or describe the admin's own actions.
GENERIC_ANSWER = re.compile(r"слежу|ожидайте|спешу|лечу|в пути|уточните|помогу|помогли|приятной игры|работаю|"
                            r"подключил|иду|телепорт|сейчас|постараюсь|наказан|жалоб|нет\w* информац|не владеем|"
                            r"самостоятельно|никак|не знаю|не оказываем|не предоставляем|не занимаемся", re.IGNORECASE)
# Only answers that actually tell where / how ("навигатор > раздел > пункт", "в лаунчере").
INFORMATIVE = re.compile(r"навигатор|\S\s*>\s*\S|в лаунчере|в планшете|в меню|кнопк\w* \w+", re.IGNORECASE)
NICKNAME = re.compile(r"\b[A-Z][a-z]+_[A-Z][a-z]+\d*(?:\[\d+\])?")


def cite_label(cite: str, lang: str, rules) -> str:
    t = T[lang]
    p = rules.by_cite.get(cite)
    if p is None:
        return cite
    if p.section_id == 30:
        return t["op"].format(ref=p.ref)
    if p.section_id in (31, 32):
        return t["code"].format(cite=cite)
    if not p.ref[0].isdigit():
        return t["term"].format(ref=p.ref, title=p.section)
    return t["titled"].format(ref=p.ref, title=p.section)


def kb_answer(kb, text: str) -> str | None:
    """A factual answer to a game-mechanics question, taken from a near-identical past question."""
    if kb is None or len(text) < 8:
        return None
    for e in kb.similar("report", text, k=8, lang="ru"):
        if e["score"] < 0.5:
            break
        a = re.sub(r"^\W*(приветствую|здравствуйте|доброго времени суток)[\s,!.]*", "", e["a"], flags=re.IGNORECASE).strip()
        if (len(a) < 8 or not INFORMATIVE.search(a) or GENERIC_ANSWER.search(a) or PUNISHMENT.search(a)
                or PROMISE.search(a) or NICKNAME.search(a)):
            continue
        if "<" in a:  # anonymisation tokens like <url>
            continue
        return a[0].upper() + a[1:]
    return None


def situation(ctx: dict) -> str:
    cat, text = ctx["category"], ctx["text"]
    if cat == "report.account":
        return "account"
    if cat == "report.payment":
        return "payment"
    if cat == "complaint.admin":
        return "admin"
    if cat.startswith("complaint."):
        if cat == "complaint.invalid":
            if ctx.get("main_cite") and not ABUSE.search(text):
                return "review"  # short but names a violation and has evidence
            if ABUSE.search(text):
                return "invalid_toxic"
            if QUESTIONISH.search(text):
                return "invalid_question"
            return "invalid"
        return "review" if ctx["has_evidence"] else "need_evidence"
    if cat.startswith("report.") and HOW_TO_COMPLAIN.search(text):
        return "how_to_complain"
    if cat == "report.help":
        # The tow-truck tip only makes sense when the report is about a vehicle.
        return "help" if TRANSPORT.search(text) else "need_help"
    if cat == "report.complaint":
        return "report_player"
    if cat == "report.question":
        return "question"
    # report.offtopic
    if GREETING.match(text):
        return "greeting"
    if ABUSE.search(text):
        return "abuse"
    if HELP_ONLY.match(text):
        return "need_help"
    return "offtopic"


def local_draft(ctx: dict, rules, kb=None) -> tuple[str, str, list[str], str]:
    """(reply_lang, text, cited, situation) built without any external service."""
    lang = ctx["lang_reply"] if ctx["lang_reply"] in T else ("en" if ctx["lang_reply"] in ("es", "pt") else "ru")
    t = T[lang]
    sit = situation(ctx)
    cited: list[str] = list(TEMPLATE_CITES.get(sit, []))
    if sit == "review":
        nick = ctx.get("nick")
        on = t["on"].format(nick=nick) if nick else ""
        main = ctx.get("main_cite")
        if main:
            body = t["review"].format(label=cite_label(main, lang, rules), on=on)
            cited = [main]
        else:
            body = t["review_nocite"].format(on=on)
        if on and lang in ("kk", "uz") and body.startswith(on):
            rest = body[len(on):]
            body = on + rest[:1].lower() + rest[1:]  # "Arman ойыншысына шағым ...", "Arman o'yinchisi ustidan shikoyat ..."
        if ctx["category"] in VIDEO_CATEGORIES:
            body += " " + t["video"]
        body += " " + t["thanks"]
    elif sit == "report_player":
        m = PLAYER_ID.search(ctx["text"])
        pid = next((g for g in m.groups() if g), None) if m else None
        body = t["report_player"].format(who=t["who_id"].format(id=pid) if pid else t["who"])
    elif sit == "how_to_complain":
        body = t["how_to_complain"]
    elif sit == "question":
        answer = kb_answer(kb, ctx["text"]) if lang == "ru" else None
        if answer:
            body, sit = answer, "question_kb"
        elif len(re.findall(r"\w+", ctx["text"])) >= 4:
            body, sit = t["question_pass"], "question_pass"  # a clear question: no need to ask to clarify it
        else:
            body = t["question"]
    else:
        body = t[sit]
    if ctx.get("kind") == "complaint" and "hello_complaint" in t:
        text = f"{t['hello_complaint']} {body}{t['sign_complaint']}"
    else:
        text = f"{t['hello']} {body}"
    return lang, text, [c for c in cited if c in rules.by_cite], sit
