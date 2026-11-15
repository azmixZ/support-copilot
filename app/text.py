"""Text cleanup and language detection for short, noisy player messages."""
import re
import unicodedata

# The game runs in cp1251, so on kz servers Kazakh letters arrive as look-alike characters.
CP1251_KK = str.maketrans({
    "ќ": "қ", "Ќ": "Қ",
    "ґ": "ғ", "Ґ": "Ғ",
    "ѕ": "ң", "Ѕ": "Ң",
    "є": "ә", "Є": "Ә",
    "ј": "ө", "Ј": "Ө",
    "ў": "ұ", "Ў": "Ұ",
    "ї": "ү", "Ї": "Ү",
})
# Characters that only appear in cp1251-encoded Kazakh (ў is excluded: it is also Uzbek Cyrillic).
CP1251_MARKERS = set("ќЌґҐѕЅєЄјЈїЇ")
# і ї є ґ are also Ukrainian letters, so alone they don't prove Kazakh; these do.
STRONG_KK_CP = set("ќЌѕЅјЈ")
KK_UNIQUE = set("әғқңөұүһӘҒҚҢӨҰҮҺ")

KK_LETTERS = set("әғқңөұүһіӘҒҚҢӨҰҮҺІ")
UZ_CYR_LETTERS = set("ўҳЎҲ")
CYRILLIC = re.compile(r"[а-яёА-ЯЁ]")
LATIN = re.compile(r"[a-zA-Z]")

KK_WORDS = {"мен", "сен", "маған", "бар", "жоқ", "емес", "қалай", "неге", "көмектесіңіз", "ойыншы", "және", "үшін",
            "болды", "керек", "өтінемін", "рахмет", "сәлем", "мені", "менің", "осы", "деп", "бір"}
UZ_WORDS = {"yordam", "iltimos", "admin", "menga", "meni", "mening", "qanday", "nima", "kerak", "rahmat", "salom",
            "bering", "qiling", "bor", "yo'q", "yoq", "uchun", "bilan", "edi", "qildi", "bervorila", "ber", "o'yinchi",
            "oyinchi", "haqorat", "sokyapti", "shikoyat", "yordamga", "kuting", "sababsiz", "qamoqqa", "pul", "mashina",
            "ishlamayapti", "qotib", "qoldim", "kun", "talab", "qilaman", "sorab", "berish", "kere", "ozi", "uje", "ham", "lekin", "endi", "hamma", "siz", "biz", "u", "bu", "va",
            # Uzbek Latin words like "berila" (give), "nega" (why): on the case data they changed 5 tickets, all of them Uzbek
            "berila", "berilsin", "beringlar", "berilsa", "mindirish", "tushirish", "bilmedi", "blmedi", "bilmadi",
            "bilmaydi", "moshin", "moshinani", "mashinani", "qilinsin", "qilish", "qilib", "olib", "tashlab", "nega", "chiqib"}
# Russian typed in Latin letters ("латиница" in the data is mostly this): reply in Russian.
RU_TRANSLIT = {"osk", "oskorblenie", "rodni", "rodnyh", "rodnih", "pomogite", "pomogi", "pomoshch", "chat", "chate",
               "ne", "na", "menya", "menia", "mne", "ya", "on", "ona", "eto", "za", "po", "kak", "chto", "che", "chto-to",
               "ubil", "ubili", "ubivaet", "soglasen", "soflasem", "dal", "dali", "mut", "bez", "prichiny", "prichin",
               "pochemu", "igrok", "igroka", "nakazanie", "zhaloba", "pozhaluysta", "pojaluista", "pzh", "privet",
               "spasibo", "zastryal", "pereverni", "perevernite", "vas", "vy", "tut", "u", "s", "v", "i", "a", "nas",
               "eshe", "esche", "uzhe", "uje", "net", "da", "kop", "nrp", "mashinu", "tachku", "ego", "ee", "moyu", "moy"}
# Uzbek written in Cyrillic without ў/ҳ (e.g. "мен учун жазо керак эмас").
UZ_CYR_WORDS = {"менга", "мени", "учун", "билан", "эди", "йук", "йўқ", "ёрдам", "илтимос", "керак", "эмас", "емас",
                "емасман", "шахс", "учинчи", "мана", "кабул", "қабул", "коида", "қоида", "бузилмаган", "хакорат",
                "ҳақорат", "сокяпти", "сўкди", "нима", "қандай", "кандай", "лекин", "энди", "хамма", "ҳамма", "биз",
                "сиз", "ва", "бу", "квоти", "кутинг", "бериш", "берди", "қилди", "килди", "уйинчи", "ўйинчи", "жазо"}
RU_CORE = {"не", "на", "в", "и", "что", "меня", "мне", "он", "я", "это", "за", "по", "как", "у", "с", "мой", "мою",
           "его", "был", "без", "причины", "игрок", "помогите", "пожалуйста", "админ", "он", "она", "вы", "нас"}
# Kazakh in Latin script (2021 alphabet or informal).
KK_LAT_WORDS = {"salem", "salemetsiz", "rahmet", "qalai", "qalaisyz", "komek", "kómek", "komektesiniz",
                "kómektesińiz", "oiynshy", "oyinshy", "maǵan", "magan", "menin", "meniń", "joq", "kerek", "emes",
                "nege", "jane", "jáne", "ushin", "úshin", "shaǵym", "shagym", "boldy", "otinemin", "ótinemin"}
KK_LAT_LETTERS = set("áǵıńóúýÁǴŃÓÚÝ")
UZ_SUFFIXES = ("ni", "ga", "dan", "man", "yapti", "yapdi", "yabdi", "ingiz", "lar", "lari", "dim", "moqda", "sizmi")

EN_WORDS = {"help", "player", "the", "is", "me", "my", "stuck", "hacked", "account", "why",
            "what", "how", "killed", "insult", "can", "you", "not", "and", "he", "she", "money", "please"}
ES_PT_WORDS = {"ayuda", "por", "favor", "jugador", "cuenta", "ajuda", "jogador", "conta", "não", "nao", "obrigado",
               "gracias", "que", "el", "la", "de", "me", "mi", "meu", "minha"}
PT_ONLY = {"ajuda", "jogador", "conta", "não", "nao", "obrigado", "meu", "minha", "você", "voce"}

WORD = re.compile(r"[\w'‘’`]+", re.UNICODE)


def looks_kazakh(text: str, region: str | None = None) -> bool:
    """Kazakh vs. Ukrainian/surzhyk: both use і/ї/є/ґ, only Kazakh has ќ ѕ ј or real қ ә ө …"""
    if not text:
        return False
    if region == "kz" or any(ch in STRONG_KK_CP or ch in KK_UNIQUE for ch in text):
        return True
    if any(ch in CP1251_MARKERS or ch in "іІ" for ch in text):
        words = set(re.findall(r"\w+", text.lower().translate(CP1251_KK)))
        return len(words & KK_WORDS) >= 1
    return False


def fix_cp1251_kazakh(text: str, region: str | None = None) -> str:
    """Restore Kazakh letters typed through the cp1251 client."""
    if text and looks_kazakh(text, region):
        return text.translate(CP1251_KK)
    return text


def normalize(text: str, region: str | None = None) -> str:
    text = fix_cp1251_kazakh(unicodedata.normalize("NFC", text or ""), region)
    text = text.replace("‘", "'").replace("’", "'").replace("`", "'").replace("ʻ", "'").replace("ʼ", "'")
    return re.sub(r"\s+", " ", text).strip()


def detect_lang(text: str, region: str | None = None) -> str:
    """Return ru | kk | uz | en | es | pt | und. Tuned for very short, mixed-language messages.
    The server region is only a tie-breaker: a player on an Uzbek server may write in Russian."""
    raw = text or ""
    t = normalize(raw, region)
    cleaned = re.sub(r"<\w+>", " ", t)  # anonymisation tokens like <url>
    words = [w.lower().strip("'") for w in WORD.findall(cleaned)]
    letters = [c for c in cleaned if c.isalpha()]
    if len(letters) < 2:
        return "und"

    if looks_kazakh(raw, region):
        # Uzbek Cyrillic shares қ/ғ but has ў/ҳ and no ә/ө/ү/ң/і.
        kk_only = sum(c in "әөүңіұһӘӨҮҢІҰҺ" for c in cleaned)
        if kk_only or region == "kz":
            return "kk"
    if any(c in UZ_CYR_LETTERS for c in cleaned) and region != "kz":
        return "uz"

    cyr = len(CYRILLIC.findall(cleaned))
    lat = len(LATIN.findall(cleaned)) + sum(c in KK_LAT_LETTERS for c in cleaned)
    wset = set(words)
    if cyr >= lat:
        kk = len(wset & KK_WORDS)
        uz = len(wset & UZ_CYR_WORDS) + (0.5 if region == "uz" else 0)
        ru = len(wset & RU_CORE) + (0.5 if region in (None, "cis") else 0)
        if kk >= 2 or (kk and region == "kz"):
            return "kk"
        if uz >= 1.5 and uz > ru:
            return "uz"
        return "ru"

    kk = len(wset & KK_LAT_WORDS) + 2 * any(c in KK_LAT_LETTERS for c in cleaned) + (0.5 if region == "kz" else 0)
    uz_suffix = sum(len(w) >= 5 and w not in RU_TRANSLIT and w.endswith(UZ_SUFFIXES) for w in words)
    uz = (len(wset & UZ_WORDS) + sum(("o'" in w or "g'" in w) for w in words) + 0.5 * uz_suffix
          + (0.5 if region == "uz" else 0))
    ru = len(wset & RU_TRANSLIT) + (0.5 if region in (None, "cis") else 0)
    en = len(wset & EN_WORDS)
    espt = len(wset & ES_PT_WORDS)
    scores = [("uz", uz), ("kk", kk), ("ru", ru), ("en", en), ("pt" if wset & PT_ONLY else "es", espt)]
    best = max(scores, key=lambda x: x[1])
    if best[1] >= 1:
        return best[0]
    return {"uz": "uz", "kz": "kk"}.get(region or "", "ru")


def script_of(text: str) -> str:
    """'latin' or 'cyrillic': the draft is written in the same script the player used."""
    t = re.sub(r"<\w+>", " ", text or "")
    lat = len(LATIN.findall(t)) + sum(c in KK_LAT_LETTERS for c in t)
    return "latin" if lat > len(CYRILLIC.findall(t)) + sum(c in KK_LETTERS or c in UZ_CYR_LETTERS for c in t) else "cyrillic"


def player_text(messages: list[dict]) -> str:
    return "\n".join(str(m.get("text") or "") for m in messages if (m.get("role") or "player") == "player")
