"""LLM draft generation (Claude Haiku 5.5) with a closed list of citable rules."""
import json
import os
import re

import anthropic

MODEL = os.getenv("DRAFT_MODEL", "claude-haiku-5-5")
TIMEOUT_S = float(os.getenv("DRAFT_TIMEOUT_S", "4.0"))
# $ per 1M tokens for Claude Haiku 5.5 (prompts under 100K tokens)
PRICE_IN, PRICE_OUT, PRICE_CACHE_READ, PRICE_CACHE_WRITE = 0.10, 0.50, 0.01, 0.125

LANG_NAMES = {
    ("ru", None): "русском",
    ("kk", "cyrillic"): "казахском, кириллицей (настоящие казахские буквы қ ғ ң ә ө ұ ү і)",
    ("kk", "latin"): "казахском, кириллицей (настоящие казахские буквы қ ғ ң ә ө ұ ү і), даже если игрок пишет латиницей",
    ("uz", "latin"): "узбекском латиницей (oʻ, gʻ, sh, ch)",
    ("uz", "cyrillic"): "узбекском латиницей (oʻ, gʻ, sh, ch), как пишут администраторы",
    ("en", None): "английском", ("es", None): "испанском", ("pt", None): "португальском",
}


def lang_name(lang: str, script: str | None) -> str:
    return LANG_NAMES.get((lang, script)) or LANG_NAMES.get((lang, None)) or LANG_NAMES[("ru", None)]

SYSTEM = """Ты помощник администратора поддержки мобильной ролевой игры Grand Mobile. Пишешь ЧЕРНОВИК ответа игроку; администратор его проверит.

Жёсткие правила:
1. Ссылаться можно ТОЛЬКО на пункты из блока «Доступные пункты правил». Если в тексте ответа упоминаешь пункт, он обязан быть в cited. Никаких других номеров правил.
2. НИКОГДА не называй наказание и его срок (мут, бан, варн, джайл, деморган, «получит … на 30 минут»), даже если так написано в примерах админов. Не обещай разбан, снятие наказания, возврат денег или имущества, телепорт. Решение о наказании принимает администратор после проверки доказательств — так и пиши.
3. Факты из жалобы не принимай за истину: игроки преувеличивают и ошибаются в пунктах правил. Скриншотов ты не видишь, видишь только их количество.
4. Пиши коротко (1–3 предложения), вежливо, в стиле администраторов из примеров, без лишних благодарностей. Обращайся на «вы».
5. Язык ответа указан в задании. Казахский пиши настоящими казахскими буквами.
6. Не повторяй оскорбления и не раскрывай личные данные.
7. Ты отвечаешь ЗАЯВИТЕЛЮ (роль player). Имени заявителя ты не знаешь: начинай с приветствия без имени. Не обращайся к обвиняемому.

Как отвечать по типам:
- Жалоба с описанием нарушения и доказательствами: жалоба принята, похоже на нарушение пункта X, решение примет администратор после проверки.
- Жалоба без доказательств: нужны скриншоты или видео (ЖБ 1.2.2), подать повторно.
- Убийство без причины (DM/DB/SK), читы, нарушения РП: администраторы обычно требуют видеофиксацию, одного скриншота мало. Если в примерах админы просили видео, попроси его и ты.
- Жалоба расплывчатая (непонятно кто, что, когда): попроси описать суть, ник нарушителя и доказательства (ЖБ 1.2.3).
- Не жалоба или бессодержательная: раздел жалоб только для нарушений (ЖБ 1.1.2 или ЖБ 1.2.3).
- Несогласие с наказанием или жалоба на администратора: передано старшей администрации, решение после проверки.
- Репорт «застрял/перевернулся»: запрос передан администратору; можно подсказать кнопку эвакуатора на спидометре.
- Вопрос по механике: ответь, только если ответ есть в примерах или правилах; иначе попроси уточнить вопрос.
- Оффтоп/«помогите» без сути: попроси описать проблему (репорт для вопросов и жалоб, пункт 3.1).
- Аккаунт, взлом, пропажа имущества, платежи: передано специалисту поддержки, ничего не обещать.

needs_human=true, если нужно решение о наказании, проверка доказательств, работа администратора в игре, апелляция, платёж, аккаунт, или ты не уверен. reasons — 1–2 короткие причины ВСЕГДА на русском, независимо от языка игрока, до 8 слов каждая. cited — только идентификаторы из списка, точно как там написано (например "2.1", "ЖБ 1.2.2", "GZ 1.1", "Правила для ОПГ 1.3"). Разделы без кода называй полным названием, без своих сокращений.
Ответь только JSON по схеме."""


# One fixed schema for every request: a per-request enum would force grammar recompilation (slow).
# Citations are validated against the rule base after the call instead.
SCHEMA = {
    "type": "object",
    "properties": {
        "draft_text": {"type": "string"},
        "cited": {"type": "array", "items": {"type": "string"}},
        "nickname": {"type": ["string", "null"]},
        "needs_human": {"type": "boolean"},
        "confidence": {"type": "number"},
        "reasons": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["draft_text", "cited", "nickname", "needs_human", "confidence", "reasons"],
    "additionalProperties": False,
}


class Drafter:
    def __init__(self):
        # Production is local-only: player texts must not leave the server. The Claude backend is kept for
        # development experiments and is used only with DRAFT_BACKEND=api.
        self.enabled = (os.getenv("DRAFT_BACKEND", "local") == "api" and bool(os.getenv("ANTHROPIC_API_KEY"))
                        and os.getenv("DRAFT_LLM", "1") != "0")
        # No retries: on timeout we answer from a template so p95 stays under 5 s.
        self.client = anthropic.Anthropic(timeout=TIMEOUT_S, max_retries=0) if self.enabled else None

    def build_prompt(self, t: dict, ctx: dict) -> str:
        parts = [f"Тип обращения: {t['kind']}", f"Категория (классификатор): {ctx['category']}",
                 f"Язык ответа: {lang_name(ctx['lang'], ctx.get('script'))}",
                 f"Вложений (скриншотов): {ctx['attachments']}; доказательства: {'есть' if ctx['has_evidence'] else 'нет'}"]
        if ctx.get("player_refs"):
            parts.append(f"Игрок указал пункты: {', '.join(ctx['player_refs'])} (может ошибаться)")
        if ctx.get("reported_nick"):
            parts.append(f"Ник ОБВИНЯЕМОГО (это не заявитель, к нему не обращайся): {ctx['reported_nick']}")
        parts.append("\nСообщения:")
        for m in t["messages"]:
            if m.get("role") == "admin":
                continue
            parts.append(f"[{m.get('role')}] {ctx['norm'](m.get('text', ''))[:1200]}")
        parts.append("\nДоступные пункты правил:\n" + (ctx["rules_text"] or "(нет — ответ без ссылок на правила)"))
        if ctx.get("examples"):
            parts.append("\nПохожие обращения и реальные ответы администраторов (стиль и типичное решение):")
            for e in ctx["examples"]:
                # Punishments are hidden so the model has nothing to copy (drafts must not name them).
                answer = PUNISHMENT.sub("<наказание>", e["a"][:300])
                parts.append(f"- игрок: {e['q'][:160]}\n  админ: {answer}")
        return "\n".join(parts)

    def _parse(self, msg, cites):
        text = next((b.text for b in msg.content if b.type == "text"), None)
        data = json.loads(text)
        u = msg.usage
        cost = ((u.input_tokens or 0) * PRICE_IN + (u.output_tokens or 0) * PRICE_OUT
                + (u.cache_read_input_tokens or 0) * PRICE_CACHE_READ
                + (u.cache_creation_input_tokens or 0) * PRICE_CACHE_WRITE) / 1e6
        data["_cost_usd"] = cost
        return data

    def _params(self, prompt: str, cites: list[str]) -> dict:
        return dict(
            model=MODEL,
            max_tokens=400,
            thinking={"type": "disabled"},
            output_config={"effort": "low", "format": {"type": "json_schema", "schema": SCHEMA}},
            system=[{"type": "text", "text": SYSTEM, "cache_control": {"type": "ephemeral"}}],
            messages=[{"role": "user", "content": prompt}],
        )

    def draft(self, prompt: str, cites: list[str]) -> dict | None:
        if not self.enabled:
            return None
        try:
            msg = self.client.messages.create(**self._params(prompt, cites))
            return self._parse(msg, cites)
        except (anthropic.APIError, anthropic.APITimeoutError, json.JSONDecodeError, StopIteration, TypeError):
            return None


RULE_MENTION = re.compile(r"(?:ЖБ|GZ|ГЗ|ОП)?\s?\d{1,2}\.\d{1,2}(?:\.\d{1,2})?(?:\s?(?:ОП|оп|OP))?")
PROMISE = re.compile(
    r"разбан|сним(ем|ут|у)\s+(бан|мут|наказан|блокир)|верн(ём|ем|ут)|возврат|компенс|"
    r"(получит|будет выдан[оа]?|выдадим|накажем|будет наказан)\s*(мут|бан|варн|джайл|деморган|наказан|блокир)?|"
    r"телепортир(ую|уем)|unban|refund|qaytar|бан(ы|ын)? алын|"
    # Actions the service can't perform itself ("слежу за игроком", "проверим", "администратор поможет"): the draft
    # says what is required instead ("требуется проверка администратора").
    r"слеж(у|им)|провер(им|ит|ят)\b|помож(ет|ем|ете)\b|подскаж(ет|ем)|запрос(им|ит)\b|сообщ(им|ит|ат)\b|"
    r"тексер(еміз|еді)\b|көмектес(еді|еміз)\b|сұратамыз|хабарлаймыз|"
    r"tekshir(amiz|adi)\b|yordam beradi|so'raymiz|bildiramiz|"
    r"\bwill (help|check|request|review)\b|\bwe('ll| will) (check|help|let|request|review)", re.IGNORECASE)


# A concrete punishment in the draft ("получит мут на 30 минут", "7 kunga ban"). The organisers asked not to
# name punishments: the admin decides them after checking the evidence.
PUNISHMENT = re.compile(
    r"(получит|получат|будет выдан\w*|выдадут|выдаём|выдаем|накаж\w+|понес[её]т)\s+(\S+\s+){0,3}?"
    r"(мут\w*|бан\w*|блокировк\w*|варн\w*|джайл\w*|деморган\w*|предупрежд\w*|jail|demorgan|warn|mute|ban)|"
    r"(мут|бан|блокировк\w*|джайл|деморган|jail|demorgan|mute|ban)\w*\s+на\s+\d+|"
    r"\d+\s*(kunga|kun)\s+ban|\d+\s*daqiqa\w*\s+(mute|jail)|jazo\s+oladi|"
    r"\d+\s*(күнге|минутқа|тәулікке)", re.IGNORECASE)


def names_punishment(text: str) -> bool:
    return PUNISHMENT.search(text or "") is not None


def rule_mentions(text: str) -> list[str]:
    return [m.group(0).strip() for m in RULE_MENTION.finditer(text or "")]


def has_promise(text: str) -> bool:
    return PROMISE.search(text or "") is not None
