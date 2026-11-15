"""Likely admin verdict for a complaint, shown to the admin as a hint ("похожие жалобы обычно закрывались как ...").

Admins decide by screenshots we never see, so the verdict never goes into the draft text: putting it there lowered
blind-judge scores on validation (see REPORT.md). Also provides violation_key(), used by the rule model.
"""
import pickle
import re
from pathlib import Path

from .text import normalize, player_text

VERDICTS = ["approved", "need_evidence", "rejected", "invalid_format", "appeal_rejected", "appeal_approved",
            "duplicate", "already_punished"]

# Admin replies whose template wasn't recognised (verdict "other", ~35% of complaints) are relabelled
# with these patterns for training. On replies with a known verdict they agree 85% of the time.
RELABEL = [
    ("appeal_rejected", r"наказание выдан\w* (верно|правильно|корректно)|выдано верно|верно выдал|jazo to.g.ri"),
    ("appeal_approved", r"(выдано|выдал\w*) (не ?верно|ошибочно)|наказание (будет )?(снят|сним)|беседа|дисциплинар|приносим .*извинен"),
    ("duplicate", r"дубл|уже (была )?рассмотрен|уже подавал"),
    ("already_punished", r"уже (был\w* )?наказан|уже получил"),
    ("need_evidence", r"доказательств|видеофикс|видеозапис|тайм-?код|фрапс|dalil|недостаточно|предоставьте|запись"),
    ("invalid_format", r"оформлен\w* (с нарушением|неверно|не по форме)|не подлежит|1\.2\.\d"),
    ("rejected", r"нарушени\w* (нет|не обнаружен|не выявлен)|не нарушает|не вижу нарушен|нарушений не|qoidabuzarlik aniqlanmadi"),
    ("approved", r"получит|будет наказан|будет выдан|понес[её]т наказание|наказан\w* в виде|выдан\w* (мут|бан|варн|джайл|деморган|блокир)|oladi|jazo"),
]



def relabel(admin_text: str) -> str | None:
    a = (admin_text or "").lower()
    for v, rx in RELABEL:
        if re.search(rx, a):
            return v
    return None


def training_verdict(t: dict) -> str | None:
    v = t.get("verdict")
    if v in VERDICTS:
        return v
    if v == "other":
        admin = next((m["text"] for m in reversed(t["messages"]) if m["role"] == "admin"), "")
        return relabel(admin)
    return None


def features(t: dict) -> str:
    text = normalize(player_text(t.get("messages") or []), t.get("region"))
    accused = " ".join(str(m.get("text") or "") for m in t.get("messages") or [] if m.get("role") == "reported_player")
    n = t.get("images_count")
    if n is None:
        n = sum(max(int(m.get("attachments") or 0), 0) for m in t.get("messages") or [] if m.get("role") == "player")
    return (f"[img{min(int(n or 0), 4)}] [srv:{t.get('server_lang') or ''}] [refs:{t.get('rule_refs_raw') or ''}] "
            f"[acc:{int(bool(accused))}] {text} || {accused[:200]}")


def violation_key(violation: str | None, canonical) -> str | None:
    """Admin's free-form violation ('2.1 ОП', 'DM', '1.2 GZ') -> key for the punishment table."""
    if not violation:
        return None
    if re.search(r"\d[.,]\d", violation):
        c = canonical(violation)
        if c:
            return c
    v = violation.lower()
    for k, key in [("dm", "DM"), ("дм", "DM"), ("db", "DB"), ("дб", "DB"), ("sk", "SK"), ("ск", "SK"),
                   ("nrp", "NRP"), ("нрп", "NRP"), ("gz", "GZ"), ("зз", "GZ")]:
        if k in v:
            return key
    return None


class VerdictModel:
    def __init__(self, model=None):
        self.model = model

    @classmethod
    def load(cls, models_dir: Path) -> "VerdictModel":
        if not (models_dir / "verdict.pkl").exists():
            return cls()
        with open(models_dir / "verdict.pkl", "rb") as f:
            return cls(pickle.load(f))

    def proba(self, t: dict) -> dict[str, float]:
        if self.model is None:
            return {}
        return {str(c): float(p) for c, p in zip(self.model.classes_, self.model.predict_proba([features(t)])[0])}

    def predict(self, t: dict) -> tuple[str | None, float]:
        p = self.proba(t)
        if not p:
            return None, 0.0
        v = max(p, key=p.get)
        return v, p[v]
