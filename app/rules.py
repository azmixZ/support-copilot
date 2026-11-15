"""Rule base: canonical citation ids, retrieval (RAG) and citation validation.

Citation format:
  "2.1"         general rules (ОП) — the default, as in the case contract
  "ЖБ 1.2.7"    complaint rules
  "GZ 1.1"      green zone rules
  "Правила для ОПГ 1.3"   sections without an official code: full section title + point
                          (organisers asked not to invent abbreviations)
"""
import json
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# Codes the game and its admins actually use. Other sections are cited by their full title.
SECTION_CODES = {
    30: "",      # Общие правила игры (ОП) — bare point number, as in the case contract
    31: "ЖБ",    # Правила жалоб
    32: "GZ",    # Правила нахождения в Зелёной Зоне
}
SECTION_ALIASES = {
    "": ["оп", "op", "общие", "общих"],
    "ЖБ": ["жб", "правила жалоб", "жалоб"],
    "GZ": ["gz", "гз", "зз", "зелен", "зелён", "green"],
}


TERM_RE = re.compile(r"•\s*([^(•\n]+?)\s*\(сокращенно\s+([^)]+)\)\s*[—-]\s*([^•]+)")
# How players and admins write the terms.
TERM_ALIASES = {"dm": "dm", "дм": "dm", "deathmatch": "dm", "db": "db", "дб": "db", "sk": "sk", "ск": "sk",
                "tk": "tk", "тк": "tk", "rk": "rk", "рк": "rk", "pg": "pg", "пг": "pg", "mg": "mg", "мг": "mg",
                "nrp": "nonrp", "нрп": "nonrp", "nonrp": "nonrp", "non rp": "nonrp", "нонрп": "nonrp",
                "mass dm": "mass dm", "масс дм": "mass dm", "массдм": "mass dm", "mass sk": "mass sk"}


@dataclass
class RulePoint:
    cite: str        # canonical id used in answers
    section_id: int
    section: str     # section title
    ref: str         # point number inside the section
    text: str

    @property
    def penalty(self) -> str:
        parts = [p.strip() for p in self.text.split("|")[1:]]
        return " | ".join(parts)

    @property
    def body(self) -> str:
        return self.text.split("|")[0].strip()


def cite_id(section_id: int, ref: str, title: str = "") -> str:
    code = SECTION_CODES.get(section_id)
    if code is None:
        code = title
    return f"{code} {ref}".strip()


class RuleBase:
    def __init__(self, path: str | Path):
        # NFC: the source file stores some letters decomposed ("й" as "и" + U+0306), which breaks matching.
        data = json.loads(unicodedata.normalize("NFC", Path(path).read_text(encoding="utf-8")))
        self.points: list[RulePoint] = []
        self.sections: dict[int, str] = {}
        self.terms: dict[str, str] = {}  # "dm" -> "RP термины и их описание DM"
        for sec in data["rules"]:
            self.sections[sec["id"]] = sec["category"]
            if not sec.get("points"):
                # "RP термины и их описание" has no numbered points: each term is citable by its abbreviation,
                # e.g. "RP термины и их описание DM".
                for name, abbr, desc in TERM_RE.findall(sec.get("text", "")):
                    cite = f"{sec['category']} {abbr.strip()}"
                    self.points.append(RulePoint(cite, sec["id"], sec["category"], abbr.strip(),
                                                 f"{name.strip()} ({abbr.strip()}) — {desc.strip()}"))
                    self.terms[abbr.strip().lower()] = cite
            for p in sec.get("points", []):
                text = p["text"].strip()
                # Skip pure headers like "ОБЩИЕ ПОЛОЖЕНИЯ:" — they are not citable rules.
                if len(text) < 25 and text.endswith(":"):
                    continue
                self.points.append(RulePoint(cite_id(sec["id"], p["ref"], sec["category"]), sec["id"], sec["category"], p["ref"], text))
        self.by_cite = {p.cite: p for p in self.points}
        corpus = [f"{p.section}. {p.text}" for p in self.points]
        self.vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True, min_df=1)
        self.matrix = self.vec.fit_transform(corpus)

    # ---------- validation ----------
    def exists(self, cite: str) -> bool:
        return cite in self.by_cite

    def validate(self, cites: list[str]) -> list[str]:
        """Keep only citations that exist in the rule base (dedup, keep order)."""
        out = []
        for c in cites or []:
            c = self.canonical(c)
            if c and c in self.by_cite and c not in out:
                out.append(c)
        return out

    def canonical(self, raw: str) -> str | None:
        """Map free-form references like '2.1 оп', 'ЖБ 1.2.7', 'gz1.1' to a canonical id."""
        if not raw:
            return None
        s = unicodedata.normalize("NFC", raw).strip().lower().replace(",", ".")
        m = re.search(r"(\d+(?:\.\d+){1,2})", s)
        if not m:
            return self.term(s)
        num = m.group(1)
        rest = (s[:m.start()] + " " + s[m.end():]).strip()
        code = ""
        titled = [t for sid, t in self.sections.items() if sid not in SECTION_CODES and t.lower() in rest]
        if titled:
            code = max(titled, key=len)
        else:
            for c, aliases in SECTION_ALIASES.items():
                if any(a in rest for a in aliases):
                    code = c
                    break
        cand = f"{code} {num}".strip()
        if cand in self.by_cite:
            return cand
        return None

    def term(self, s: str) -> str | None:
        """'дм', 'DM в зз', 'non rp cop' -> citation of the RP term (longest alias wins)."""
        s = s.lower()
        for alias in sorted(TERM_ALIASES, key=len, reverse=True):
            if re.search(rf"(?<!\w){re.escape(alias)}(?!\w)", s):
                cite = self.terms.get(TERM_ALIASES[alias])
                if cite:
                    return cite
        return None

    def parse_player_refs(self, raw: str | None) -> list[str]:
        """Extract rule references the player typed (often wrong — used as a hint only)."""
        if not raw:
            return []
        found = []
        for m in re.finditer(r"([a-zа-яё]{0,3}\s*\d+(?:[.,]\d+){1,2}\s*[a-zа-яё]{0,3})", raw.lower()):
            c = self.canonical(m.group(1))
            if c and c not in found:
                found.append(c)
        term = self.term(raw)  # "дм", "нрп коп" -> RP term
        if term and term not in found:
            found.append(term)
        return found

    # ---------- retrieval ----------
    def search(self, query: str, k: int = 6, allowed: list[str] | None = None) -> list[RulePoint]:
        if not query.strip():
            return []
        sims = linear_kernel(self.vec.transform([query]), self.matrix).ravel()
        order = sims.argsort()[::-1]
        out = []
        for i in order:
            p = self.points[i]
            if allowed is not None and not any(
                    p.cite == a or p.cite.startswith(a + ".") or p.cite.startswith(a + " ") for a in allowed):
                continue
            out.append(p)
            if len(out) >= k:
                break
        return out

    def context(self, cites: list[str], max_chars: int = 420) -> str:
        lines = []
        for c in cites:
            p = self.by_cite.get(c)
            if p:
                lines.append(f"[{p.cite}] ({p.section}) {p.text[:max_chars]}")
        return "\n".join(lines)
