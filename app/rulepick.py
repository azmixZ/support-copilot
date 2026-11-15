"""Main rule of a complaint, learned from admin answers (a small local model, no external API).

In train, admins name the violation in ~15k complaint answers ("2.1 ОП", "DM", "1.2 GZ"). A TF-IDF +
logistic regression model maps the player's text to that rule; the draft cites it as the main rule.
"""
import gzip
import json
import pickle
from pathlib import Path

from .text import normalize, player_text
from .verdict import violation_key


def features(t: dict) -> str:
    text = normalize(player_text(t.get("messages") or []), t.get("region"))
    return f"[refs:{t.get('rule_refs_raw') or ''}] {text}"


def key_to_cite(key: str | None, rules) -> str | None:
    """'2.1' / 'GZ 1.1' / 'DM' -> a citation that exists in the rule base."""
    if not key:
        return None
    if key in rules.by_cite:
        return key
    return rules.term(key)


_TERM = "RP термины и их описание "
SPECIFIC_RULES = {"2.1", "2.4", "2.6", "5.2", "4.1", "4.2", "2.3", "4.4", "5.3", "5.4"}
CATEGORY_RULES = {
    "complaint.insult": lambda c: c.split(".")[0] == "2" or c in ("1.6", "3.2", "5.6", "5.7", "5.8"),
    "complaint.dm": lambda c: c.startswith(("GZ ", _TERM)) or c in ("5.1", "5.1.1", "5.3", "5.4", "5.5"),
    "complaint.cheat": lambda c: c in ("5.2",),
    "complaint.scam": lambda c: c.split(".")[0] == "4",
    "complaint.rmt_ads": lambda c: c in ("2.3", "4.4", "4.7"),
    # "Other" violations: any rule that doesn't belong to one of the specific categories above
    # (otherwise the very frequent 2.1 wins and an "other" complaint gets an insult citation).
    "complaint.other": lambda c: not (c.startswith(("GZ ", _TERM)) or c in SPECIFIC_RULES),
}


class RulePicker:
    def __init__(self, model=None):
        self.model = model

    @classmethod
    def load(cls, models_dir: Path) -> "RulePicker":
        path = models_dir / "rulepick.pkl"
        if not path.exists():
            return cls()
        with open(path, "rb") as f:
            return cls(pickle.load(f))

    def predict(self, t: dict, category: str | None = None) -> tuple[str | None, float]:
        """Most likely rule; restricted to rules that fit the category, so that the very frequent 2.1
        doesn't win for a DM or a scam complaint."""
        if self.model is None:
            return None, 0.0
        proba = self.model.predict_proba([features(t)])[0]
        fits = CATEGORY_RULES.get(category)
        best, best_p, total = None, 0.0, 0.0
        for c, p in zip(self.model.classes_, proba):
            if fits is not None and not fits(str(c)):
                continue
            total += p
            if p > best_p:
                best, best_p = str(c), float(p)
        return best, (best_p / total if total else 0.0)


def training_rows(train_path: Path, rules, min_count: int = 20) -> list[tuple[str, str]]:
    from collections import Counter
    rows = []
    with gzip.open(train_path, "rt", encoding="utf-8") as f:
        for line in f:
            t = json.loads(line)
            if t["kind"] != "complaint" or not t.get("violation"):
                continue
            cite = key_to_cite(violation_key(t["violation"], rules.canonical), rules)
            if cite:
                rows.append((features(t), cite))
    counts = Counter(c for _, c in rows)
    return [(x, c) for x, c in rows if counts[c] >= min_count]
