"""Knowledge base of past tickets with admin answers (the "база знаний" part of RAG).

For a new ticket we find similar past tickets of the same kind and use:
  * their admin replies as style examples for the draft,
  * the distribution of verdicts as a hint of the likely outcome.
"""
import gzip
import json
import pickle
import re
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

from .text import normalize, player_text

GENERIC_REPLY = re.compile(r"^\W*(ожидайте|слежу|в пути|\.|нет)\W*$", re.IGNORECASE)


def _clean_reply(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


class KnowledgeBase:
    def __init__(self, kinds: dict):
        self.kinds = kinds  # kind -> {"vec", "matrix", "items"}

    @classmethod
    def build(cls, train_path: str | Path, max_per_kind: int = 60000) -> "KnowledgeBase":
        rows = {"complaint": [], "report": []}
        with gzip.open(train_path, "rt", encoding="utf-8") as f:
            for line in f:
                t = json.loads(line)
                admin = [m["text"] for m in t["messages"] if m["role"] == "admin"]
                ptxt = normalize(player_text(t["messages"]), t.get("region"))
                if not admin or not ptxt or len(ptxt) < 2:
                    continue
                reply = _clean_reply(admin[-1])
                if len(reply) < 8 or GENERIC_REPLY.match(reply):
                    continue
                rows[t["kind"]].append({
                    "q": ptxt[:400],
                    "a": reply[:700],
                    "verdict": t.get("verdict"),
                    "violation": t.get("violation"),
                    "punishment": t.get("punishment"),
                    "lang": t.get("lang"),
                    "images": t.get("images_count") or 0,
                })
        kinds = {}
        for kind, items in rows.items():
            items = items[:max_per_kind]
            vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), sublinear_tf=True, min_df=2, max_features=200000)
            matrix = vec.fit_transform([i["q"] for i in items])
            kinds[kind] = {"vec": vec, "matrix": matrix, "items": items}
        return cls(kinds)

    def save(self, path: str | Path):
        with open(path, "wb") as f:
            pickle.dump(self.kinds, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: str | Path) -> "KnowledgeBase":
        with open(path, "rb") as f:
            return cls(pickle.load(f))

    def similar(self, kind: str, text: str, k: int = 5, lang: str | None = None) -> list[dict]:
        kb = self.kinds.get(kind)
        if kb is None or not text.strip():
            return []
        sims = linear_kernel(kb["vec"].transform([text]), kb["matrix"]).ravel()
        top = np.argpartition(-sims, min(60, len(sims) - 1))[:60]
        top = top[np.argsort(-sims[top])]
        out, seen = [], set()
        # Prefer answers in the player's language, then fill with the rest.
        for want_lang in ([lang] if lang else []) + [None]:
            for i in top:
                it = kb["items"][i]
                if want_lang and it["lang"] != want_lang:
                    continue
                key = it["a"][:60]
                if key in seen:
                    continue
                seen.add(key)
                out.append({**it, "score": float(sims[i])})
                if len(out) >= k:
                    return out
        return out

    @staticmethod
    def verdict_hint(neigh: list[dict]) -> str | None:
        c = Counter(n["verdict"] for n in neigh if n.get("verdict") and n["verdict"] not in ("other", "closed"))
        return c.most_common(1)[0][0] if c else None
