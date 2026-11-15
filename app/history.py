"""Ticket history for the admin card: past complaints against the accused and past tickets of the reporter.

Built from the dataset (train + valid) and extended by every ticket the service processes. Records older
than HISTORY_DAYS (6 months by default, as agreed with the organisers) are dropped on load and on save.
Stored locally only (models/history.pkl + data/history_live.jsonl).
"""
import gzip
import json
import os
import pickle
import threading
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

HISTORY_DAYS = int(os.getenv("HISTORY_DAYS", "183"))


def _record(t: dict, category: str | None = None) -> dict:
    player = " ".join(str(m.get("text") or "") for m in t.get("messages") or [] if m.get("role") == "player")
    admin = next((m.get("text") for m in reversed(t.get("messages") or []) if m.get("role") == "admin"), None)
    return {
        "ticket_id": t.get("ticket_id"), "created_at": t.get("created_at"), "kind": t.get("kind"),
        "text": player[:200], "verdict": t.get("verdict"), "punishment": t.get("punishment"),
        "violation": t.get("violation"), "category": category, "admin_reply": (admin or "")[:160] or None,
        "reported_nick": t.get("reported_nick"),
    }


def _fresh(rec: dict, now: datetime) -> bool:
    try:
        return datetime.fromisoformat(rec["created_at"]) >= now - timedelta(days=HISTORY_DAYS)
    except (TypeError, ValueError, KeyError):
        return True


class History:
    def __init__(self, by_reporter=None, by_reported=None, live_path: Path | None = None):
        self.by_reporter = by_reporter or defaultdict(list)
        self.by_reported = by_reported or defaultdict(list)
        self.live_path = live_path
        self.lock = threading.Lock()

    @classmethod
    def build(cls, paths: list[Path]) -> "History":
        h = cls(defaultdict(list), defaultdict(list))
        for path in paths:
            opener = gzip.open if str(path).endswith(".gz") else open
            with opener(path, "rt", encoding="utf-8") as f:
                for line in f:
                    t = json.loads(line)
                    h._add(t, _record(t))
        return h

    def save(self, path: Path):
        with open(path, "wb") as f:
            pickle.dump((dict(self.by_reporter), dict(self.by_reported)), f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: Path, live_path: Path | None = None, now: datetime | None = None) -> "History":
        if not path.exists():
            return cls(live_path=live_path)
        with open(path, "rb") as f:
            rep, red = pickle.load(f)
        h = cls(defaultdict(list, rep), defaultdict(list, red), live_path)
        if live_path and live_path.exists():
            for line in open(live_path, encoding="utf-8"):
                rec = json.loads(line)
                h._add(rec["_keys"], rec["rec"])
        h.prune(now or datetime.now())
        return h

    def _add(self, keys: dict, rec: dict):
        if keys.get("reporter_hash"):
            self.by_reporter[keys["reporter_hash"]].append(rec)
        if keys.get("reported_hash"):
            self.by_reported[keys["reported_hash"]].append(rec)

    def add_live(self, t: dict, category: str | None):
        """Remember a processed ticket (for the next cards) and persist it locally."""
        if not (t.get("reporter_hash") or t.get("reported_hash")):
            return
        rec = _record(t, category)
        keys = {"reporter_hash": t.get("reporter_hash"), "reported_hash": t.get("reported_hash")}
        with self.lock:
            self._add(keys, rec)
            if self.live_path:
                with open(self.live_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps({"_keys": keys, "rec": rec}, ensure_ascii=False) + "\n")

    def prune(self, now: datetime):
        for index in (self.by_reporter, self.by_reported):
            for k in list(index):
                index[k] = [r for r in index[k] if _fresh(r, now)]
                if not index[k]:
                    del index[k]

    def lookup(self, reporter: str | None, reported: str | None, exclude: str | None = None, limit: int = 10) -> dict:
        def recent(items):
            items = [r for r in items if r.get("ticket_id") != exclude]
            return sorted(items, key=lambda r: r.get("created_at") or "", reverse=True)[:limit]

        against = recent(self.by_reported.get(reported, [])) if reported else []
        mine = recent(self.by_reporter.get(reporter, [])) if reporter else []
        return {"against_accused": against, "from_reporter": mine,
                "against_accused_total": len(self.by_reported.get(reported, [])) if reported else 0,
                "from_reporter_total": len(self.by_reporter.get(reporter, [])) if reporter else 0}
