"""Admin web UI backend: queue, bulk actions, history in the card, urgent-ticket alerts, edits as training signal.

Everything is local: the queue, admin edits and history are files next to the service. Edits and history
are kept for HISTORY_DAYS (6 months by default) and pruned on start.
"""
import json
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from .history import HISTORY_DAYS

ROOT = Path(__file__).resolve().parent.parent
QUEUE = ROOT / "data" / "demo_queue.jsonl"
FEEDBACK = ROOT / "data" / "feedback.jsonl"
PRIORITY_ORDER = {"p0": 0, "p1": 1, "p2": 2, "p3": 3}
URGENT_CATEGORIES = {"report.account", "report.payment"}

router = APIRouter()
_lock = threading.Lock()
state = {"history": None, "triage": None}  # set by app.main


def prune_feedback(now: datetime | None = None):
    """Keep admin edits for HISTORY_DAYS only."""
    if not FEEDBACK.exists():
        return
    limit = (now or datetime.now()) - timedelta(days=HISTORY_DAYS)
    rows = [json.loads(l) for l in open(FEEDBACK, encoding="utf-8") if l.strip()]
    keep = [r for r in rows if datetime.fromisoformat(r.get("ts", "2100-01-01T00:00:00")) >= limit]
    if len(keep) != len(rows):
        with _lock, open(FEEDBACK, "w", encoding="utf-8") as f:
            f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in keep)


def is_urgent(pred: dict) -> bool:
    """Tickets that need an alert: p0 (when enabled), account / payment, threats outside the game."""
    return (pred.get("priority") == "p0" or pred.get("category") in URGENT_CATEGORIES
            or any("угроз" in r for r in pred.get("reasons", [])))


def _done_ids() -> set:
    if not FEEDBACK.exists():
        return set()
    return {json.loads(l)["ticket_id"] for l in open(FEEDBACK, encoding="utf-8") if l.strip()}


def _load_queue() -> list[dict]:
    if not QUEUE.exists():
        return []
    done = _done_ids()
    items = [json.loads(l) for l in open(QUEUE, encoding="utf-8") if l.strip()]
    items = [i for i in items if i["ticket"].get("ticket_id") not in done]
    for i in items:
        i["urgent"] = is_urgent(i["pred"])
    # Urgent first, then priority, then tickets that need a human, then oldest first.
    items.sort(key=lambda i: (not i["urgent"], PRIORITY_ORDER.get(i["pred"]["priority"], 9),
                              not i["pred"]["needs_human"], i["ticket"].get("created_at") or ""))
    return items


@router.get("/api/queue")
def queue(limit: int = 300):
    items = _load_queue()
    return {"total": len(items), "urgent": sum(i["urgent"] for i in items), "items": items[:limit]}


@router.get("/api/history")
def history(reporter: str | None = None, reported: str | None = None, exclude: str | None = None):
    h = state["history"]
    if h is None:
        return {"against_accused": [], "from_reporter": [], "against_accused_total": 0, "from_reporter_total": 0}
    return h.lookup(reporter, reported, exclude)


class Feedback(BaseModel):
    ticket_id: str
    action: str  # send | edit | escalate | close
    draft_original: str = ""
    draft_final: str | None = None
    category: str | None = None
    priority: str | None = None
    category_corrected: str | None = None
    note: str | None = None
    ticket: dict | None = None  # kind, player messages, refs, attachments: lets edits become training data


def _write(rows: list[dict]):
    with _lock, open(FEEDBACK, "a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _row(fb: Feedback) -> dict:
    if fb.action not in ("send", "edit", "escalate", "close"):
        raise HTTPException(400, "action must be send | edit | escalate | close")
    row = fb.model_dump()
    row["edited"] = bool(fb.draft_final and fb.draft_final.strip() != fb.draft_original.strip())
    row["ts"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    return row


@router.post("/api/feedback")
def feedback(fb: Feedback):
    _write([_row(fb)])
    return {"ok": True}


class Bulk(BaseModel):
    action: str  # send | escalate | close
    items: list[Feedback]


@router.post("/api/feedback/bulk")
def feedback_bulk(b: Bulk):
    """Mass actions: send the drafts as they are, escalate, or close spam / duplicates."""
    rows = [_row(Feedback(**{**i.model_dump(), "action": b.action})) for i in b.items]
    _write(rows)
    return {"ok": True, "n": len(rows)}


@router.get("/api/stats")
def stats():
    rows = [json.loads(l) for l in open(FEEDBACK, encoding="utf-8")] if FEEDBACK.exists() else []
    return {
        "handled": len(rows),
        "sent_as_is": sum(r["action"] == "send" and not r["edited"] for r in rows),
        "edited": sum(r["edited"] for r in rows),
        "escalated": sum(r["action"] == "escalate" for r in rows),
        "closed": sum(r["action"] == "close" for r in rows),
    }


@router.get("/admin", response_class=HTMLResponse)
def admin_page():
    return (Path(__file__).parent / "static" / "admin.html").read_text(encoding="utf-8")
