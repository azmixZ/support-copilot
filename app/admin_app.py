"""Admin web UI (bonus), a separate local app: `make admin` -> http://localhost:8001/admin"""
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from fastapi import FastAPI  # noqa: E402
from fastapi.responses import RedirectResponse  # noqa: E402

from . import admin  # noqa: E402
from .history import History  # noqa: E402
from .main import TriageRequest  # noqa: E402
from .pipeline import MODELS, ROOT, Triage  # noqa: E402

app = FastAPI(title="Support Copilot · admin")
triage = Triage()
history = History.load(MODELS / "history.pkl", live_path=ROOT / "data" / "history_live.jsonl")
admin.state.update(history=history, triage=triage)
admin.prune_feedback()
app.include_router(admin.router)


@app.post("/api/triage")
def triage_for_admin(req: TriageRequest):
    """Same pipeline as POST /triage; the ticket is also remembered for the history in the next cards."""
    ticket = req.model_dump()
    result = triage.triage(ticket)
    history.add_live(ticket, result["category"])
    result.pop("_meta", None)
    return result


@app.get("/api/rules")
def rules(cites: str = ""):
    return {c: triage.rules.by_cite[c].text for c in cites.split("|") if c in triage.rules.by_cite}


@app.get("/")
def root():
    return RedirectResponse("/admin")
