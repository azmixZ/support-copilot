"""Triage service: exactly one endpoint, POST /triage (the case contract). Runs fully locally.
The admin web UI is a separate app (app/admin_app.py, `make admin`), as the organisers asked to keep
the triage service strictly to /triage."""
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from fastapi import FastAPI  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

from .bootstrap import ensure_models  # noqa: E402
from .pipeline import Triage  # noqa: E402

app = FastAPI(title="Grand Mobile Support Copilot", version="1.0", docs_url=None, redoc_url=None, openapi_url=None)
ensure_models()  # builds the models once if `make build` was skipped
triage = Triage()


class Message(BaseModel):
    role: str = "player"
    text: str | None = ""
    attachments: int | None = None
    ts: str | None = None


class TriageRequest(BaseModel):
    kind: str = Field("report", pattern="^(complaint|report)$")
    messages: list[Message]
    # The full dataset record may be sent; every extra field is optional.
    ticket_id: str | None = None
    created_at: str | None = None
    region: str | None = None
    server_lang: str | None = None
    lang: str | None = None
    reporter_hash: str | None = None
    reported_hash: str | None = None
    reported_nick: str | None = None
    rule_refs_raw: str | None = None
    images_count: int | None = None
    status: int | None = None
    verdict: str | None = None


@app.post("/triage")
def triage_endpoint(req: TriageRequest):
    # Sync handler: FastAPI runs it in a thread pool, so requests are processed in parallel.
    result = triage.triage(req.model_dump())
    result.pop("_meta", None)
    return result
