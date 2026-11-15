"""HTTP latency of POST /triage: sequential requests with validation tickets (without the admin answer).

  python scripts/bench_http.py [N]     # the service must be running: make serve
Writes reports/http_latency.json.
"""
import json
import statistics
import sys
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
N = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
ADMIN = ("verdict", "violation", "punishment", "status")

tickets = []
for line in open(ROOT / "data/handout/tickets_valid.jsonl", encoding="utf-8"):
    t = {k: v for k, v in json.loads(line).items() if k not in ADMIN}
    t["messages"] = [m for m in t["messages"] if m.get("role") != "admin"]
    tickets.append(t)
tickets = (tickets * (N // len(tickets) + 1))[:N]

lat, ok = [], 0
with httpx.Client(base_url="http://127.0.0.1:8000", timeout=10) as c:
    c.post("/triage", json=tickets[0])  # warm-up
    for t in tickets:
        s = time.perf_counter()
        r = c.post("/triage", json=t)
        lat.append((time.perf_counter() - s) * 1000)
        ok += r.status_code == 200
lat.sort()
rep = {"requests": N, "ok": ok, "p50_ms": round(statistics.median(lat), 2),
       "p95_ms": round(lat[int(N * 0.95) - 1], 2), "max_ms": round(lat[-1], 2),
       "note": "sequential requests to a local server, Windows 10, laptop CPU"}
(ROOT / "reports/http_latency.json").write_text(json.dumps(rep, indent=1), encoding="utf-8")
print(json.dumps(rep, indent=1))
