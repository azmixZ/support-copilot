# python3 when it exists (many Linux systems have no `python`), otherwise python; override with PY=...
PY ?= $(shell command -v python3 >/dev/null 2>&1 && echo python3 || echo python)
IN ?= tickets.jsonl
OUT ?= pred.jsonl

.PHONY: install build predict serve eval demo report test dedup learn admin bench categories

install:
	$(PY) -m pip install -r requirements.txt

# Builds the knowledge base and the classifier from data/handout (one-time, ~1 min).
build:
	$(PY) scripts/build_models.py

predict:
	$(PY) predict.py IN=$(IN) OUT=$(OUT)

serve:
	$(PY) -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 1

eval:
	$(PY) scripts/evaluate.py

demo:
	$(PY) scripts/make_demo_queue.py 60

report:
	$(PY) scripts/make_report.py

test:
	$(PY) -m pytest -q tests

dedup:
	$(PY) scripts/eval_dedup.py

# Retrain the classifier with admin corrections from the admin UI (data/feedback.jsonl).
learn:
	$(PY) scripts/build_models.py

# Admin web UI (bonus): http://localhost:8001/admin
admin:
	$(PY) -m uvicorn app.admin_app:app --host 127.0.0.1 --port 8001

bench:
	$(PY) scripts/bench_http.py 1000

categories:
	$(PY) scripts/eval_categories.py
