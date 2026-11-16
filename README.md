**English** · [Русский](README.ru.md)

# Support Copilot · Apex Protocol

Solution to case 03 "Support Copilot" (Grand Mobile × Decentrathon 6.0).

> **The data, the labels and the trained weights are not part of this repository.** Under the hackathon rules, Grand Mobile data and anything derived from it must not be published. Only the code, the report and the documents are here. To run the service you need the case archive that was given to participants. Examples in tests and documents are our own phrases; the nickname `Arman_Petrov` comes from the example in the case statement.

The service triages player tickets **fully locally**, with no external APIs: player texts never leave the server. What it does:
- detects the language (Russian, Kazakh, Uzbek, English; Cyrillic, Latin script and transliteration), the category and the priority;
- filters out spam and duplicates, including implicit ones;
- extracts facts: nickname, rule references, evidence, server region;
- drafts a reply in the player's language that cites a verified rule;
- flags when a human is needed.

A response takes a few milliseconds on an ordinary CPU (p95 ≈ 7 ms).

**Documents:** [report with metrics and examples](REPORT.md) · [presentation](docs/presentation.html) (open in a browser, flip with the arrow keys) · [pitch text](docs/PITCH.md) (Russian)

| | |
|---|---|
| Invented rule references | **0** on 1000 validation tickets |
| Category accuracy | **80.7%** against LLM labels (14 categories, held-out set), **88.8%** agreement with admin answers |
| Priority | macro-F1 **0.87** by the organizers' rule (violation type) |
| Main rule = the admin's rule | **90%** |
| Speed / cost | p95 **7 ms** on CPU / **$0** |

## Quick start

Python 3.10+ is required (tested on 3.12 in Docker and 3.14 on Windows).

```bash
python3 -m venv .venv && . .venv/bin/activate     # Windows: python -m venv .venv && .venv\Scripts\activate
pip install -r requirements.txt
mkdir -p data && tar -xzf grand-support-copilot-handout-2026-10-27.tgz -C data   # case archive -> data/handout/
#   (inside: rules.json, taxonomy.json, tickets_train.jsonl.gz, tickets_valid.jsonl)
make build                                         # models, knowledge base and history (~2 min, local)
make predict IN=tickets.jsonl OUT=pred.jsonl       # batch prediction (CLI)
make serve                                         # service: POST /triage on :8000
make admin                                         # admin UI (bonus) at http://localhost:8001/admin
make demo                                          # admin queue built from validation tickets
make test                                          # 73 tests (need the case data in data/handout)
make eval                                          # validation metrics -> reports/metrics.json
```

Without `make` (Windows): `python scripts/build_models.py`, `python predict.py IN=tickets.jsonl OUT=pred.jsonl`, `python -m uvicorn app.main:app --port 8000`, admin UI: `python -m uvicorn app.admin_app:app --port 8001`.

With Docker (the data is mounted, not baked into the image):

```bash
docker build -t apex-copilot .
docker run -p 8000:8000 -v "$PWD/data:/app/data" apex-copilot
```

No API key is needed to run the service. `make build` trains the main-rule model, the verdict model and the knowledge base from the case data (~2–3 minutes, locally). At the hackathon the category classifier was trained on labels for 6000 tickets produced by a strong model (`scripts/label_llm.py`, needs an Anthropic key); those labels and the weights are not published. Without them the classifier falls back to keyword rules: category accuracy is lower (63.9% vs 80.7% on validation), everything else works the same. If you skip `make build`, `make predict` and `make serve` build the models on first run.

## Contract

`POST /triage`: request and response follow the case contract. Extra fields from the dataset schema (`ticket_id`, `created_at`, `region`, `reporter_hash`, `reported_hash`, `reported_nick`, `rule_refs_raw`, `images_count`) are optional. When present, they improve Kazakh normalization, fact extraction, deduplication and history.

```json
{
  "lang": "ru", "category": "complaint.insult", "priority": "p2", "is_spam_or_duplicate": false,
  "extracted": {"nickname": "Arman_Petrov", "rule_refs": ["2.1"], "has_evidence": true, "server": null},
  "draft_reply": {"lang": "ru", "cited": ["2.1"],
    "text": "Приветствую! Жалоба на игрока Arman_Petrov принята и передана администратору. По описанию это похоже на нарушение пункта 2.1 ОП. Решение примет администратор после проверки доказательств. Спасибо за обращение."},
  "needs_human": true, "confidence": 0.75,
  "reasons": ["the punishment decision is confirmed by an administrator"]
}
```

(The draft reply is in Russian because the ticket is in Russian. Roughly: "Hello! The complaint about player Arman_Petrov has been accepted and passed to an administrator. By the description it looks like a violation of rule 2.1 of the general rules. The administrator will decide after checking the evidence. Thank you for contacting us.")

`make predict` writes `pred.jsonl`: one line per ticket, `{"ticket_id": ..., <the /triage response>}`. A broken line in the input does not crash the run: it gets a safe "needs a human" response.

`extracted.server` is the server **region** (`cis`, `kz`, `uz`, `eu`, `latam`), like the `region` field in the data, not a server number. It is taken from the request or from an explicit mention in the text ("kz server"). Otherwise it is `null`: the region is never guessed from the language.

The service has **a single endpoint**, `POST /triage`, as the organizers asked. The admin UI is a separate application (`app/admin_app.py`, port 8001) with its own `/admin` and `/api/*`.

The `lang` field in the response takes `ru`, `kk`, `uz`, `en` or `und`: the case has four languages, others (`es`, `pt` in the data) are returned as `und`.

### Rule reference format

| Section | Format | Example |
|---|---|---|
| General rules (ОП) | rule number | `2.1` |
| Complaint rules | `ЖБ <rule>` | `ЖБ 1.2.7` |
| Green zone | `GZ <rule>` | `GZ 1.1` |
| Sections without a code | full section name + rule | `Правила для ОПГ 1.3` |
| RP terms | section name + term | `RP термины и их описание DM` |

## How it works

```
ticket
  → normalization: Unicode NFC, cp1251-Kazakh (ќ→қ, ґ→ғ, ѕ→ң, є→ә, ј→ө, ў→ұ, ї→ү)
  → language and script: ru / kk / uz / en, transliteration ("osk rodni" → ru), Uzbek Cyrillic, Kazakh Latin
  → category: TF-IDF + logistic regression; explicit "hack / donate" always map to account / payment
  → priority: by violation type, screenshots do not matter (taxonomy + organizers' answer): p0 account access, payment,
     off-game threats; p1 cheats, scam, RMT, complaints about admins; p2 insults, DM, RP; p3 mechanics, off-topic, empty complaints
  → spam ("hello?", "+", flood, empty) and duplicates (complaint: same player against the same accused within an hour;
     report: same player, similar text within an hour). History uses only tickets the service processed itself,
     train/valid are not used
  → facts: nickname, the player's rules and RP terms, evidence, region
  → main rule: a local model trained on 12.7k answers where an admin named the violation (90% agreement with the admin)
  → draft: situational templates in ru / kk (Cyrillic, with Kazakh letters) / uz (Latin) / en
     + answers from the admins' knowledge base
  → guard: every citation is checked against rules.json; punishments and promises never reach the draft
  → needs_human and a hint for the admin: likely verdict from 59k past complaints
```

**No hallucinations.** The draft is assembled from verified phrases, and a rule reference is inserted only from the rule base. An invented rule cannot get into a reply. This is checked by tests and on validation: 0 non-existent citations.

**The service is called before an admin answers**, so it relies only on the contract fields: `kind`, the messages of the player and of the accused, and the attachment count. All other fields are optional. The category follows the nature of the violation, the priority follows the violation type, and screenshots go to `has_evidence`. If a record does contain an appeal verdict or a violation named by an admin, the service takes it into account.

**There are no punishments in the draft**, as the organizers asked. The punishment decision belongs to the admin. The likely verdict is shown to them as a separate hint.

**When a human is needed** (`needs_human: true`). Escalation is not penalized, so only spam and greetings are closed without a human. These always go to a human:
- payments, account and hacking;
- complaints about an administrator and appeals;
- unban and refund requests;
- threats.

**Classification.** The provided data has no categories or priorities. So 6000 tickets (in two rounds) were labeled once by a strong model against the taxonomy (`scripts/label_llm.py`, development only). The local classifier is trained on those labels (`scripts/build_models.py`). No external model is used when the service runs.

**Development mode.** `DRAFT_BACKEND=api` with an Anthropic key switches drafting to Claude Haiku. It is kept for comparison only and is off by default: in production, player texts must not be sent outside.

## Admin interface (bonus)

`make admin` → `http://localhost:8001/admin`, local access:
- **bulk actions**: select tickets and send drafts, escalate, or close as spam or duplicate;
- a priority queue: urgent first, then p1 → p3; filters "needs a human", "can be sent", "spam and duplicates";
- **urgent tickets** (account, payment, threats): a red banner, a counter in the tab title, a browser notification;
- ticket card: the draft, the text of the cited rules, the verdict hint, **history** (past complaints about the accused and tickets from the reporter);
- editing the draft, correcting the category, triaging a new ticket live;
- works on a phone.

Admin edits are written to `data/feedback.jsonl`: the original and final draft and the corrected category. This is a training signal. Edits and history are kept for 6 months (`HISTORY_DAYS`), older records are deleted on startup.

## Bonus: new templates

`scripts/suggest_templates.py` clusters tickets and proposes a new template from the most frequent admin answers in each group: `reports/template_suggestions.md`.

## Metrics

`make eval` writes `reports/metrics.json`. Final numbers and the hypotheses we tested are in [REPORT.md](REPORT.md).

## Layout

```
app/text.py         normalization, language, script
app/rules.py        rule base, RP terms, search, citation check
app/knowledge.py    knowledge base built from past admin answers
app/labels.py       category and priority rules
app/rulepick.py     main-rule model
app/verdict.py      likely verdict (hint for the admin)
app/local_draft.py  local drafts in ru / kk / uz / en
app/draft.py        draft checks (punishments, promises, rule numbers) and the API development mode
app/pipeline.py     the whole pipeline
app/history.py      ticket history, 6-month retention
app/main.py         HTTP service
app/admin.py, app/static/admin.html   admin interface
predict.py          batch prediction (CLI)
scripts/            training, evaluation, labeling, templates
tests/              73 tests
```

## Third-party materials

- scikit-learn, NumPy, FastAPI, Uvicorn, python-dotenv (BSD / MIT licenses).
- Claude Sonnet 5.5 (Anthropic API), at development time only: a one-off labeling of 6000 tickets (two rounds; allowed by the organizers), quality evaluation of the drafts and proofreading of our Kazakh and Uzbek templates. It is not used when the service runs.
- Anthropic Python SDK (MIT), for the `DRAFT_BACKEND=api` development mode.
- The code was written during the hackathon together with the AI assistant Claude Code (Anthropic): the problem framing, hypothesis testing, testing and decisions were the team's; commits made with the assistant's participation are marked `Co-Authored-By: Claude`.
- Grand Mobile data, labels, trained weights and derived samples are not in the repository: under the hackathon rules they must not be published.

## Presentation

`docs/presentation.html`: 8 slides, opens in a browser, flip with the arrow keys.

## License

The code is released under the [MIT](LICENSE) license: you may use, modify and distribute it, keeping the author's attribution. The license covers only the code of this repository: Grand Mobile data is not included and is not covered by it.
