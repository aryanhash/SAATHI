# SAATHI — Smart Assistant for Total Health Intelligence

Voice-first healthcare assistant for India's frontline health workers (ANM/ASHA). Workers speak patient visit details; SAATHI extracts structured data via AI, detects emergencies, stores history in **Qdrant**, maps to government portal shapes (**ANMOL/RCH, U-WIN, NCD**), and serves a **mobile-first dashboard** with emergency routing, **multi-patient OPD sessions**, **printable OPD slips**, and optional **SMS / WhatsApp** reminders.

## Key Features

- **OPD session mode (Vapi)** — one call for the whole shift: **language first** (keypad 1–5: Hindi, English, Kannada, Tamil, Telugu), then **silent-scribe** behaviour: ANM dictates **ID → name → vitals → …**; saying **next** / **agla** / **aage** saves that patient and continues.
- **Server-assisted gap prompts** — after ~2s pause, the UI can ask the backend for a **one-line** missing-field hint and speak it via the SDK (`/api/session-gap-prompt`).
- **40+ field AI extraction** — Gemini (or **Ollama** fallback) extracts name, vitals, symptoms, medicines, vaccines, referral, follow-up date, phone, and more from natural speech.
- **Emergency detection & routing** — rule-based thresholds plus keywords; severity **critical / urgent**; dashboard **108** and SOS tab.
- **Portal mapping** — ANMOL/RCH, U-WIN (NIP schedule), NCD/NPCDCS-style prefill JSON per patient.
- **Risk triage** — red / amber / green with factors.
- **OPD prescription slip** — `GET /patient/{id}/prescription` returns **print-ready HTML**; dashboard and detail views open it for the ward boy / patient (`?autoprint=1` supported).
- **Optional notifications (Twilio)** — **visit-day SMS** when `follow_up_date` is today (IST) and `phone` is set; **end-of-day WhatsApp register** text to the ANM; manual buttons on the dashboard plus an **hourly** background check for SMS.

## OPD session flow (summary)

1. ANM opens **`/ui/`**, starts **Start Session** (Vapi loads session prompt from **`/api/session-prompt`**).
2. Assistant plays the **welcome + language** line; ANM chooses language, then dictates **patient 1** (ID first helps disambiguation).
3. Optional: short **gap** question after pause if vitals/clinical gaps are detected.
4. ANM says **next** → browser posts that segment to **`/simulate-call`** → patient stored → dashboard refreshes; **Print OPD** on the card opens the slip.
5. Repeat for more patients; **End Session** or **session end** closes the call and saves any in-progress patient.

## Architecture

```text
Browser (Vapi SDK)  →  FastAPI  →  Gemini or Ollama (extraction + embeddings when configured)
       ↓                    ↓
  Transcript chunks    Risk engine → Portal mapper
       ↓                    ↓
  /simulate-call       Qdrant (Docker :6333, cloud, or in-memory fallback)
       ↓                    ↓
  Dashboard + OPD HTML     Optional: Twilio SMS / WhatsApp
```

## Project Layout

```
.
├── backend/
│   ├── main.py                    # FastAPI: routes, Vapi webhook, prompts, prescription HTML
│   ├── requirements.txt
│   ├── llm/
│   │   └── extractor.py           # Structured extraction (40+ fields)
│   ├── services/
│   │   ├── risk_engine.py         # Risk + emergency detection
│   │   ├── portal_mapper.py       # ANMOL/RCH, U-WIN, NCD mapping
│   │   ├── gap_prompt.py          # Rule-based missing-field line for session pause
│   │   └── notifications.py       # Twilio SMS / WhatsApp + reminder worker
│   └── db/
│       └── qdrant_client.py       # Qdrant client, embed, store, search, patch
├── frontend/
│   └── index.html                 # SPA: session UI, dashboard, print OPD, notifications buttons
├── docker-compose.yml             # Local Qdrant (ports 6333, 6334)
├── Dockerfile                     # Render / container image
├── render.yaml                    # Render Blueprint
├── screenshots/                   # UI screenshots (demo / docs)
└── .env.example                   # Environment template
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Health check |
| `GET` | `/health/llm` | LLM backend, keys, embedding mode |
| `GET` | `/patients` | All patients |
| `GET` | `/risk-flags` | Red-risk patients |
| `GET` | `/emergencies` | Emergency cases |
| `GET` | `/analytics` | Dashboard analytics |
| `GET` | `/search` | Semantic search (`q`, `limit`) |
| `GET` | `/patients/by-type/{visit_type}` | Filter by visit type |
| `GET` | `/portal-prefill/{id}` | ANMOL, U-WIN, NCD JSON for a patient |
| `GET` | `/patient/{id}/prescription` | **Printable OPD slip** (HTML; `?autoprint=1`) |
| `GET` | `/api/vapi-client-config` | Vapi public key + assistant id for browser |
| `GET` | `/api/system-prompt` | Legacy **single-visit** assistant prompt + first message |
| `GET` | `/api/session-prompt` | **OPD session** prompt + first message (language → multi-patient) |
| `POST` | `/api/session-gap-prompt` | `{ "transcript": "..." }` → `{ "say": "..." \| null }` |
| `GET` | `/api/notifications/status` | Twilio / WhatsApp env readiness |
| `POST` | `/api/notifications/follow-up-reminders/run` | Run visit-day SMS scan (IST) |
| `GET` | `/api/notifications/daily-register/preview` | JSON preview of today’s register text |
| `POST` | `/api/notifications/daily-register/whatsapp` | Send daily register via WhatsApp |
| `POST` | `/simulate-call` | Process transcript → extract → risk → **store** |
| `POST` | `/vapi-webhook` | Vapi server webhook (transcript on call end) |
| `POST` | `/process-visit` | Extract + risk without storing |
| `POST` | `/store-patient` | Store JSON patient |
| `POST` | `/seed-demo` | Load 10 demo patients |
| `GET` | `/ui/` | Injected SPA (Vapi keys from env) |

## Quick Start (Local)

```bash
cd backend
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

**Optional — persistent Qdrant** (recommended so data survives restarts):

```bash
# From repo root, with Docker running
docker compose up -d        # http://127.0.0.1:6333
```

Copy **`.env.example`** to **`.env`** in the repo root (or `backend/`) and set at least:

- `VAPI_PUBLIC_KEY`, `VAPI_ASSISTANT_ID` — for voice from `/ui/`
- `GEMINI_API_KEY` (or rely on **Ollama** locally with `SAATHI_LLM=ollama`)

```bash
cp .env.example .env
```

Run the API:

```bash
cd backend
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Open **`http://localhost:8000/ui/`** (trailing slash matters for the static route).

## Deploy on Render

1. Push this repo to GitHub.
2. Create a **Web Service** on Render (Dockerfile at repo root, or native Python with root `backend` and `uvicorn main:app --host 0.0.0.0 --port $PORT`).
3. Set environment variables (see table below). **`QDRANT_URL`** can be **Qdrant Cloud** HTTPS + `QDRANT_API_KEY`, or a reachable Qdrant instance.
4. Vapi **webhook** (if you use server-side call end instead of only browser `simulate-call`):

   ```
   https://<your-service>.onrender.com/vapi-webhook
   ```

5. **Assistant text in Vapi dashboard** — keep in sync with what the app sends, or rely on overrides:
   - Session UI: fetches **`/api/session-prompt`** and passes **assistantOverrides** (system + first message).
   - For tests inside Vapi only, paste the same prompt from that endpoint into the assistant.

## Vapi Assistant Setup

### Session mode (OPD — default in `/ui/`)

- **`GET /api/session-prompt`** — language selection (1–5), then multi-patient silent scribe; boundary words **next / agla / aage / …**; session end phrases.
- First message includes the **welcome + language** line.
- The browser still POSTs transcripts to **`/simulate-call`** per patient boundary (and on session end).

### Legacy single-visit prompt

- **`GET /api/system-prompt`** — older guided flow (language → one visit recap). Kept for reference or alternate clients.

### Browser SDK

Keys are injected into `index.html` server-side. The app also calls **`/api/vapi-client-config`** as a fallback.

## Optional: Twilio (SMS + WhatsApp)

| Variable | Purpose |
|----------|---------|
| `TWILIO_ACCOUNT_SID` | Twilio account |
| `TWILIO_AUTH_TOKEN` | Twilio auth token |
| `TWILIO_SMS_FROM` | E.164 sender for **SMS** (e.g. `+1…`) |
| `TWILIO_WHATSAPP_FROM` | e.g. `whatsapp:+14155238886` (sandbox or approved) |
| `SAATHI_ANM_WHATSAPP_TO` | e.g. `whatsapp:+9198xxxxxxxx` for **daily register** |

Visit-day SMS runs when **`follow_up_date`** equals **today (IST)**, patient has **`phone`**, and Twilio SMS is configured. After send, **`follow_up_reminder_sent_for`** is stored on the patient record to avoid duplicates.

## Emergency Detection

The risk engine (`backend/services/risk_engine.py`) auto-detects:

| Condition | Severity |
|-----------|----------|
| BP ≥ 180 systolic | Critical |
| BP ≥ 160/110 (pre-eclampsia range) | Critical |
| Hb < 5 g/dL | Critical |
| Temperature ≥ 104°F | Critical |
| SpO2 < 90% | Critical |
| Blood sugar > 500 mg/dL | Critical |
| Pulse < 40 or > 150 bpm | Critical |
| Emergency keywords (convulsions, bleeding, chest pain, unconsciousness) | Critical |
| Pregnancy danger signs (headache, blurred vision, swelling, reduced fetal movement) | Critical |
| BP ≥ 160 systolic | Urgent |
| Hb < 7 g/dL | Urgent |
| Temperature ≥ 102°F | Urgent |
| SpO2 < 94% | Urgent |
| Blood sugar > 300 mg/dL | Urgent |

**Critical** → "IMMEDIATE: Call 108/ambulance. Refer to nearest PHC/CHC/District Hospital."  
**Urgent** → "URGENT: Refer to PHC within 2 hours. Monitor vitals."

## Portal Mapping

### ANMOL/RCH (Reproductive Child Health)

ANC registration (LMP, EDD, gravida/para, gestational weeks), visit records (BP, Hb, urine albumin/sugar, TT doses, IFA/calcium tablets), referral tracking, danger signs counseling.

### U-WIN (Universal Immunization)

Vaccination sessions, immunization card tracking per India's National Immunization Program schedule (BCG → OPV → Pentavalent → MR → DPT → Td), age-based auto-inference of due vaccines.

### NCD/NPCDCS (Non-Communicable Disease Screening)

CBAC-style screening, vitals with BMI calculation and category, hypertension/diabetes status classification, cancer screening fields, lab results, medicines, counseling, referral.

## Environment Variables (Full Reference)

| Variable | Purpose |
|----------|---------|
| `PORT` | Render sets this; bind `$PORT` in production |
| `QDRANT_URL` | `http://127.0.0.1:6333` (Docker), Qdrant Cloud HTTPS, or `:memory:` |
| `QDRANT_API_KEY` | Qdrant Cloud only; omit for local Docker |
| `GEMINI_API_KEY` / `GOOGLE_API_KEY` | Prefer Gemini for extraction + embeddings |
| `GEMINI_MODEL` | Default `gemini-2.5-flash` |
| `SAATHI_LLM` | `gemini` or `ollama` to force one backend |
| `OLLAMA_BASE_URL` | Default `http://127.0.0.1:11434` |
| `VAPI_PUBLIC_KEY` | Browser SDK |
| `VAPI_ASSISTANT_ID` | Assistant id for `vapiSDK.run` |
| `TWILIO_*`, `SAATHI_ANM_WHATSAPP_TO` | Optional; see **Optional: Twilio** above |

## Local Development (Optional Services)

### Qdrant (Docker)

```bash
docker compose up -d        # REST: localhost:6333
docker compose down         # stop
```

If Qdrant is unreachable, the app **falls back to in-memory** storage (data is lost when the process exits).

### Ollama (Local LLM)

```bash
ollama pull qwen3:8b
# Daemon at http://localhost:11434
```

Used when Gemini is not configured or `SAATHI_LLM=ollama`.

## Screenshots

The **`screenshots/`** folder contains UI captures for demos and documentation (dashboard, history, patient detail, SOS).
