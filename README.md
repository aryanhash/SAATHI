# SAATHI — Smart ANM Assistant for Total Health Intelligence

Voice-first healthcare assistant for India's frontline health workers (ANM/ASHA). Workers speak patient visit details in their preferred language; SAATHI extracts structured data via AI, detects emergencies, stores history in a vector DB, maps to real government portal forms (ANMOL/RCH, U-WIN, NCD), and surfaces a mobile dashboard with emergency routing.

## Key Features

- **Multi-language voice** — asks preferred language at call start; supports Hindi, English, Tamil, Telugu, Bengali, Kannada, Marathi, Gujarati, Malayalam, Odia, Punjabi, Assamese
- **40+ field AI extraction** — Gemini extracts patient name, vitals (BP, Hb, SpO2, pulse, temperature), symptoms, medicines, vaccines, referral details, emergency signs, and more from natural conversation
- **Emergency detection & routing** — auto-detects critical conditions (hypertensive crisis, severe anemia, hypoxia, convulsions, pregnancy danger signs) with severity levels (critical/urgent) and one-tap 108 ambulance calling
- **Real government portal mapping** — ANMOL/RCH (ANC registration, visit records), U-WIN (immunization schedule per India NIP), NCD/NPCDCS (CBAC screening, hypertension/diabetes classification)
- **Rural + Urban** — works for village sub-centres and tier-1 city UPHCs alike
- **Risk triage** — automated red/amber/green risk classification with risk factors
- **Analytics dashboard** — patient counts, risk distribution, visit type breakdown, emergency/referral tracking

## Architecture

```text
Vapi (voice)  →  Render (FastAPI)  →  Gemini API (extraction)  →  Qdrant Cloud (storage)
     ↑                  ↓
Browser SDK      Emergency detection → Risk triage → Portal mapping
(12 languages)        ↓                                    ↓
                  Dashboard UI                    ANMOL / U-WIN / NCD
```

## Project Layout

```
.
├── backend/
│   ├── main.py                    # FastAPI app, routes, Vapi webhook, system prompt
│   ├── requirements.txt
│   ├── llm/
│   │   └── extractor.py           # Gemini/Ollama structured extraction (40+ fields)
│   ├── services/
│   │   ├── risk_engine.py         # Risk triage + emergency detection
│   │   └── portal_mapper.py       # ANMOL/RCH, U-WIN, NCD portal mapping
│   └── db/
│       └── qdrant_client.py       # Qdrant persistence (cloud or in-memory fallback)
├── frontend/
│   └── index.html                 # Mobile-first SPA (vanilla JS + Tailwind)
├── Dockerfile                     # Docker image for Render
├── render.yaml                    # Render Blueprint
└── .env.example                   # Environment variable template
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Health check |
| `GET` | `/health/llm` | LLM backend status, key presence |
| `GET` | `/patients` | All patients from Qdrant |
| `GET` | `/risk-flags` | Red-risk patients only |
| `GET` | `/emergencies` | Active emergency cases (critical + urgent) |
| `GET` | `/analytics` | Dashboard analytics (counts, distributions) |
| `GET` | `/portal-prefill/:id` | ANMOL/RCH, U-WIN, NCD pre-filled forms for a patient |
| `GET` | `/api/vapi-client-config` | Vapi SDK keys for browser |
| `GET` | `/api/system-prompt` | Multi-language system prompt for Vapi assistant |
| `POST` | `/simulate-call` | Process a transcript (same pipeline as webhook) |
| `POST` | `/vapi-webhook` | Vapi webhook endpoint (call-ended transcripts) |
| `POST` | `/process-visit` | Extract + risk-triage without storing |
| `POST` | `/store-patient` | Direct patient storage |
| `POST` | `/seed-demo` | Insert 10 realistic demo patients |
| `GET` | `/ui/` | Frontend SPA with injected Vapi keys |

## Quick Start (Local)

```bash
cd backend
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Copy `.env.example` to `.env` in the repo root and fill in your keys:

```bash
cp .env.example .env
# Edit .env with your VAPI_PUBLIC_KEY, VAPI_ASSISTANT_ID, GEMINI_API_KEY, etc.
```

Run the server:

```bash
cd backend
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Open **`http://localhost:8000/ui/`** (trailing slash required).

## Deploy on Render

1. Push this repo to GitHub.
2. Create a **Web Service** on Render:
   - **Docker** runtime (recommended) — Render finds the `Dockerfile` at repo root.
   - Or **Native Python**: root dir `backend`, build `pip install -r requirements.txt`, start `uvicorn main:app --host 0.0.0.0 --port $PORT`.
3. Set environment variables on Render:

| Variable | Required | Description |
|----------|----------|-------------|
| `GEMINI_API_KEY` | Yes | [Google AI Studio](https://aistudio.google.com/apikey) API key |
| `QDRANT_URL` | Yes | Qdrant Cloud HTTPS endpoint |
| `QDRANT_API_KEY` | Yes | Qdrant Cloud database API key |
| `VAPI_PUBLIC_KEY` | Yes | Vapi public key (browser SDK) |
| `VAPI_ASSISTANT_ID` | Yes | Your Vapi assistant ID |
| `GEMINI_MODEL` | No | Default `gemini-2.5-flash` |
| `SAATHI_LLM` | No | Force `gemini` or `ollama` |
| `OLLAMA_BASE_URL` | No | Only for remote Ollama |

4. Set your Vapi assistant's **webhook URL** to:
   ```
   https://<your-service>.onrender.com/vapi-webhook
   ```

5. Set your Vapi assistant's **System Prompt** — fetch the latest from:
   ```
   https://<your-service>.onrender.com/api/system-prompt
   ```

## Vapi Assistant Setup

### System Prompt (Multi-Language)

The system prompt is served at `GET /api/system-prompt`. It instructs the assistant to:

1. **Ask language preference first** — supports 12 Indian languages
2. Use gender-neutral, respectful addressing ("aap")
3. Capture visit data through natural conversation (one question at a time)
4. Detect emergency signs and suggest 108 ambulance / PHC referral
5. Support both rural (village, sub-centre) and urban (UPHC, polyclinic) contexts
6. Recap as structured bullets and call `send_transcript` when confirmed

### Browser SDK

The frontend loads the Vapi HTML Script Tag SDK. Keys are injected server-side into meta tags (never committed to git). The browser also fetches `/api/vapi-client-config` as a fallback.

When a call ends, the frontend captures the transcript client-side and POSTs it to `/simulate-call` — this ensures data saves even without a public webhook (useful for local dev).

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
| `PORT` | Set by Render; use `$PORT` in start command |
| `QDRANT_URL` | Qdrant REST URL (cloud HTTPS, local `http://127.0.0.1:6333`, or `:memory:`) |
| `QDRANT_API_KEY` | Qdrant Cloud database API key; omit for local Docker |
| `GEMINI_API_KEY` / `GOOGLE_API_KEY` | If set, extraction uses Gemini instead of Ollama |
| `GEMINI_MODEL` | Gemini model id (default `gemini-2.5-flash`) |
| `SAATHI_LLM` | `gemini` or `ollama` to force one backend |
| `OLLAMA_BASE_URL` | Ollama base URL (default `http://127.0.0.1:11434`) |
| `VAPI_PUBLIC_KEY` | Vapi public key for browser SDK |
| `VAPI_ASSISTANT_ID` | Assistant id for `vapi.start(...)` |

## Local Development (Optional Services)

### Qdrant (Docker)

```bash
docker compose up -d        # REST: localhost:6333
docker compose down          # stop
```

Not required — the app falls back to in-memory Qdrant if the remote connection fails.

### Ollama (Local LLM)

```bash
ollama pull qwen3:8b         # one-time
# Ollama daemon runs at http://localhost:11434
```

Only needed if not using Gemini (`SAATHI_LLM=ollama` or no Gemini key set).
