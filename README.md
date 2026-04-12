# SAATHI — Smart ANM Assistant for Total Health Intelligence

Voice-first healthcare MVP: ANM workers speak patient details; the system extracts structured data, stores history, maps to ANMOL / U-WIN / NCD-style forms, and surfaces a mobile dashboard.

### Production-style pipeline (hackathon / Render)

```text
Vapi  →  Render (FastAPI)  →  Gemini API  →  Qdrant Cloud
```

1. **Vapi** ends a call and `POST`s the transcript to `https://<your-app>.onrender.com/vapi-webhook`.
2. **Render** runs this repo’s FastAPI service (`$PORT`, `0.0.0.0`).
3. **Gemini** runs structured JSON extraction when `GEMINI_API_KEY` or `GOOGLE_API_KEY` is set (see below). Without a key, the app uses **Ollama** at `OLLAMA_BASE_URL` (typical for local dev only).
4. **Qdrant Cloud** (or any reachable Qdrant) is configured with **`QDRANT_URL`** so visits persist after deploy.

## Project layout

```
.
├── backend/           # FastAPI application
│   ├── main.py
│   └── requirements.txt
├── frontend/          # Mobile-first PWA UI (vanilla JS + Tailwind)
├── data/              # Local data (e.g. Qdrant volume mount)
└── docker-compose.yml # Local Qdrant
```

## Prerequisites

- Python 3.11+ recommended
- **Local:** [Docker](https://docs.docker.com/get-docker/) (optional, for local Qdrant) and [Ollama](https://ollama.com/) if you are **not** using Gemini
- **Render:** [Google AI Studio](https://aistudio.google.com/apikey) API key for **Gemini**, and a **Qdrant Cloud** cluster URL for **`QDRANT_URL`**

## Backend setup

```bash
cd backend
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Run FastAPI (local)

```bash
cd backend
source .venv/bin/activate
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

API base URL: `http://localhost:8000`  
Dashboard UI (same server): `http://localhost:8000/ui/`

### Deploy on Render (stable URL for Vapi — no ngrok)

1. Push this repo to GitHub.
2. **Choose a runtime on Render**
   - **Docker (recommended if you picked “Docker” before):** leave the **root** as the repo root; Render will find **`Dockerfile`**. No `rootDir: backend` in the service settings.
   - **Native Python:** set **Root directory** to `backend`, **Build:** `pip install -r requirements.txt`, **Start:** `uvicorn main:app --host 0.0.0.0 --port $PORT` (`PORT` is set by Render).
3. **Environment** on the service — set at least:
   - **`QDRANT_URL`** — cluster HTTPS URL from [Qdrant Cloud](https://cloud.qdrant.io/) (same host you use in `curl`; often ends with `:6333` for REST).
   - **`QDRANT_API_KEY`** — **Database API key** from the cluster (create under **API Keys** on the cluster detail page). Qdrant expects this on every request as header `api-key` or `Authorization: Bearer …`; the Python client sends it when you pass `api_key=` (see [Authentication](https://qdrant.tech/documentation/cloud/authentication/)).
   - **`GEMINI_API_KEY`** (or **`GOOGLE_API_KEY`**) — so extraction uses **Gemini** on the public internet (recommended on Render).
   - Optional **`GEMINI_MODEL`** — default is `gemini-2.0-flash` (override if your project uses another model name).
   - **`SAATHI_LLM=ollama`** — only if you intentionally want Ollama instead of Gemini when both could be configured.
   - **`OLLAMA_BASE_URL`** — only for remote Ollama; not required when Gemini is configured.
4. After deploy, set Vapi’s webhook to:

   `https://<your-service>.onrender.com/vapi-webhook`

5. **Logs:** Render dashboard → your service → **Logs**. Search for `WEBHOOK_HIT`, `pipeline.`, and `http.in` / `http.out`.

## Qdrant Cloud (Render / Vapi backend)

Official docs: **[Database Authentication](https://qdrant.tech/documentation/cloud/authentication/)** · Python client: **`QdrantClient(host, api_key=...)`** or **`QdrantClient(url=..., api_key=...)`** ([interfaces](https://qdrant.tech/documentation/interfaces/)).

1. In [Qdrant Cloud](https://cloud.qdrant.io/), open your cluster → copy the **REST endpoint** (HTTPS, often port **6333**).
2. **Cluster → API Keys → Create** — copy the key once; SAATHI reads it from **`QDRANT_API_KEY`** (do not commit it to git).
3. On Render, set **`QDRANT_URL`** and **`QDRANT_API_KEY`**. Boot logs show `QDRANT_API_KEY=set` when present.

Quick connectivity test (from Qdrant docs):

```bash
curl -sS -X GET "$QDRANT_URL" --header "api-key: $QDRANT_API_KEY"
```

You should see JSON with `"title":"qdrant - vector search engine"` and a `version` field.

## Qdrant (Docker, local only)

From the project root:

```bash
docker compose up -d
```

- REST API: `http://localhost:6333` (no API key)
- Dashboard (if enabled in image): `http://localhost:6334`

Stop services:

```bash
docker compose down
```

## Ollama (local LLM)

1. Install Ollama from [ollama.com](https://ollama.com/).
2. Start the Ollama app (daemon).
3. Pull the model used in later tasks:

```bash
ollama pull qwen3:8b
```

Default API: `http://localhost:11434`

## Frontend (static preview)

Open `frontend/index.html` in a browser, or serve the folder:

```bash
cd frontend && python -m http.server 3000
```

Then visit `http://localhost:3000` (UI only until Task 9 connects the API).

## Environment variables (summary)

| Variable | Purpose |
|----------|---------|
| `PORT` | Set by Render on deploy; use `$PORT` in the start command. |
| `QDRANT_URL` | Qdrant REST URL (Qdrant Cloud HTTPS, or local `http://127.0.0.1:6333`, or `:memory:`). |
| `QDRANT_API_KEY` | [Qdrant Cloud Database API key](https://qdrant.tech/documentation/cloud/authentication/); omit for local Docker. |
| `GEMINI_API_KEY` / `GOOGLE_API_KEY` | If set, extraction uses **Gemini** instead of Ollama. |
| `GEMINI_MODEL` | Gemini model id (default `gemini-2.0-flash`). |
| `SAATHI_LLM` | `gemini` or `ollama` to force one backend when debugging. |
| `OLLAMA_BASE_URL` | Ollama base (default `http://127.0.0.1:11434`). Used when no Gemini key (or `SAATHI_LLM=ollama`). |
