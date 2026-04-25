# SAATHI Demo Pitch — Speaker Script (15 slides)

## Slide 1 — Title
- SAATHI is a voice-first assistant for ANMs/ASHAs.
- Goal: convert natural conversation into a structured care record fast, in the OPD workflow.
- In this demo: Vapi (voice) + Gemini (extraction/embeddings) + Qdrant (patient memory) + FastAPI + web UI + Twilio notifications.

## Slide 2 — The problem
- Frontline workers capture many fields per visit (vitals, symptoms, medicines, vaccines, referral, follow-up).
- Real-world constraints: mixed language (Hindi/English), interruptions, and time pressure.
- Current systems break when critical fields are missed and follow-ups are hard to track.

## Slide 3 — Our approach
- End-to-end pipeline:
  1) Speak naturally → 2) Gemini extracts strict JSON → 3) Store/search in Qdrant → 4) Act via dashboard + alerts.
- Designed for speed, language-first UX, safety triage, and automation.

## Slide 4 — Demo flow
- Start session in the web UI → pick language → dictate patient details → say “next” to save.
- Dashboard updates instantly; emergencies/risk flags appear automatically.
- Then we trigger visit-day SMS reminders and the per-ANM daily WhatsApp register.

## Slide 5 — Vapi: voice layer
- Vapi runs the voice experience in the browser (Vapi Web SDK).
- We use a session prompt: language selection first, then “silent scribe” mode.
- Transcripts go to the backend via `/simulate-call` (per patient) and optionally `/vapi-webhook` (call end).

## Slide 6 — Gemini: structured extraction
- Gemini converts messy transcripts into valid JSON with a fixed schema (40+ fields).
- We keep temperature low and request JSON output (responseMimeType) to make parsing reliable.
- This is where the Gemini key is used “effectively”: predictable structure, easier validation, safer downstream rules.

## Slide 7 — Risk triage + emergency detection
- AI extracts values; rules evaluate thresholds and keywords.
- Output is simple and actionable: red/amber/green + an emergency banner with “call 108 / refer urgently”.
- This reduces “black box” risk for critical decisions.

## Slide 8 — Qdrant: patient memory
- Qdrant stores the latest patient snapshot + visit history + embedding vectors.
- Enables semantic search like “pregnant headache” or “diabetes follow-up”.
- In production we use Qdrant Cloud; local Docker is only for development.

## Slide 9 — Backend (FastAPI)
- Backend orchestrates everything:
  - prompts (`/api/session-prompt`)
  - ingestion (`/simulate-call`)
  - patient APIs (`/patients`, `/risk-flags`, `/emergencies`)
  - printable OPD slip (`/patient/{id}/prescription`)
  - notifications and registers

## Slide 10 — Frontend (single-page web UI)
- Mobile-first dashboard built with plain HTML + JS for speed and deploy simplicity.
- Shows KPIs, patient cards, emergency routing, patient detail view, and portal payload previews.
- Also has “Run visit-day SMS” and “Daily register” actions.

## Slide 11 — How a visit becomes a record
- Tool responsibilities are cleanly separated:
  - Vapi = capture
  - Gemini = extract
  - risk engine = safety decisions
  - Qdrant = persistence + memory
  - UI = workflow + visibility
  - Twilio = notifications

## Slide 12 — Notifications (SMS + WhatsApp)
- Patients: SMS on the scheduled day (`follow_up_date == today IST`) and we store `follow_up_reminder_sent_for` to avoid duplicates.
- ANMs: per-ANM daily WhatsApp register—patients are tagged by `registered_by_anm_whatsapp`, then grouped and sent.

## Slide 13 — Why this architecture works
- Reliable in the field: language-first + minimal interruptions.
- Safety-first: deterministic thresholds + explainable signals.
- Fast iteration: prompts/rules live server-side; deploy quickly.
- Auditable: transcript → JSON → stored record → notifications logs.

## Slide 14 — Roadmap
- Add per-facility analytics, offline-first capture, stronger identity/RBAC, and WhatsApp templates/localization for scale.

## Slide 15 — Close / ask
- Next step: a 2-week pilot OPD shift.
- Measure time saved, data completeness, and follow-up/referral outcomes.

