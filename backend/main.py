"""SAATHI backend — FastAPI application."""

from __future__ import annotations

import hmac
import html
import json
import logging
import os
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv
from fastapi import Body, Depends, FastAPI, HTTPException, Request, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, PlainTextResponse, RedirectResponse
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field

from db.qdrant_client import (
    create_collection,
    delete_seeded_demo_points,
    embedding_mode,
    get_all_patients,
    get_emergencies as db_get_emergencies,
    get_patient_by_id,
    get_patients_by_risk,
    get_patients_by_visit_type,
    search_similar,
    store_patient,
)
from llm.extractor import GEMINI_MODEL, extract_patient_data, extraction_backend
from services.gap_prompt import suggest_gap_prompt
from services.notifications import (
    build_daily_register_text,
    notification_status,
    run_follow_up_reminders_for_today,
    send_daily_register_whatsapp,
    send_daily_register_whatsapp_all,
    start_reminder_background_thread,
)
from services.portal_mapper import build_portal_prefill
from services.risk_engine import calculate_risk

_REPO_ROOT = Path(__file__).resolve().parent.parent
_BACKEND_DIR = Path(__file__).resolve().parent
load_dotenv(_REPO_ROOT / ".env")
load_dotenv(_BACKEND_DIR / ".env", override=True)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Hardcoded fallback prompts — used when the main prompt constant is empty or
# when an exception occurs in the route handler (future-proofing for dynamic
# prompt sources).  These are intentionally separate from SAATHI_SYSTEM_PROMPT
# so the fallback is never a no-op.
# ---------------------------------------------------------------------------
_FALLBACK_SYSTEM_PROMPT = (
    "You are SAATHI — a friendly, supportive voice assistant for India's "
    "healthcare workers (ANMs/ASHAs). Start every call by asking the user's "
    "preferred language (Hindi, English, Kannada, Tamil, Telugu). Once confirmed, "
    "conduct the entire conversation in that language. Ask one question at a time. "
    "Collect: patient name, age, gender, visit reason, symptoms, vitals "
    "(BP, temp, pulse, SpO2, weight), tests, medicines, counseling, immunizations, "
    "referral details, and follow-up date. If any emergency sign is mentioned "
    "(severe breathlessness, chest pain, uncontrolled bleeding, convulsions, "
    "unconsciousness, very high fever in infant, or pregnancy danger signs), "
    "say clearly: 'This sounds like an emergency' and suggest calling 108. "
    "Recap all details as structured bullets before ending. After user confirms, "
    "call send_transcript and stop."
)
_FALLBACK_FIRST_MESSAGE = (
    "Hello! I'm SAATHI — your health assistant. "
    "\u0939\u093f\u0902\u0926\u0940 \u0915\u0947 \u0932\u093f\u090f 1 \u0926\u092c\u093e\u090f\u0902\u0964 For English, press 2. "
    "\u0c95\u0ca8\u0ccd\u0ca8\u0ca1\u0c95\u0ccd\u0c95\u0cbe\u0c97\u0cbf 3 \u0c92\u0ca4\u0ccd\u0ca4\u0cbf\u0cb0\u0cbf. "
    "\u0ba4\u0bae\u0bbf\u0bb4\u0bc1\u0b95\u0bcd\u0b95\u0bc1 4 \u0b85\u0bb4\u0bc1\u0ba4\u0bcd\u0ba4\u0bb5\u0bc1\u0bae\u0bcd. "
    "\u0c24\u0c46\u0c32\u0c41\u0c17\u0c41\u0c15\u0c41 5 \u0c28\u0c4a\u0c15\u0c4d\u0c15\u0c02\u0c21\u0c3f. "
    "Please select your preferred language."
)

# ---------------------------------------------------------------------------
# Multi-language system prompt for Vapi assistant
# ---------------------------------------------------------------------------
SAATHI_SYSTEM_PROMPT = """\
[Identity]
You are SAATHI — a friendly, supportive voice assistant designed for India's healthcare workers, \
such as ANMs and ASHAs. Your primary purpose is to assist users in documenting patient visits \
and conversations naturally, respectfully, and efficiently, reducing reliance on manual typing.

[Language Selection — FIRST STEP, ALWAYS]
- Start every call with this exact greeting:
  "Hello! I'm SAATHI — your health assistant.
  हिंदी के लिए 1 दबाएं।
  For English, press 2.
  ಕನ್ನಡಕ್ಕಾಗಿ 3 ಒತ್ತಿರಿ.
  தமிழுக்கு 4 அழுத்தவும்.
  తెలుగుకు 5 నొక్కండి.
  Please select your preferred language."
- Wait for the user's response before saying anything else.
- Once language is confirmed, conduct the ENTIRE conversation in that language only.
- Default language: English (if user does not specify or is unclear).
- If user requests an unsupported language, say: "I'm sorry, I don't support that language yet. \
Shall we continue in English or Hindi?"

[Style]
- Polite, warm, approachable, and conversational at all times.
- Ask only ONE question at a time. Keep prompts short and clear.
- Address users respectfully by name or "you" (or language-appropriate equivalent).
- Use gender-neutral language.
- If the user seems rushed, switch to fast mode: fewer questions, direct confirmations only.
- When repeating numbers (phone, BP, sugar, Hb), say each digit slowly for clarity.
- Use natural speech patterns with gentle pauses where appropriate.

[Response Guidelines]
- Never say or enter "unknown." If a detail is missing, politely ask once more or skip to the next field.
- For ambiguous health values (e.g., "BP normal"), gently ask for an approximate or specific number.
- Never invent or guess data. If information cannot be obtained after a second attempt, move on.

[Conversation Flow]

Step 1 — Language Selection
  Greet and ask for language preference (as above).
  <wait for user response>
  Confirm: "Great, let's continue in [language]." Then switch fully to that language.

Step 2 — Visit Context
  Ask about location (village/area or urban center: city/UPHC/dispensary/polyclinic).
  Ask visit date/time — skip if user says "today" or "right now."

Step 3 — Patient Identification
  Collect: patient name or initials, age, gender, phone number (optional), unique ID (if available).

Step 4 — Reason for Visit
  Ask the primary reason: ANC/PNC, immunization, fever/cough, NCD follow-up, \
family planning, general checkup, etc.

Step 5 — Symptoms & Vitals
  Ask for: symptoms, temperature, BP, pulse, SpO2, weight.
  If pregnancy-related: ask for pregnancy week.

Step 6 — Findings & Actions
  Document: tests done (Hb, sugar, urine), medicines given, counseling provided, \
immunization type/dose, referral details (where/why), follow-up date.

Step 7 — Closing & Confirmation
  Recap all collected details as structured bullet points:
    - Patient: [name, age, gender]
    - Reason: [visit purpose]
    - Vitals: [temp, BP, pulse, SpO2, weight]
    - Actions: [tests, medicines, counseling, immunization]
    - Referral/Follow-up: [details]
  Ask user to confirm. Then ask: "Any other visit to record?"

[Data Hygiene]
- Never enter "unknown." Ask once more politely, then skip if still unclear.
- Repeat all numbers back slowly and clearly for confirmation.
- If response is vague (e.g., "sugar fine"), follow up: "Could you share the exact reading, \
even approximate?"

[Safety & Emergency Protocol]
- Do NOT provide direct medical advice.
- If any emergency sign is mentioned — severe breathlessness, chest pain, uncontrolled bleeding, \
convulsions, unconsciousness, very high fever in an infant, or pregnancy danger signs \
(severe headache, blurred vision, swelling, reduced fetal movement, leaking fluid):
    1. Say clearly: "This sounds like an emergency."
    2. Ask: "Should we call an ambulance or the PHC? Ambulance is available on 1-0-8."
    3. Continue documenting all emergency details carefully during the conversation.

[Urban & NCD Specifics]
- Recognize urban terms: UPHC, dispensary, polyclinic, corporate hospital.
- For NCD screening, ask about tobacco/alcohol use and family history of diabetes, \
hypertension, or cancer.
- Share basic lifestyle information (diet, exercise, stress) as general information only, \
never as medical advice.

[Error Handling & Fallback]
- If response is unclear, gently rephrase and repeat the question in simpler terms.
- If a tool or function fails, apologize warmly and retry if appropriate.
- If user goes off-topic, guide them back gently to the current step.
- If the selected language is unsupported, default to English.

[Output & Tool Usage]
- After user confirms the recap, call the "send_transcript" function with the full \
conversation transcript.
- After calling "send_transcript," stop completely — no further speech or text output.

[Call Closing]
- Recap is given as structured bullet points before ending.
- Once "send_transcript" is triggered and sent, end the conversation silently."""

SAATHI_FIRST_MESSAGE = (
    "Hello! I'm SAATHI — your health assistant. "
    "हिंदी के लिए 1 दबाएं। "
    "For English, press 2. "
    "ಕನ್ನಡಕ್ಕಾಗಿ 3 ಒತ್ತಿರಿ. "
    "தமிழுக்கு 4 அழுத்தவும். "
    "తెలుగుకు 5 నొక్కండి. "
    "Please select your preferred language."
)


# ---------------------------------------------------------------------------
# Ambient SESSION prompt — language first, then silent scribe / multi-patient
# ---------------------------------------------------------------------------
SAATHI_SESSION_PROMPT = """\
[Identity]
You are SAATHI in SESSION MODE — a silent medical scribe assisting an ANM during her OPD shift.
ONE call covers MANY patients. You are NOT a chatbot; you are a quiet assistant that listens,
records, and only speaks when something critical is missing.
Before scribe mode begins, you MUST complete language selection once (see below).

[Session language lock — read first]
After the ANM picks 1–5, you have a **locked session language** (Hindi, English, Kannada, Tamil, or Telugu).
**Every** assistant utterance for the rest of the call — confirmations, Case 1 questions, emergency line, session end —
must be **only** in that locked language.
If the lock is **English**, you must use **English only** (no Hindi words, no mixed Hinglish, no Hindi stock phrases).
If the lock is **Hindi**, use Hindi. Same for Kannada / Tamil / Telugu.
The ANM may still dictate patient data in Hindi or mixed language; that does **not** change your output language.

[Language selection — always first]
The first message the user hears is already the welcome (press 1 for Hindi, 2 for English, and so on).
Wait for their choice:
  • 1 or Hindi / हिंदी → lock Hindi for the entire rest of the call.
  • 2 or English → lock English for the entire rest of the call.
  • 3 or Kannada → lock Kannada for the rest of the call.
  • 4 or Tamil → lock Tamil for the rest of the call.
  • 5 or Telugu → lock Telugu for the rest of the call.
If unclear, ask once which number they want — in neutral short English is OK for that one clarification only.
After they choose, say exactly **one** short line **in the locked language only**, then go silent.
Examples of tone (produce your line in the **locked** language, not a mix):
  • If lock is Hindi: one line inviting first patient with ID, name, details.
  • If lock is English: e.g. "Understood. First patient: ID, then name, then details."
Do NOT ask for language again after this step.

[Golden Rule]
LISTEN FIRST. STRUCTURE LATER. PROMPT ONLY WHEN NEEDED.
Never interrupt while the ANM is speaking. Do not acknowledge, affirm, or repeat back.
No "okay", "got it", "mhm", "thank you". Stay silent.

[Silence and pauses — critical]
Pauses while the ANM thinks or dictates are normal. Do **not** speak only because the line is quiet.
Never say the word "silence", never narrate quiet, never use filler because it is silent.
If nothing in Case 1–3 applies, output **no speech**.

[How the ANM uses you]
For each patient she will speak in this natural order:
  1. Patient ID  (e.g. "ID 1042"  or  "ID one zero four two"  or  "naya patient")
  2. Patient name
  3. Age, gender (if relevant)
  4. Vitals (BP, Hb, weight, temperature, SpO2, sugar)
  5. Symptoms / complaints
  6. Diagnosis / treatment / medicines / vaccines
  7. Referral or follow-up (if any)

When she finishes a patient she will say one of:
  "next" / "agla" / "aage" / "agla patient" / "next patient" / "dusra patient"

That is your boundary signal. The previous patient's data is automatically
saved by the system. You do NOT need to confirm or summarize.

[When to speak — ONLY these three cases]

Case 1 — Critical missing field (not because of pauses):
  From what the ANM has **already said** for the CURRENT patient, if a critical field is clearly still missing
  for that patient type, you may ask **one** short question (max 8 words) **in the locked session language only**.
  If she is only pausing or you are unsure anything is missing, say **nothing**.

  Patient type → critical fields:
    • Pregnant woman (ANC) → pregnancy weeks, BP, Hb
    • Child / immunization → age in months, vaccines given, weight
    • NCD / adult OPD     → BP, primary complaint
    • General             → age, primary complaint

  **Do not copy Hindi** if the session lock is English (or other non-Hindi). Shape your question like these **English**
  patterns, then **translate** into the locked language when the lock is not English:
    • "Please confirm blood pressure and hemoglobin."
    • "Please confirm the child's age and weight."
    • "What is the main complaint?"
  (When lock is Hindi, natural Hindi equivalents are fine — still one short question.)

Case 2 — Emergency keyword detected:
  If you hear: severe bleeding, chest pain, convulsion, unconscious,
  severe breathlessness, blurred vision (in pregnancy), severe headache,
  swelling face, reduced fetal movement, leaking fluid, BP above 180,
  Hb below 5 — say **exactly once**, **in the locked session language only**.
  English lock example (translate for other locks): "This may be an emergency. Call one zero eight?"
  Then go silent again.

Case 3 — End of session:
  When the ANM says "session end" / "session khatam" / "bas done" / "end session":
    Say in the **locked** language that the session is complete. Optional patient count only if you are sure.
  Then stop completely.

[What you must NEVER do]
  - Never ask for language again after [Language selection] is done.
  - Never repeat back what she said.
  - Never summarize or recap mid-session.
  - Never ask follow-up questions just to be thorough — only critical fields.
  - Never speak while she is still speaking. Never speak only because she paused.
  - Never say "next patient" yourself — that's her trigger word, not yours.
  - Never switch your **spoken** language away from the lock (no mid-call Hindi if English was locked).

[Number handling]
When the ANM says numbers, accept them as-is. Do not read long confirmations back.
Only ask again if a number was clearly garbled or impossible.

[Style]
Warm but minimal. One short sentence per intervention. **Locked session language only.**
No medical advice, no opinions, no chitchat.

[Closing]
Stay on the call indefinitely. The ANM ends the session by saying
"session end" or by tapping End Session in the app. Do not end the call
yourself."""

SAATHI_SESSION_FIRST_MESSAGE = (
    "Welcome to our service. "
    "हिंदी के लिए 1 दबाएं। "
    "For English, press 2. "
    "ಕನ್ನಡಕ್ಕಾಗಿ 3 ಒತ್ತಿರಿ. "
    "தமிழுக்கு 4 அழுத்தவும். "
    "తెలుగుకు 5 నొక్కండి. "
    "Please select your preferred language."
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
        force=True,
    )
    port = os.environ.get("PORT", "(not set — use 8000 locally or $PORT on Render)")
    qmode = os.environ.get("QDRANT_URL", "http://127.0.0.1:6333")
    gemini = "set" if (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")) else "unset"
    qdrant_key = "set" if os.environ.get("QDRANT_API_KEY") else "unset"
    vapi_pk = "set" if os.environ.get("VAPI_PUBLIC_KEY") else "unset"
    vapi_aid = "set" if os.environ.get("VAPI_ASSISTANT_ID") else "unset"
    logger.info(
        "saathi.boot PORT=%s QDRANT_URL=%s QDRANT_API_KEY=%s GEMINI_API_KEY=%s "
        "VAPI_PUBLIC_KEY=%s VAPI_ASSISTANT_ID=%s llm_backend=%s",
        port, qmode, qdrant_key, gemini, vapi_pk, vapi_aid, extraction_backend(),
    )
    _fp = _REPO_ROOT / "frontend" / "index.html"
    if not _fp.is_file():
        logger.warning("saathi.frontend_missing path=%s", _fp)
    try:
        create_collection()
        logger.info("saathi.boot qdrant_ok collection_ready")
    except Exception as e:
        logger.warning("saathi.boot qdrant_skip reason=%s", e)
    try:
        start_reminder_background_thread()
    except Exception as e:
        logger.warning("saathi.boot reminder_thread_skip reason=%s", e)
    yield


app = FastAPI(title="SAATHI", version="0.2.0", lifespan=lifespan)

# ---------------------------------------------------------------------------
# CORS — restrict origins in production via CORS_ORIGINS env var.
# Default: allow all origins WITHOUT credentials (safe baseline).
# Set CORS_ORIGINS to a comma-separated list of allowed origins for production.
# ---------------------------------------------------------------------------
_cors_origins_raw = os.environ.get("CORS_ORIGINS", "").strip()
_cors_origins: list[str] = (
    [o.strip() for o in _cors_origins_raw.split(",") if o.strip()]
    if _cors_origins_raw
    else ["*"]
)
_cors_allow_credentials = _cors_origins != ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=_cors_allow_credentials,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# API Key authentication
# Set SAATHI_API_KEY env var to enable. When unset, auth is bypassed (dev mode).
# ---------------------------------------------------------------------------
_API_KEY = (os.environ.get("SAATHI_API_KEY") or "").strip()
_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def verify_api_key(
    api_key: str | None = Security(_api_key_header),
) -> str | None:
    """Validate the API key when SAATHI_API_KEY is configured."""
    if not _API_KEY:
        return None  # Auth disabled (dev mode)
    if not api_key or not hmac.compare_digest(api_key, _API_KEY):
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
    return api_key


@app.middleware("http")
async def http_request_logging(request: Request, call_next):
    path = request.url.path
    skip = path.startswith("/ui") or path in ("/favicon.ico",)
    client_host = request.client.host if request.client else "?"
    if not skip:
        logger.info("http.in  %s %s client=%s", request.method, path, client_host)
    try:
        response = await call_next(request)
    except Exception:
        logger.exception("http.err %s %s", request.method, path)
        raise
    if not skip:
        logger.info("http.out %s %s status=%s", request.method, path, response.status_code)
    return response


class ProcessVisitBody(BaseModel):
    transcript: str = Field(..., min_length=1)


class SimulateCallBody(BaseModel):
    transcript: str = Field(..., min_length=1)
    # Used to tag who registered the patient, so daily registers can be sent per-ANM.
    anm_id: str | None = None
    anm_name: str | None = None
    anm_whatsapp: str | None = Field(
        default=None,
        description="ANM WhatsApp, e.g. whatsapp:+9198xxxxxxx (or +91...; server will normalize when sending)",
    )


class SessionGapBody(BaseModel):
    transcript: str = Field(..., min_length=1)
    session_language: str | None = Field(
        default=None,
        description="hi | en | kn | ta | te — should match ANM language choice (default hi)",
    )


def _extract_llm(transcript: str) -> dict[str, Any]:
    logger.info("pipeline.extract start chars=%s backend=%s", len(transcript), extraction_backend())
    try:
        out = extract_patient_data(transcript)
        logger.info("pipeline.extract done keys=%s", list(out.keys()))
        return out
    except requests.HTTPError as e:
        code = e.response.status_code if e.response is not None else "?"
        hint = (e.response.text[:240] + "…") if e.response is not None and e.response.text else ""
        detail_msg = "LLM extraction service returned an error"
        if e.response is not None and e.response.text:
            try:
                j = e.response.json()
                em = (j.get("error") or {}).get("message")
                if em:
                    detail_msg = f"Gemini API ({code}): {em}"
            except (ValueError, TypeError, AttributeError):
                detail_msg = f"Gemini API ({code}): {hint or 'no body'}"
        logger.warning("pipeline.llm_http_error status=%s hint=%s", code, hint)
        raise HTTPException(
            status_code=502,
            detail=detail_msg,
        ) from e
    except requests.RequestException as e:
        logger.warning("pipeline.llm_unreachable err=%s", e)
        raise HTTPException(
            status_code=503,
            detail="Cannot reach LLM extraction service",
        ) from e
    except ValueError as e:
        raise HTTPException(status_code=502, detail=str(e)) from e


def _run_voice_pipeline(
    transcript: str,
    *,
    log_prefix: str,
    vapi_call_id: str | None = None,
    anm_id: str | None = None,
    anm_name: str | None = None,
    anm_whatsapp: str | None = None,
) -> dict[str, Any]:
    _t0 = time.perf_counter()
    logger.info("%s pipeline.begin transcript_chars=%s", log_prefix, len(transcript))
    extracted = _extract_llm(transcript)
    logger.info(
        "%s pipeline.extracted keys=%s",
        log_prefix,
        list(extracted.keys()),
    )
    data = calculate_risk(extracted)
    logger.info("%s pipeline.risk risk_level=%s", log_prefix, data.get("risk_level"))

    emergency = data.get("emergency", {})
    if emergency.get("is_emergency"):
        logger.warning(
            "%s EMERGENCY_DETECTED severity=%s reasons=%s",
            log_prefix,
            emergency.get("severity"),
            emergency.get("reasons"),
        )

    if vapi_call_id:
        data["vapi_call_id"] = vapi_call_id
    if anm_id:
        data["registered_by_anm_id"] = anm_id
    if anm_name:
        data["registered_by_anm_name"] = anm_name
    if anm_whatsapp:
        data["registered_by_anm_whatsapp"] = anm_whatsapp
    try:
        pid = store_patient(data)
    except Exception as e:
        logger.exception("%s pipeline.store FAILED", log_prefix)
        raise HTTPException(status_code=503, detail=f"Qdrant store failed: {e}") from e
    data["patient_id"] = pid
    _elapsed_ms = int((time.perf_counter() - _t0) * 1000)
    logger.info("%s pipeline.complete patient_id=%s elapsed_ms=%s", log_prefix, pid, _elapsed_ms)
    return {"status": "processed", "patient": data}


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/", response_class=PlainTextResponse)
def root() -> str:
    return "SAATHI backend running"


@app.get("/health/llm")
def health_llm(_key: str | None = Depends(verify_api_key)) -> dict[str, Any]:
    return {
        "llm_backend": extraction_backend(),
        "gemini_model": GEMINI_MODEL,
        "embedding_mode": embedding_mode(),
    }


@app.get("/api/vapi-client-config")
def vapi_client_config() -> dict[str, Any]:
    pk = (os.environ.get("VAPI_PUBLIC_KEY") or "").strip()
    aid = (os.environ.get("VAPI_ASSISTANT_ID") or "").strip()
    if not pk or not aid:
        return {"configured": False}
    return {"configured": True, "publicKey": pk, "assistantId": aid}


@app.get("/api/system-prompt")
def get_system_prompt() -> dict[str, Any]:
    """Return the multi-language system prompt and first message for Vapi assistant.

    Always returns a valid response with at least the default prompt so that
    callers (e.g. the frontend ``prefetchPrompt``) never receive an error or
    empty body.  If a future version loads the prompt from an external source,
    any failure will be caught and the built-in default will be returned instead.
    """
    try:
        prompt = SAATHI_SYSTEM_PROMPT
        first_message = SAATHI_FIRST_MESSAGE
        if not prompt:
            logger.warning("system_prompt.empty falling back to hardcoded default")
            prompt = _FALLBACK_SYSTEM_PROMPT
            first_message = _FALLBACK_FIRST_MESSAGE
        return {
            "prompt": prompt,
            "firstMessage": first_message,
            "supported_languages": [
                "Hindi", "English", "Kannada", "Tamil", "Telugu",
            ],
            "usage": "These are automatically applied as Vapi assistant overrides when starting a call.",
            "fallback": not bool(SAATHI_SYSTEM_PROMPT),
        }
    except Exception as exc:
        logger.exception("system_prompt.error falling back to hardcoded default: %s", exc)
        return {
            "prompt": _FALLBACK_SYSTEM_PROMPT,
            "firstMessage": _FALLBACK_FIRST_MESSAGE,
            "supported_languages": [
                "Hindi", "English", "Kannada", "Tamil", "Telugu",
            ],
            "usage": "These are automatically applied as Vapi assistant overrides when starting a call.",
            "fallback": True,
        }


@app.get("/api/session-prompt")
def get_session_prompt() -> dict[str, Any]:
    """Return the AMBIENT SESSION prompt — silent scribe, multi-patient single call."""
    try:
        prompt = SAATHI_SESSION_PROMPT
        first_message = SAATHI_SESSION_FIRST_MESSAGE
        if not prompt:
            logger.warning("session_prompt.empty falling back to system prompt default")
            prompt = _FALLBACK_SYSTEM_PROMPT
            first_message = _FALLBACK_FIRST_MESSAGE
        return {
            "prompt": prompt,
            "firstMessage": first_message,
            "boundary_words": [
                "next", "next patient",
                "aage", "agla", "agla patient", "dusra patient",
            ],
            "end_session_words": [
                "session end", "session khatam", "bas done", "end session",
            ],
            "mode": "ambient_session",
            "fallback": not bool(SAATHI_SESSION_PROMPT),
        }
    except Exception as exc:
        logger.exception("session_prompt.error falling back to hardcoded default: %s", exc)
        return {
            "prompt": _FALLBACK_SYSTEM_PROMPT,
            "firstMessage": _FALLBACK_FIRST_MESSAGE,
            "boundary_words": [
                "next", "next patient",
                "aage", "agla", "agla patient", "dusra patient",
            ],
            "end_session_words": [
                "session end", "session khatam", "bas done", "end session",
            ],
            "mode": "ambient_session",
            "fallback": True,
        }


@app.post("/api/session-gap-prompt")
def session_gap_prompt(body: SessionGapBody) -> dict[str, Any]:
    """
    Optional host hook: transcript blob → one line to speak for missing critical fields.
    (Browser may call this after a pause; keep Vapi assistant prompt aligned so it does not narrate silence.)
    """
    say = suggest_gap_prompt(body.transcript.strip(), body.session_language)
    logger.info(
        "route.session_gap_prompt chars=%s lang=%s say=%s",
        len(body.transcript.strip()),
        body.session_language or "(default)",
        repr(say) if say else "null",
    )
    return {"say": say}


@app.get("/api/notifications/status")
def notifications_status_route() -> dict[str, Any]:
    return notification_status()


@app.post("/api/notifications/follow-up-reminders/run")
def notifications_run_follow_ups() -> dict[str, Any]:
    """Manually trigger visit-day SMS scan (same logic as hourly background job)."""
    return run_follow_up_reminders_for_today()


@app.get("/api/notifications/daily-register/preview")
def notifications_register_preview() -> dict[str, Any]:
    today = datetime.now(ZoneInfo("Asia/Kolkata")).date()
    return {
        "date": today.isoformat(),
        "preview": build_daily_register_text(get_all_patients(), today),
    }


@app.post("/api/notifications/daily-register/whatsapp")
def notifications_register_whatsapp() -> dict[str, Any]:
    try:
        return send_daily_register_whatsapp()
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@app.post("/api/notifications/daily-register/whatsapp/all")
def notifications_register_whatsapp_all() -> dict[str, Any]:
    """Send one daily register per ANM based on patient.registered_by_anm_whatsapp."""
    try:
        return send_daily_register_whatsapp_all()
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@app.get("/patients")
def list_patients(include_demo: bool = False, _key: str | None = Depends(verify_api_key)) -> list[dict[str, Any]]:
    try:
        rows = get_all_patients(include_demo=include_demo)
        logger.info("route.patients ok count=%s include_demo=%s", len(rows), include_demo)
        return rows
    except Exception as e:
        logger.warning("route.patients failed err=%s", e)
        return []


@app.get("/risk-flags")
def risk_flags(include_demo: bool = False, _key: str | None = Depends(verify_api_key)) -> list[dict[str, Any]]:
    """Red-risk patients — filtered at Qdrant DB level (not in Python)."""
    try:
        reds = get_patients_by_risk("red", include_demo=include_demo)
        logger.info("route.risk_flags ok red_count=%s (db-filtered)", len(reds))
        return reds
    except Exception as e:
        logger.warning("route.risk_flags failed err=%s", e)
        return []


@app.get("/emergencies")
def emergencies(include_demo: bool = False, _key: str | None = Depends(verify_api_key)) -> list[dict[str, Any]]:
    """Active emergencies — filtered at Qdrant DB level by emergency.is_emergency."""
    try:
        emg_patients = db_get_emergencies(include_demo=include_demo)
        result: list[dict[str, Any]] = []
        for p in emg_patients:
            em = p.get("emergency", {})
            result.append({
                "patient_id": p.get("patient_id"),
                "patient_name": p.get("patient_name"),
                "age_years": p.get("age_years"),
                "blood_pressure": p.get("blood_pressure"),
                "severity": em.get("severity"),
                "reasons": em.get("reasons", []),
                "recommended_action": em.get("recommended_action"),
                "detected_at": em.get("detected_at"),
                "location": p.get("location"),
                "phone": p.get("phone"),
            })
        return result
    except Exception as e:
        logger.warning("route.emergencies failed err=%s", e)
        return []


@app.get("/analytics")
def analytics(include_demo: bool = False, _key: str | None = Depends(verify_api_key)) -> dict[str, Any]:
    """Dashboard analytics: counts by risk, visit type, emergency stats."""
    try:
        patients = get_all_patients(include_demo=include_demo)
    except Exception:
        patients = []

    total = len(patients)
    risk_counts = {"red": 0, "amber": 0, "green": 0}
    visit_type_counts: dict[str, int] = {}
    emergency_count = 0
    has_referral = 0

    for p in patients:
        rl = p.get("risk_level", "green")
        risk_counts[rl] = risk_counts.get(rl, 0) + 1
        vt = p.get("visit_type") or "unknown"
        visit_type_counts[vt] = visit_type_counts.get(vt, 0) + 1
        em = p.get("emergency", {})
        if isinstance(em, dict) and em.get("is_emergency"):
            emergency_count += 1
        if p.get("referral_needed"):
            has_referral += 1

    return {
        "total_patients": total,
        "risk_distribution": risk_counts,
        "visit_types": visit_type_counts,
        "emergency_count": emergency_count,
        "referral_count": has_referral,
    }


@app.post("/api/clear-seeded-demo")
def clear_seeded_demo(_key: str | None = Depends(verify_api_key)) -> dict[str, Any]:
    """
    Delete patients created via ``POST /seed-demo`` (flag ``seeded_demo``).
    Called by the UI on each full page load unless ``SAATHI_KEEP_DEMO_ON_RELOAD`` is set.
    """
    if (os.environ.get("SAATHI_KEEP_DEMO_ON_RELOAD") or "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    ):
        return {"ok": True, "removed": 0, "skipped": True, "reason": "SAATHI_KEEP_DEMO_ON_RELOAD"}
    n = delete_seeded_demo_points()
    return {"ok": True, "removed": n}


@app.get("/search")
def search_patients(q: str = "", limit: int = 10, include_demo: bool = False, _key: str | None = Depends(verify_api_key)) -> dict[str, Any]:
    """
    Hybrid search: vector similarity + lexical overlap on the same text used to embed patients.
    With Gemini embeddings, results align better with short clinical phrases (e.g. \"swelling feet\").
    """
    if not q.strip():
        return {"query": q, "results": [], "embedding_mode": embedding_mode()}
    try:
        results = search_similar(q.strip(), limit=min(max(limit, 1), 50), include_demo=include_demo)
        logger.info("route.search query=%r results=%s mode=%s", q, len(results), embedding_mode())
        return {"query": q, "results": results, "embedding_mode": embedding_mode()}
    except Exception as e:
        logger.warning("route.search failed err=%s", e)
        return {"query": q, "results": [], "error": str(e)}


@app.get("/patients/by-type/{visit_type}")
def patients_by_type(visit_type: str, _key: str | None = Depends(verify_api_key)) -> list[dict[str, Any]]:
    """Filter patients by visit type at DB level (ANC, PNC, NCD, immunization)."""
    try:
        rows = get_patients_by_visit_type(visit_type)
        logger.info("route.by_type type=%s count=%s", visit_type, len(rows))
        return rows
    except Exception as e:
        logger.warning("route.by_type failed err=%s", e)
        return []


@app.get("/portal-prefill/{patient_id}")
def portal_prefill(patient_id: str, _key: str | None = Depends(verify_api_key)) -> dict[str, Any]:
    logger.info("route.portal_prefill lookup patient_id=%s", patient_id)
    row = get_patient_by_id(patient_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Patient not found")
    portals = build_portal_prefill(row)
    return {
        "patient_id": patient_id,
        "visit_program": portals["visit_program"],
        "anmol": portals["anmol"],
        "uwin": portals["uwin"],
        "ncd": portals["ncd"],
    }


# ---------------------------------------------------------------------------
# Patient-facing OPD prescription slip (printable HTML)
# ---------------------------------------------------------------------------

def _patient_advice(risk_level: str | None) -> str:
    return {
        "red": "Please visit District Hospital urgently — within 24 hours.",
        "amber": "Please follow up at PHC within 2-3 days.",
        "green": "Routine follow-up as scheduled.",
    }.get(risk_level or "green", "Routine follow-up as scheduled.")


def _esc(v: Any) -> str:
    """HTML-escape a possibly-None value, preserving '—' for empties."""
    if v is None or v == "":
        return "—"
    if isinstance(v, list):
        if not v:
            return "—"
        return html.escape(", ".join(str(x) for x in v))
    return html.escape(str(v))


def _render_prescription_html(p: dict[str, Any]) -> str:
    """Build a print-optimized OPD slip for the ward boy / patient."""
    name = _esc(p.get("patient_name"))
    pid = _esc(p.get("patient_id"))
    age = p.get("age_years")
    gender = p.get("gender")
    age_sex = " / ".join(filter(None, [str(age) if age is not None else None, gender])) or "—"
    phone = _esc(p.get("phone"))
    location = _esc(p.get("location"))
    visit_type = _esc(p.get("visit_type"))
    date_now = datetime.now(timezone.utc).strftime("%d-%b-%Y  %H:%M")
    date_iso = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    # Vitals
    bp = _esc(p.get("blood_pressure"))
    hb = _esc(p.get("hemoglobin_g_dl"))
    weight = _esc(p.get("weight_kg"))
    temp = _esc(p.get("temperature_f"))
    spo2 = _esc(p.get("spo2_percent"))
    sugar = _esc(p.get("random_blood_sugar_mg_dl"))
    pulse = _esc(p.get("pulse_rate"))

    # Clinical
    symptoms = _esc(p.get("symptoms"))
    diagnosis = _esc(p.get("diagnosis"))
    medicines = p.get("medicines_given") or []
    vaccines = p.get("vaccines_given") or []
    counseling = _esc(p.get("counseling_done"))

    # Pregnancy details (ANC)
    preg_months = p.get("pregnancy_months")
    preg_line = (
        f"<tr><td>Pregnancy</td><td>{_esc(preg_months)} months</td></tr>"
        if preg_months else ""
    )

    # Risk and advice
    risk = (p.get("risk_level") or "green").lower()
    risk_color = {"red": "#b91c1c", "amber": "#b45309", "green": "#047857"}[
        risk if risk in ("red", "amber", "green") else "green"
    ]
    advice = _patient_advice(risk)

    # Referral
    ref_needed = bool(p.get("referral_needed"))
    ref_facility = _esc(p.get("referral_facility"))
    ref_reason = _esc(p.get("referral_reason"))
    follow_up = _esc(p.get("follow_up_date"))

    # Medicines list HTML
    if medicines:
        meds_html = "<ol style='margin:6px 0 0 18px;padding:0'>" + "".join(
            f"<li style='margin:2px 0'>{html.escape(str(m))}</li>" for m in medicines
        ) + "</ol>"
    else:
        meds_html = "<p style='margin:4px 0;color:#64748b'>—</p>"

    # Vaccines list HTML
    vacc_block = ""
    if vaccines:
        vacc_block = (
            "<div class='block'><div class='label'>Vaccines administered</div>"
            "<ul style='margin:6px 0 0 18px'>"
            + "".join(f"<li>{html.escape(str(v))}</li>" for v in vaccines)
            + "</ul></div>"
        )

    # Referral block
    ref_block = ""
    if ref_needed:
        ref_block = (
            "<div class='block referral'><div class='label'>Referral</div>"
            f"<p><strong>To:</strong> {ref_facility}</p>"
            f"<p><strong>Reason:</strong> {ref_reason}</p></div>"
        )

    return f"""<!DOCTYPE html>
<html lang="en"><head>
<meta charset="UTF-8">
<title>OPD Slip — {name}</title>
<style>
  * {{ box-sizing: border-box; }}
  body {{
    font-family: ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, sans-serif;
    background: #f1f5f9; margin: 0; padding: 20px; color: #0f172a;
  }}
  .slip {{
    max-width: 720px; margin: 0 auto; background: #fff;
    border: 2px solid #1d4ed8; border-radius: 12px;
    padding: 24px 28px; box-shadow: 0 6px 24px rgba(15,23,42,.08);
  }}
  .header {{
    display:flex; justify-content:space-between; align-items:center;
    border-bottom: 2px solid #1d4ed8; padding-bottom: 10px; margin-bottom: 14px;
  }}
  .header h1 {{ font-size: 20px; margin: 0; color: #1d4ed8; letter-spacing: 1px; }}
  .header .sub {{ font-size: 11px; color: #64748b; }}
  .meta {{ display: grid; grid-template-columns: 1fr 1fr; gap: 6px 24px; font-size: 13px; }}
  .meta .k {{ color: #64748b; font-size: 11px; text-transform: uppercase; letter-spacing: .5px; }}
  .meta .v {{ font-weight: 600; }}
  .block {{ margin-top: 16px; }}
  .label {{
    font-size: 10px; font-weight: 700; letter-spacing: 1px;
    text-transform: uppercase; color: #64748b; margin-bottom: 6px;
  }}
  table.vit {{ width:100%; border-collapse: collapse; font-size: 13px; }}
  table.vit td {{ padding: 4px 8px; border-bottom: 1px dashed #e2e8f0; }}
  table.vit td:first-child {{ color:#64748b; width: 40%; }}
  table.vit td:last-child  {{ font-weight: 600; }}
  .risk-pill {{
    display: inline-block; padding: 4px 12px; border-radius: 999px;
    font-size: 11px; font-weight: 700; letter-spacing: .5px;
    color:#fff; background: {risk_color};
  }}
  .advice {{
    margin-top: 14px; padding: 12px 14px; border-radius: 10px;
    background: #fef3c7; border-left: 4px solid #d97706; font-size: 13px;
    color: #78350f;
  }}
  .referral {{
    background: #fee2e2; border-left: 4px solid #b91c1c; padding: 10px 14px;
    border-radius: 10px;
  }}
  .footer {{
    margin-top: 24px; display: flex; justify-content: space-between; align-items: end;
    padding-top: 14px; border-top: 1px dashed #cbd5e1; font-size: 12px;
  }}
  .sig {{ width: 200px; border-top: 1px solid #94a3b8; padding-top: 4px; text-align: center; color:#64748b; }}
  .actions {{ max-width: 720px; margin: 14px auto 0; display: flex; gap: 8px; justify-content: flex-end; }}
  .btn {{
    background: #1d4ed8; color: #fff; border: 0; border-radius: 8px;
    padding: 10px 18px; font-weight: 600; cursor: pointer; font-size: 13px;
  }}
  .btn.secondary {{ background: #e2e8f0; color: #0f172a; }}
  @media print {{
    body {{ background: #fff; padding: 0; }}
    .slip {{ border: none; box-shadow: none; max-width: 100%; }}
    .actions {{ display: none; }}
  }}
</style></head>
<body>
  <div class="slip">
    <div class="header">
      <div>
        <h1>SAATHI · OPD SLIP</h1>
        <div class="sub">Sub-Centre / UPHC · Auto-generated from voice consultation</div>
      </div>
      <div style="text-align:right">
        <div class="sub">Date</div>
        <div style="font-weight:600">{date_now}</div>
      </div>
    </div>

    <div class="meta">
      <div><div class="k">Patient ID</div><div class="v">{pid}</div></div>
      <div><div class="k">Visit Type</div><div class="v">{visit_type}</div></div>
      <div><div class="k">Name</div><div class="v">{name}</div></div>
      <div><div class="k">Age / Sex</div><div class="v">{age_sex}</div></div>
      <div><div class="k">Phone</div><div class="v">{phone}</div></div>
      <div><div class="k">Address</div><div class="v">{location}</div></div>
    </div>

    <div class="block">
      <div class="label">Vitals</div>
      <table class="vit">
        <tr><td>Blood Pressure</td><td>{bp}</td></tr>
        <tr><td>Hemoglobin (g/dL)</td><td>{hb}</td></tr>
        <tr><td>Weight (kg)</td><td>{weight}</td></tr>
        <tr><td>Pulse</td><td>{pulse}</td></tr>
        <tr><td>Temperature (°F)</td><td>{temp}</td></tr>
        <tr><td>SpO₂ (%)</td><td>{spo2}</td></tr>
        <tr><td>Random Blood Sugar (mg/dL)</td><td>{sugar}</td></tr>
        {preg_line}
      </table>
    </div>

    <div class="block">
      <div class="label">Complaints</div>
      <p style="margin:4px 0">{symptoms}</p>
    </div>

    <div class="block">
      <div class="label">Diagnosis</div>
      <p style="margin:4px 0">{diagnosis}</p>
    </div>

    <div class="block">
      <div class="label">Medicines Prescribed</div>
      {meds_html}
    </div>

    {vacc_block}

    <div class="block">
      <div class="label">Counseling</div>
      <p style="margin:4px 0;color:#0f172a">{counseling}</p>
    </div>

    {ref_block}

    <div class="advice">
      <strong>Advice:</strong> {html.escape(advice)}
      <br><strong>Follow-up:</strong> {follow_up}
    </div>

    <div class="footer">
      <div>
        <div><span class="risk-pill">{risk.upper()}</span></div>
        <div style="margin-top:6px;color:#64748b;font-size:11px">
          Generated: {date_iso} · For patient reference. Not a medical certificate.
        </div>
      </div>
      <div class="sig">ANM Signature</div>
    </div>
  </div>

  <div class="actions">
    <button class="btn secondary" onclick="window.close()">Close</button>
    <button class="btn" onclick="window.print()">Print OPD slip</button>
  </div>

  <script>
    // Auto-trigger print dialog when ?autoprint=1
    if (new URLSearchParams(location.search).get('autoprint') === '1') {{
      setTimeout(function () {{ window.print(); }}, 300);
    }}
  </script>
</body></html>
"""


@app.get("/patient/{patient_id}/prescription", response_class=HTMLResponse)
def patient_prescription(patient_id: str) -> HTMLResponse:
    """Print-optimized OPD slip for the patient. Ward boy clicks → prints → hands to patient."""
    logger.info("route.prescription lookup patient_id=%s", patient_id)
    row = get_patient_by_id(patient_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Patient not found")
    return HTMLResponse(_render_prescription_html(row))


@app.post("/seed-demo")
def seed_demo(_key: str | None = Depends(verify_api_key)) -> dict[str, Any]:
    """Insert realistic demo patients covering ANC, immunization, NCD, PNC, and emergencies."""
    mocks: list[dict[str, Any]] = [
        # ANC - Normal
        {
            "patient_name": "Sunita Sharma", "age_years": 24, "gender": "female",
            "pregnancy_months": 5, "visit_type": "ANC", "location": "Rampur Village",
            "blood_pressure": "120/80", "blood_pressure_systolic": 120, "blood_pressure_diastolic": 80,
            "hemoglobin_g_dl": 10.5, "weight_kg": 55, "height_cm": 155,
            "tt_doses": 1, "ifa_tablets_given": 100,
            "counseling_done": ["nutrition", "danger signs", "birth preparedness"],
            "follow_up_date": "2026-05-15",
        },
        # ANC - High Risk (pre-eclampsia)
        {
            "patient_name": "Kavita Devi", "age_years": 30, "gender": "female",
            "pregnancy_months": 8, "visit_type": "ANC", "location": "Sitapur Block",
            "blood_pressure": "155/100", "blood_pressure_systolic": 155, "blood_pressure_diastolic": 100,
            "hemoglobin_g_dl": 8.2, "weight_kg": 62, "urine_albumin": "++",
            "symptoms": ["headache", "swelling in feet"],
            "referral_needed": True, "referral_facility": "District Hospital Sitapur",
            "referral_reason": "Suspected pre-eclampsia",
            "emergency_signs": ["severe headache", "swelling face"],
        },
        # NCD - Diabetes + Hypertension (Urban)
        {
            "patient_name": "Rahul Mehra", "age_years": 52, "gender": "male",
            "visit_type": "NCD", "location": "Sector 15, Noida UPHC",
            "blood_pressure": "148/92", "blood_pressure_systolic": 148, "blood_pressure_diastolic": 92,
            "hemoglobin_g_dl": 13.0, "bmi": 28.4, "weight_kg": 82, "height_cm": 170,
            "random_blood_sugar_mg_dl": 245, "pulse_rate": 88,
            "medicines_given": ["Amlodipine 5mg", "Metformin 500mg"],
            "symptoms": ["frequent urination", "blurred vision"],
            "counseling_done": ["diet modification", "exercise", "tobacco cessation"],
            "follow_up_date": "2026-04-26",
        },
        # Immunization - Baby
        {
            "patient_name": "Baby of Anita", "age_years": 0, "gender": "male",
            "visit_type": "immunization", "location": "Anganwadi Centre, Lakhimpur",
            "child_age_months": 3, "child_weight_kg": 5.2,
            "vaccines_given": ["BCG", "OPV-0", "Hep-B Birth", "OPV-1", "Pentavalent-1", "RVV-1", "fIPV-1", "PCV-1"],
            "vaccines_due": ["OPV-2", "Pentavalent-2", "RVV-2"],
            "breastfeeding_status": "exclusive",
            "notes": "Healthy infant, gaining weight well",
            "follow_up_date": "2026-05-10",
        },
        # NCD - Urban elderly
        {
            "patient_name": "Geeta Kaur", "age_years": 58, "gender": "female",
            "visit_type": "NCD", "location": "UPHC Janakpuri, Delhi",
            "blood_pressure": "118/76", "blood_pressure_systolic": 118, "blood_pressure_diastolic": 76,
            "hemoglobin_g_dl": 12.2, "random_blood_sugar_mg_dl": 110,
            "weight_kg": 65, "height_cm": 160, "bmi": 25.4,
            "counseling_done": ["breast self-examination", "cervical screening"],
            "notes": "CBAC score 3 — routine follow-up",
        },
        # ANC - Anemia
        {
            "patient_name": "Priya Singh", "age_years": 22, "gender": "female",
            "pregnancy_months": 4, "visit_type": "ANC", "location": "Bareilly CHC",
            "blood_pressure": "132/84", "blood_pressure_systolic": 132, "blood_pressure_diastolic": 84,
            "hemoglobin_g_dl": 6.8, "weight_kg": 48,
            "ifa_tablets_given": 200, "calcium_tablets_given": 100,
            "referral_needed": True, "referral_facility": "District Hospital",
            "referral_reason": "Severe anemia — Hb 6.8 needs parenteral iron",
        },
        # NCD - Senior diabetes (Urban)
        {
            "patient_name": "Vikram Joshi", "age_years": 61, "gender": "male",
            "visit_type": "NCD", "location": "Polyclinic Bandra, Mumbai",
            "blood_pressure": "138/88", "blood_pressure_systolic": 138, "blood_pressure_diastolic": 88,
            "random_blood_sugar_mg_dl": 320, "fasting_blood_sugar_mg_dl": 185,
            "hemoglobin_g_dl": 14.0, "weight_kg": 78, "height_cm": 172,
            "medicines_given": ["Glimepiride 2mg", "Telmisartan 40mg"],
            "symptoms": ["tingling in feet", "excessive thirst"],
            "diagnosis": "Uncontrolled Type 2 DM with peripheral neuropathy",
            "referral_needed": True, "referral_facility": "Diabetology OPD, KEM Hospital",
        },
        # NCD - Severe anemia elderly
        {
            "patient_name": "Meena Yadav", "age_years": 45, "gender": "female",
            "visit_type": "NCD", "location": "Varanasi Sub Centre",
            "blood_pressure": "128/82", "blood_pressure_systolic": 128, "blood_pressure_diastolic": 82,
            "hemoglobin_g_dl": 6.2, "weight_kg": 42,
            "symptoms": ["fatigue", "breathlessness on exertion", "pallor"],
            "medicines_given": ["Ferrous Sulphate", "Folic Acid"],
            "referral_needed": True, "referral_facility": "CHC Varanasi",
            "referral_reason": "Severe anemia for evaluation",
        },
        # PNC - Post delivery
        {
            "patient_name": "Asha Rani", "age_years": 29, "gender": "female",
            "visit_type": "PNC", "location": "Lucknow PHC",
            "blood_pressure": "125/78", "blood_pressure_systolic": 125, "blood_pressure_diastolic": 78,
            "hemoglobin_g_dl": 10.0, "temperature_f": 98.6,
            "breastfeeding_status": "exclusive",
            "child_weight_kg": 3.1,
            "counseling_done": ["breastfeeding", "family planning", "newborn care"],
            "notes": "Day 7 PNC visit. Mother and baby healthy.",
        },
        # EMERGENCY - Hypertensive crisis (Urban)
        {
            "patient_name": "Deepak Nair", "age_years": 55, "gender": "male",
            "visit_type": "NCD", "location": "UPHC Kochi, Kerala",
            "blood_pressure": "185/110", "blood_pressure_systolic": 185, "blood_pressure_diastolic": 110,
            "hemoglobin_g_dl": 14.0, "pulse_rate": 110, "spo2_percent": 93,
            "random_blood_sugar_mg_dl": 180,
            "symptoms": ["severe headache", "chest tightness", "dizziness"],
            "emergency_signs": ["chest pain", "severe headache"],
            "referral_needed": True, "referral_facility": "Government Medical College Hospital",
            "referral_reason": "Hypertensive emergency with chest pain",
        },
    ]
    ids: list[str] = []
    try:
        for raw in mocks:
            raw["seeded_demo"] = True
            row = calculate_risk(raw)
            ids.append(store_patient(row))
    except Exception as e:
        logger.exception("route.seed_demo failed")
        raise HTTPException(status_code=503, detail=str(e)) from e
    logger.info("route.seed_demo ok inserted=%s", len(ids))
    return {"status": "ok", "inserted": len(ids), "patient_ids": ids}


@app.post("/process-visit")
def process_visit(body: ProcessVisitBody, _key: str | None = Depends(verify_api_key)) -> dict[str, Any]:
    logger.info("route.process_visit begin chars=%s", len(body.transcript))
    extracted = _extract_llm(body.transcript)
    out = calculate_risk(extracted)
    return out


_ALLOWED_PATIENT_FIELDS = {
    "patient_name", "age_years", "gender", "phone", "location",
    "visit_type", "pregnancy_months", "gravida", "para", "lmp_date", "edd_date",
    "blood_pressure", "blood_pressure_systolic", "blood_pressure_diastolic",
    "hemoglobin_g_dl", "weight_kg", "height_cm", "bmi",
    "temperature_f", "pulse_rate", "spo2_percent",
    "random_blood_sugar_mg_dl", "fasting_blood_sugar_mg_dl",
    "urine_albumin", "urine_sugar", "blood_group",
    "symptoms", "diagnosis", "medicines_given",
    "vaccines_given", "vaccines_due", "child_age_months", "child_weight_kg",
    "breastfeeding_status", "tt_doses", "ifa_tablets_given", "calcium_tablets_given",
    "referral_needed", "referral_facility", "referral_reason",
    "follow_up_date", "counseling_done", "emergency_signs", "notes",
    "language_used",
}


@app.post("/store-patient")
def store_patient_route(
    body: dict[str, Any] = Body(...),
    _key: str | None = Depends(verify_api_key),
) -> dict[str, Any]:
    if not isinstance(body, dict) or not body:
        raise HTTPException(status_code=400, detail="JSON object body required")
    filtered = {k: v for k, v in body.items() if k in _ALLOWED_PATIENT_FIELDS}
    if not filtered:
        raise HTTPException(status_code=400, detail="No valid patient fields provided")
    enriched = calculate_risk(filtered)
    try:
        pid = store_patient(enriched)
    except Exception as e:
        logger.exception("route.store_patient failed")
        raise HTTPException(status_code=503, detail="Storage service unavailable") from e
    return {"status": "stored", "patient_id": pid}


@app.post("/vapi-webhook")
async def vapi_webhook(
    request: Request,
) -> dict[str, Any]:
    logger.info("WEBHOOK_HIT path=/vapi-webhook")
    try:
        payload = await request.json()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON: {e}") from e

    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="Body must be a JSON object")

    logger.info("WEBHOOK payload_keys=%s", list(payload.keys()))

    transcript = payload.get("transcript")
    if transcript is None or not isinstance(transcript, str):
        raise HTTPException(status_code=400, detail="Payload must include string field transcript")
    if not transcript.strip():
        return {"status": "ignored", "reason": "empty_transcript"}

    event = payload.get("event")
    final_events = {None, "call-ended", "end-of-call-report"}
    if event not in final_events:
        return {"status": "ignored", "event": event}

    call_block = payload.get("call")
    call_id = call_block.get("id") if isinstance(call_block, dict) else None
    call_id_str = str(call_id) if call_id is not None else None

    return _run_voice_pipeline(
        transcript.strip(),
        log_prefix="[vapi-webhook]",
        vapi_call_id=call_id_str,
    )


@app.post("/simulate-call")
def simulate_call(body: SimulateCallBody, _key: str | None = Depends(verify_api_key)) -> dict[str, Any]:
    logger.info("route.simulate_call begin chars=%s", len(body.transcript))
    return _run_voice_pipeline(
        body.transcript.strip(),
        log_prefix="[simulate-call]",
        vapi_call_id=None,
        anm_id=body.anm_id,
        anm_name=body.anm_name,
        anm_whatsapp=body.anm_whatsapp,
    )


# ---------------------------------------------------------------------------
# Frontend serving with Vapi key injection
# ---------------------------------------------------------------------------

_frontend_dir = _REPO_ROOT / "frontend"
_UI_INDEX = _frontend_dir / "index.html"


def _inject_ui_index() -> str:
    if not _UI_INDEX.is_file():
        return "<!DOCTYPE html><html><body>frontend/index.html missing</body></html>"
    raw = _UI_INDEX.read_text(encoding="utf-8")
    pk = html.escape(os.environ.get("VAPI_PUBLIC_KEY", ""), quote=True)
    aid = html.escape(os.environ.get("VAPI_ASSISTANT_ID", ""), quote=True)
    return raw.replace("__INJECT_VAPI_PUBLIC_KEY__", pk).replace("__INJECT_VAPI_ASSISTANT_ID__", aid)


@app.get("/ui", include_in_schema=False)
def ui_redirect() -> RedirectResponse:
    return RedirectResponse(url="/ui/", status_code=307)


@app.get("/ui/", include_in_schema=False, response_class=HTMLResponse)
def ui_index() -> HTMLResponse:
    return HTMLResponse(_inject_ui_index())


@app.get("/ui/index.html", include_in_schema=False, response_class=HTMLResponse)
def ui_index_file() -> HTMLResponse:
    return HTMLResponse(_inject_ui_index())
