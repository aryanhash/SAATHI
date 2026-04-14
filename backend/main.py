"""SAATHI backend — FastAPI application."""

from __future__ import annotations

import html
import json
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv
from fastapi import Body, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, PlainTextResponse, RedirectResponse
from pydantic import BaseModel, Field

from db.qdrant_client import (
    create_collection,
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
from services.portal_mapper import build_portal_prefill
from services.risk_engine import calculate_risk, detect_emergency

_REPO_ROOT = Path(__file__).resolve().parent.parent
_BACKEND_DIR = Path(__file__).resolve().parent
load_dotenv(_REPO_ROOT / ".env")
load_dotenv(_BACKEND_DIR / ".env", override=True)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Multi-language system prompt for Vapi assistant
# ---------------------------------------------------------------------------
SAATHI_SYSTEM_PROMPT = """\
[Identity]
You are SAATHI — a friendly, supportive voice assistant designed for India's healthcare workers, \
such as ANMs and ASHAs. Your primary purpose is to assist users in documenting patient visits \
and conversations naturally, respectfully, and efficiently, reducing reliance on manual typing.

[Language Selection — FIRST STEP, ALWAYS]
- Start every call with the numbered language menu exactly as given in your first message.
- Wait for the user to say a number (1–12) or a language name before saying anything else.
- Map the response:
  1 = Hindi, 2 = English, 3 = Tamil, 4 = Telugu, 5 = Bengali,
  6 = Kannada, 7 = Marathi, 8 = Gujarati, 9 = Malayalam,
  10 = Odia, 11 = Punjabi, 12 = Assamese.
- Once confirmed, conduct the ENTIRE rest of the conversation in that language only.
- Default: if user does not pick or is unclear, ask once more; if still unclear, use Hindi.
- If user requests an unsupported language, say: "Sorry, woh language abhi support nahi hai. \
Kya Hindi ya English mein baat karein?"

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
  Say the numbered language menu from your first message.
  <wait for user to say a number or language name>
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
    "Namaste! Main SAATHI hoon — aapka health assistant. "
    "Apni bhasha chuniye: "
    "Hindi ke liye 1 bolein, "
    "English ke liye 2, "
    "Tamil ke liye 3, "
    "Telugu ke liye 4, "
    "Bengali ke liye 5, "
    "Kannada ke liye 6, "
    "Marathi ke liye 7, "
    "Gujarati ke liye 8, "
    "Malayalam ke liye 9, "
    "Odia ke liye 10, "
    "Punjabi ke liye 11, "
    "Assamese ke liye 12."
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
    ollama = os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434 (default)")
    gemini = "set" if (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")) else "unset"
    qdrant_key = "set" if os.environ.get("QDRANT_API_KEY") else "unset"
    vapi_pk = "set" if os.environ.get("VAPI_PUBLIC_KEY") else "unset"
    vapi_aid = "set" if os.environ.get("VAPI_ASSISTANT_ID") else "unset"
    logger.info(
        "saathi.boot PORT=%s QDRANT_URL=%s QDRANT_API_KEY=%s OLLAMA_BASE_URL=%s GEMINI_API_KEY=%s "
        "VAPI_PUBLIC_KEY=%s VAPI_ASSISTANT_ID=%s llm_backend=%s",
        port, qmode, qdrant_key, ollama, gemini, vapi_pk, vapi_aid, extraction_backend(),
    )
    _fp = _REPO_ROOT / "frontend" / "index.html"
    if not _fp.is_file():
        logger.warning("saathi.frontend_missing path=%s", _fp)
    try:
        create_collection()
        logger.info("saathi.boot qdrant_ok collection_ready")
    except Exception as e:
        logger.warning("saathi.boot qdrant_skip reason=%s", e)
    yield


app = FastAPI(title="SAATHI", version="0.2.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


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


def _extract_llm(transcript: str) -> dict[str, Any]:
    logger.info("pipeline.extract start chars=%s backend=%s", len(transcript), extraction_backend())
    try:
        out = extract_patient_data(transcript)
        logger.info("pipeline.extract done keys=%s", list(out.keys()))
        return out
    except requests.HTTPError as e:
        code = e.response.status_code if e.response is not None else "?"
        hint = (e.response.text[:240] + "…") if e.response is not None and e.response.text else ""
        raise HTTPException(
            status_code=502,
            detail=f"LLM HTTP error ({extraction_backend()}): status={code} {hint}".strip(),
        ) from e
    except requests.RequestException as e:
        raise HTTPException(
            status_code=503,
            detail=f"Cannot reach LLM ({extraction_backend()}): {e}",
        ) from e
    except ValueError as e:
        raise HTTPException(status_code=502, detail=str(e)) from e


def _run_voice_pipeline(
    transcript: str,
    *,
    log_prefix: str,
    vapi_call_id: str | None = None,
) -> dict[str, Any]:
    logger.info("%s pipeline.begin transcript_chars=%s", log_prefix, len(transcript))
    logger.info("%s pipeline.transcript %s", log_prefix, transcript)
    extracted = _extract_llm(transcript)
    logger.info(
        "%s pipeline.extracted_json %s",
        log_prefix,
        json.dumps(extracted, ensure_ascii=False, default=str),
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
    try:
        pid = store_patient(data)
    except Exception as e:
        logger.exception("%s pipeline.store FAILED", log_prefix)
        raise HTTPException(status_code=503, detail=f"Qdrant store failed: {e}") from e
    data["patient_id"] = pid
    logger.info("%s pipeline.complete patient_id=%s", log_prefix, pid)
    return {"status": "processed", "patient": data}


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/", response_class=PlainTextResponse)
def root() -> str:
    return "SAATHI backend running"


@app.get("/health/llm")
def health_llm() -> dict[str, Any]:
    gemini_set = bool(os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"))
    return {
        "llm_backend": extraction_backend(),
        "gemini_key_present": gemini_set,
        "gemini_model": GEMINI_MODEL,
        "embedding_mode": embedding_mode(),
        "saathi_llm_env": os.environ.get("SAATHI_LLM"),
        "vapi_public_key_present": bool((os.environ.get("VAPI_PUBLIC_KEY") or "").strip()),
        "vapi_assistant_id_present": bool((os.environ.get("VAPI_ASSISTANT_ID") or "").strip()),
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
    """Return the multi-language system prompt and first message for Vapi assistant."""
    return {
        "prompt": SAATHI_SYSTEM_PROMPT,
        "firstMessage": SAATHI_FIRST_MESSAGE,
        "supported_languages": [
            "Hindi", "English", "Tamil", "Telugu", "Bengali",
            "Kannada", "Marathi", "Gujarati", "Malayalam",
            "Odia", "Punjabi", "Assamese",
        ],
        "usage": "These are automatically applied as Vapi assistant overrides when starting a call.",
    }


@app.get("/patients")
def list_patients() -> list[dict[str, Any]]:
    try:
        rows = get_all_patients()
        logger.info("route.patients ok count=%s", len(rows))
        return rows
    except Exception as e:
        logger.warning("route.patients failed err=%s", e)
        return []


@app.get("/risk-flags")
def risk_flags() -> list[dict[str, Any]]:
    """Red-risk patients — filtered at Qdrant DB level (not in Python)."""
    try:
        reds = get_patients_by_risk("red")
        logger.info("route.risk_flags ok red_count=%s (db-filtered)", len(reds))
        return reds
    except Exception as e:
        logger.warning("route.risk_flags failed err=%s", e)
        return []


@app.get("/emergencies")
def emergencies() -> list[dict[str, Any]]:
    """Active emergencies — filtered at Qdrant DB level by emergency.is_emergency."""
    try:
        emg_patients = db_get_emergencies()
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
def analytics() -> dict[str, Any]:
    """Dashboard analytics: counts by risk, visit type, emergency stats."""
    try:
        patients = get_all_patients()
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


@app.get("/search")
def search_patients(q: str = "", limit: int = 10) -> dict[str, Any]:
    """
    Semantic search: find patients whose records are most similar to the query.
    With Gemini embeddings → true semantic similarity (e.g. "headache pregnant" finds ANC patients with headache).
    """
    if not q.strip():
        return {"query": q, "results": [], "embedding_mode": embedding_mode()}
    try:
        results = search_similar(q.strip(), limit=min(limit, 50))
        logger.info("route.search query=%r results=%s mode=%s", q, len(results), embedding_mode())
        return {"query": q, "results": results, "embedding_mode": embedding_mode()}
    except Exception as e:
        logger.warning("route.search failed err=%s", e)
        return {"query": q, "results": [], "error": str(e)}


@app.get("/patients/by-type/{visit_type}")
def patients_by_type(visit_type: str) -> list[dict[str, Any]]:
    """Filter patients by visit type at DB level (ANC, PNC, NCD, immunization)."""
    try:
        rows = get_patients_by_visit_type(visit_type)
        logger.info("route.by_type type=%s count=%s", visit_type, len(rows))
        return rows
    except Exception as e:
        logger.warning("route.by_type failed err=%s", e)
        return []


@app.get("/portal-prefill/{patient_id}")
def portal_prefill(patient_id: str) -> dict[str, Any]:
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


@app.post("/seed-demo")
def seed_demo() -> dict[str, Any]:
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
            row = calculate_risk(raw)
            ids.append(store_patient(row))
    except Exception as e:
        logger.exception("route.seed_demo failed")
        raise HTTPException(status_code=503, detail=str(e)) from e
    logger.info("route.seed_demo ok inserted=%s", len(ids))
    return {"status": "ok", "inserted": len(ids), "patient_ids": ids}


@app.post("/process-visit")
def process_visit(body: ProcessVisitBody) -> dict[str, Any]:
    logger.info("route.process_visit begin chars=%s", len(body.transcript))
    extracted = _extract_llm(body.transcript)
    out = calculate_risk(extracted)
    return out


@app.post("/store-patient")
def store_patient_route(body: dict[str, Any] = Body(...)) -> dict[str, Any]:
    if not isinstance(body, dict) or not body:
        raise HTTPException(status_code=400, detail="JSON object body required")
    try:
        pid = store_patient(body)
    except Exception as e:
        logger.exception("route.store_patient failed")
        raise HTTPException(status_code=503, detail=f"Qdrant store failed: {e}") from e
    return {"status": "stored", "patient_id": pid}


@app.post("/vapi-webhook")
async def vapi_webhook(request: Request) -> dict[str, Any]:
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
def simulate_call(body: SimulateCallBody) -> dict[str, Any]:
    logger.info("route.simulate_call begin chars=%s", len(body.transcript))
    return _run_voice_pipeline(
        body.transcript.strip(),
        log_prefix="[simulate-call]",
        vapi_call_id=None,
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
