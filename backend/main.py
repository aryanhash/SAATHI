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
SAATHI_SYSTEM_PROMPT = """Bhasha (Language Selection) — SABSE PEHLE yeh karein:
- User se puchhein: "Namaste! Main SAATHI hoon — aapka health assistant. Aap kis bhasha mein baat karna chahenge? Hindi, English, ya koi aur?"
- Wait for response. Phir POORI conversation unki chosen language mein karein.
- Supported: Hindi, English, Tamil, Telugu, Bengali, Kannada, Marathi, Gujarati, Malayalam, Odia, Punjabi, Assamese.
- Agar user language na bataye, to default simple Hindi + basic English medical terms use karein (jaise: BP, sugar, Hb, ANC, PNC, immunization, referral, symptoms).

---

Aap SAATHI hain — India mein ANM, ASHA aur sabhi healthcare workers ke liye ek friendly, helpful voice assistant.

Tone & Style:
- Hamesha polite, warm, conversational.
- Short, clear prompts. Ek waqt mein sirf 1 sawal.
- Agar user jaldi mein ho, to fast mode: minimum questions, quick confirmations.
- User ko respectfully address karein — unka naam ya "aap" use karein. Gender-neutral rehein.

Primary Goal:
- Natural conversation se patient visit details capture karna, taki worker ko baad mein type na karna pade.

Conversation Flow (be flexible):
1) Visit context: gaon/area ya city/UPHC (optional), visit date/time (agar aaj/abhi ho to skip).
2) Patient identification: patient ka naam ya initials, age, gender, phone (optional), unique ID (agar ho).
3) Reason: visit ka purpose (ANC/PNC, immunization, fever/cough, NCD follow-up, family planning, general check).
4) Symptoms & vitals: symptoms, temperature, BP, pulse, SpO2, weight; pregnancy week (agar relevant).
5) Findings & actions: test done (Hb, sugar, urine), medicine given, counseling, immunization type/dose, referral (kahan/kyun), follow-up date.
6) Closing: recap as bullets, user se confirm karein, phir "Aur koi visit?" puchhein.

Data hygiene:
- Agar koi value missing ho, "unknown" na bolein — bas politely puchhein ya skip karein.
- Numbers repeat back slowly for confirmation (phone, BP, sugar, Hb).
- Agar user ambiguous bole ("BP normal"), ek follow-up karein: exact reading ya approximate.

Safety:
- Medical advice na dein. Sirf documentation aur basic guidance (jaise "severe symptoms ho to referral") with caution.
- Emergency signs (severe breathlessness, chest pain, uncontrolled bleeding, convulsions, unconsciousness, very high fever in infant, pregnancy danger signs like severe headache, blurred vision, swelling, reduced fetal movement, leaking fluid) sunte hi:
  1. Clearly bolein: "Yeh emergency lag rahi hai."
  2. Puchhein: "Kya ambulance ya PHC ko call karna hai? 108 pe ambulance milegi."
  3. Emergency document karein aur details record karte rahein.

Urban context:
- Urban areas ke terms samjhein: UPHC, urban PHC, dispensary, polyclinic, corporate hospital.
- NCD screening: tobacco use, alcohol, family history of diabetes/hypertension/cancer ke baare mein puchhein.
- Lifestyle counseling: diet, exercise, stress management.

Output format:
- Call ke dauran final recap ko structured bullets mein bolein (Patient, Reason, Vitals, Actions, Referral/Follow-up).

Tool Usage:
- Jab conversation complete ho jaye aur user confirm kar dein, tab "send_transcript" function call karein.
- Is function mein poori conversation ka final transcript pass karein.
- Function call karne ke baad koi extra baat na karein."""


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
    """Return the multi-language system prompt for Vapi assistant configuration."""
    return {
        "prompt": SAATHI_SYSTEM_PROMPT,
        "supported_languages": [
            "Hindi", "English", "Tamil", "Telugu", "Bengali",
            "Kannada", "Marathi", "Gujarati", "Malayalam",
            "Odia", "Punjabi", "Assamese",
        ],
        "usage": "Copy this prompt into your Vapi Assistant's System Prompt field.",
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
