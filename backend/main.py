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

from db.qdrant_client import create_collection, get_all_patients, get_patient_by_id, store_patient
from llm.extractor import GEMINI_MODEL, extract_patient_data, extraction_backend
from services.portal_mapper import build_portal_prefill
from services.risk_engine import calculate_risk

_REPO_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_REPO_ROOT / ".env")

logger = logging.getLogger(__name__)


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
        port,
        qmode,
        qdrant_key,
        ollama,
        gemini,
        vapi_pk,
        vapi_aid,
        extraction_backend(),
    )
    try:
        create_collection()
        logger.info("saathi.boot qdrant_ok collection_ready")
    except Exception as e:
        logger.warning("saathi.boot qdrant_skip reason=%s (stores will fail until Qdrant is reachable)", e)
    yield


app = FastAPI(title="SAATHI", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def http_request_logging(request: Request, call_next):
    """One-line request/response logging for Render + hackathon debugging."""
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
    if vapi_call_id:
        data["vapi_call_id"] = vapi_call_id
        logger.info("%s pipeline.meta vapi_call_id=%s", log_prefix, vapi_call_id)
    try:
        logger.info("%s pipeline.store start", log_prefix)
        pid = store_patient(data)
    except Exception as e:
        logger.exception("%s pipeline.store FAILED", log_prefix)
        raise HTTPException(status_code=503, detail=f"Qdrant store failed: {e}") from e
    data["patient_id"] = pid
    logger.info("%s pipeline.complete patient_id=%s status=processed", log_prefix, pid)
    return {"status": "processed", "patient": data}


@app.get("/", response_class=PlainTextResponse)
def root() -> str:
    return "SAATHI backend running"


@app.get("/health/llm")
def health_llm() -> dict[str, Any]:
    """Which LLM backend this instance will use (no secrets). For Render env debugging."""
    gemini_set = bool(os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"))
    return {
        "llm_backend": extraction_backend(),
        "gemini_key_present": gemini_set,
        "gemini_model": GEMINI_MODEL,
        "saathi_llm_env": os.environ.get("SAATHI_LLM"),
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
    try:
        reds = [p for p in get_all_patients() if p.get("risk_level") == "red"]
        logger.info("route.risk_flags ok red_count=%s", len(reds))
        return reds
    except Exception as e:
        logger.warning("route.risk_flags failed err=%s", e)
        return []


@app.get("/portal-prefill/{patient_id}")
def portal_prefill(patient_id: str) -> dict[str, Any]:
    logger.info("route.portal_prefill lookup patient_id=%s", patient_id)
    row = get_patient_by_id(patient_id)
    if row is None:
        logger.warning("route.portal_prefill not_found patient_id=%s", patient_id)
        raise HTTPException(status_code=404, detail="Patient not found")
    portals = build_portal_prefill(row)
    logger.info(
        "route.portal_prefill ok patient_id=%s program=%s",
        patient_id,
        portals.get("visit_program"),
    )
    return {
        "patient_id": patient_id,
        "visit_program": portals["visit_program"],
        "anmol": portals["anmol"],
        "uwin": portals["uwin"],
        "ncd": portals["ncd"],
    }


@app.post("/seed-demo")
def seed_demo() -> dict[str, Any]:
    """Insert ~10 mock patients for hackathon demos (no LLM)."""
    mocks: list[dict[str, Any]] = [
        {"patient_name": "Sunita Sharma", "age_years": 24, "pregnancy_months": 5, "blood_pressure": "120/80", "blood_pressure_systolic": 120, "blood_pressure_diastolic": 80, "hemoglobin_g_dl": 10.5},
        {"patient_name": "Kavita Devi", "age_years": 30, "pregnancy_months": 8, "blood_pressure": "150/95", "blood_pressure_systolic": 150, "blood_pressure_diastolic": 95, "hemoglobin_g_dl": 11.0},
        {"patient_name": "Rahul Mehra", "age_years": 52, "blood_pressure": "142/90", "blood_pressure_systolic": 142, "blood_pressure_diastolic": 90, "hemoglobin_g_dl": 13.0, "bmi": 28.4},
        {"patient_name": "Baby of Anita", "age_years": 0, "notes": "Immunization OPV due", "blood_pressure": None, "hemoglobin_g_dl": None},
        {"patient_name": "Geeta Kaur", "age_years": 38, "blood_pressure": "118/76", "blood_pressure_systolic": 118, "blood_pressure_diastolic": 76, "hemoglobin_g_dl": 12.2},
        {"patient_name": "Priya Singh", "age_years": 22, "pregnancy_months": 4, "blood_pressure": "132/84", "blood_pressure_systolic": 132, "blood_pressure_diastolic": 84, "hemoglobin_g_dl": 9.8},
        {"patient_name": "Vikram Joshi", "age_years": 61, "blood_pressure": "138/88", "blood_pressure_systolic": 138, "blood_pressure_diastolic": 88, "random_blood_sugar_mg_dl": 142},
        {"patient_name": "Meena Yadav", "age_years": 45, "blood_pressure": "128/82", "blood_pressure_systolic": 128, "blood_pressure_diastolic": 82, "hemoglobin_g_dl": 6.2},
        {"patient_name": "Asha Rani", "age_years": 29, "pregnancy_months": 6, "blood_pressure": "125/78", "blood_pressure_systolic": 125, "blood_pressure_diastolic": 78, "hemoglobin_g_dl": 10.0},
        {"patient_name": "Deepak Nair", "age_years": 55, "blood_pressure": "145/92", "blood_pressure_systolic": 145, "blood_pressure_diastolic": 92, "hemoglobin_g_dl": 14.0},
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
    logger.info("route.process_visit done risk_level=%s", out.get("risk_level"))
    return out


@app.post("/store-patient")
def store_patient_route(body: dict[str, Any] = Body(...)) -> dict[str, Any]:
    if not isinstance(body, dict) or not body:
        raise HTTPException(status_code=400, detail="JSON object body required")
    logger.info("route.store_patient keys=%s", list(body.keys()))
    try:
        pid = store_patient(body)
    except Exception as e:
        logger.exception("route.store_patient failed")
        raise HTTPException(status_code=503, detail=f"Qdrant store failed: {e}") from e
    logger.info("route.store_patient ok patient_id=%s", pid)
    return {"status": "stored", "patient_id": pid}


@app.post("/vapi-webhook")
async def vapi_webhook(request: Request) -> dict[str, Any]:
    logger.info("WEBHOOK_HIT path=/vapi-webhook (Vapi delivery received)")
    try:
        payload = await request.json()
    except Exception as e:
        logger.warning("WEBHOOK json_parse_failed err=%s", e)
        raise HTTPException(status_code=400, detail=f"Invalid JSON: {e}") from e

    if not isinstance(payload, dict):
        logger.warning("WEBHOOK invalid_body type=%s", type(payload).__name__)
        raise HTTPException(status_code=400, detail="Body must be a JSON object")

    logger.info("WEBHOOK payload_keys=%s", list(payload.keys()))
    logger.debug("WEBHOOK payload_full=%s", json.dumps(payload, default=str)[:8000])

    # Vapi often sends { "transcript": "..." } with no `event` key on the final payload.
    # We also accept explicit event == "call-ended" (and a few aliases seen in the wild).
    transcript = payload.get("transcript")
    if transcript is None or not isinstance(transcript, str):
        logger.warning(
            "WEBHOOK action=rejected reason=missing_transcript type=%s",
            type(transcript).__name__,
        )
        raise HTTPException(status_code=400, detail="Payload must include string field transcript")
    if not transcript.strip():
        # Vapi may POST multiple times; empty transcript should not be a client error (avoid 400 + retries).
        logger.info("WEBHOOK action=ignored reason=empty_transcript raw_len=%s", len(transcript))
        return {"status": "ignored", "reason": "empty_transcript"}

    event = payload.get("event")
    logger.info("WEBHOOK event=%r", event)
    final_events = {None, "call-ended", "end-of-call-report"}
    if event not in final_events:
        logger.info("WEBHOOK action=ignored reason=event_not_final event=%r", event)
        return {"status": "ignored", "event": event}

    call_block = payload.get("call")
    call_id = call_block.get("id") if isinstance(call_block, dict) else None
    call_id_str = str(call_id) if call_id is not None else None
    logger.info(
        "WEBHOOK action=process call_id=%s transcript_preview=%r",
        call_id_str,
        (transcript[:200] + "…") if len(transcript) > 200 else transcript,
    )

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


_frontend_dir = _REPO_ROOT / "frontend"
_UI_INDEX = _frontend_dir / "index.html"


def _inject_ui_index() -> str:
    """Serve SPA HTML with Vapi meta tags filled from env (never commit real keys in the file)."""
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
