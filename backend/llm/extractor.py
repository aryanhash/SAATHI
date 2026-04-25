"""Extract structured patient fields from voice transcripts via Gemini."""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any

import requests

logger = logging.getLogger(__name__)

GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
GEMINI_URL = (
    f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"
)

REQUEST_TIMEOUT_SEC = 120


def extraction_backend() -> str:
    return "gemini"


def _gemini_api_key() -> str | None:
    return (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY") or "").strip() or None


def _build_prompt(transcript: str) -> str:
    return (
        "You are a medical data extraction system for India's public health workers (ANM/ASHA).\n"
        "Extract ALL structured healthcare data from this transcript and return ONLY valid JSON "
        "(no markdown fences, no commentary).\n\n"
        "Use this shape; use null for anything not clearly stated:\n"
        "{\n"
        '  "patient_name": string | null,\n'
        '  "age_years": number | null,\n'
        '  "gender": "male" | "female" | "other" | null,\n'
        '  "phone": string | null,\n'
        '  "location": string | null,\n'
        '  "visit_type": "ANC" | "PNC" | "immunization" | "NCD" | "family_planning" | "general" | null,\n'
        '  "pregnancy_months": number | null,\n'
        '  "gravida": number | null,\n'
        '  "para": number | null,\n'
        '  "lmp_date": string | null,\n'
        '  "edd_date": string | null,\n'
        '  "blood_pressure": string | null,\n'
        '  "blood_pressure_systolic": number | null,\n'
        '  "blood_pressure_diastolic": number | null,\n'
        '  "hemoglobin_g_dl": number | null,\n'
        '  "weight_kg": number | null,\n'
        '  "height_cm": number | null,\n'
        '  "bmi": number | null,\n'
        '  "temperature_f": number | null,\n'
        '  "pulse_rate": number | null,\n'
        '  "spo2_percent": number | null,\n'
        '  "random_blood_sugar_mg_dl": number | null,\n'
        '  "fasting_blood_sugar_mg_dl": number | null,\n'
        '  "urine_albumin": string | null,\n'
        '  "urine_sugar": string | null,\n'
        '  "blood_group": string | null,\n'
        '  "symptoms": [string] | null,\n'
        '  "diagnosis": string | null,\n'
        '  "medicines_given": [string] | null,\n'
        '  "vaccines_given": [string] | null,\n'
        '  "vaccines_due": [string] | null,\n'
        '  "child_age_months": number | null,\n'
        '  "child_weight_kg": number | null,\n'
        '  "breastfeeding_status": string | null,\n'
        '  "tt_doses": number | null,\n'
        '  "ifa_tablets_given": number | null,\n'
        '  "calcium_tablets_given": number | null,\n'
        '  "referral_needed": boolean | null,\n'
        '  "referral_facility": string | null,\n'
        '  "referral_reason": string | null,\n'
        '  "follow_up_date": string | null,\n'
        '  "counseling_done": [string] | null,\n'
        '  "emergency_signs": [string] | null,\n'
        '  "notes": string | null,\n'
        '  "language_used": string | null\n'
        "}\n\n"
        "IMPORTANT RULES:\n"
        "- For emergency_signs, detect: severe breathlessness, chest pain, uncontrolled bleeding, "
        "convulsions, unconsciousness, very high fever (>104F), pregnancy danger signs "
        "(severe headache, blurred vision, swelling, reduced fetal movement, leaking fluid).\n"
        "- For visit_type, infer from context: pregnant woman → ANC, post-delivery → PNC, "
        "baby/child vaccines → immunization, BP/sugar follow-up → NCD.\n"
        "- Extract numbers carefully from Hindi/English mixed speech.\n"
        "- If BP is mentioned as 'normal', set blood_pressure to 'normal' and leave systolic/diastolic null.\n"
        f"\nTranscript:\n{transcript.strip()}"
    )


def _strip_json_fence(text: str) -> str:
    text = text.strip()
    m = re.match(r"^```(?:json)?\s*\n?(.*?)\n?```\s*$", text, re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return text


def _parse_llm_json(raw: str, *, source: str) -> dict[str, Any]:
    cleaned = _strip_json_fence(raw)
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise ValueError(f"{source}: model did not return valid JSON: {e}") from e
    if not isinstance(data, dict):
        raise ValueError(f"{source}: parsed JSON must be an object at the top level")
    return data


def _extract_with_gemini(transcript: str, api_key: str) -> dict[str, Any]:
    prompt = _build_prompt(transcript)
    body: dict[str, Any] = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.2,
            "responseMimeType": "application/json",
        },
    }
    resp = requests.post(
        GEMINI_URL,
        params={"key": api_key},
        json=body,
        timeout=REQUEST_TIMEOUT_SEC,
    )
    resp.raise_for_status()
    data = resp.json()
    candidates = data.get("candidates") or []
    if not candidates:
        block = data.get("promptFeedback") or data
        raise ValueError(f"Gemini returned no candidates: {block}")
    parts = (candidates[0].get("content") or {}).get("parts") or []
    texts = [p.get("text") for p in parts if isinstance(p, dict) and p.get("text")]
    raw = "".join(texts).strip()
    if not raw:
        raise ValueError("Gemini response missing text in candidates[0].content.parts")
    return _parse_llm_json(raw, source="gemini")


def extract_patient_data(transcript: str) -> dict[str, Any]:
    """Structured extraction via Gemini (set GEMINI_API_KEY or GOOGLE_API_KEY)."""
    key = _gemini_api_key()
    if not key:
        raise ValueError(
            "GEMINI_API_KEY or GOOGLE_API_KEY must be set in .env for extraction."
        )
    logger.info("llm.extract backend=gemini model=%s", GEMINI_MODEL)
    return _extract_with_gemini(transcript, key)
