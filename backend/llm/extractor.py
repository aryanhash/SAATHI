"""Extract structured patient fields from voice transcripts (Gemini on Render, Ollama locally)."""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any

import requests

logger = logging.getLogger(__name__)

_OLLAMA_BASE = os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
OLLAMA_GENERATE_URL = f"{_OLLAMA_BASE}/api/generate"
OLLAMA_MODEL = "qwen3:8b"

# Default must stay on a model Google still offers to new API users (2.0-flash was retired for new keys).
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
GEMINI_URL = (
    f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"
)

REQUEST_TIMEOUT_SEC = 120


def extraction_backend() -> str:
    """Which backend ``extract_patient_data`` will use (``gemini`` or ``ollama``)."""
    return _active_llm_backend()


def _active_llm_backend() -> str:
    """``gemini`` if key + not forced ollama; else ``ollama``."""
    force = (os.environ.get("SAATHI_LLM") or "").strip().lower()
    if force == "ollama":
        return "ollama"
    if force == "gemini":
        return "gemini"
    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    return "gemini" if key else "ollama"


def _gemini_api_key() -> str | None:
    return (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY") or "").strip() or None


def _build_prompt(transcript: str) -> str:
    return (
        "Extract structured healthcare data from this transcript and return ONLY valid JSON "
        "(no markdown fences, no commentary).\n\n"
        "Use this shape; use null for anything not clearly stated:\n"
        "{\n"
        '  "patient_name": string | null,\n'
        '  "age_years": number | null,\n'
        '  "pregnancy_months": number | null,\n'
        '  "blood_pressure": string | null,\n'
        '  "blood_pressure_systolic": number | null,\n'
        '  "blood_pressure_diastolic": number | null,\n'
        '  "hemoglobin_g_dl": number | null,\n'
        '  "notes": string | null\n'
        "}\n\n"
        f"Transcript:\n{transcript.strip()}"
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


def _extract_with_ollama(transcript: str) -> dict[str, Any]:
    prompt = _build_prompt(transcript)
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "format": "json",
    }
    resp = requests.post(OLLAMA_GENERATE_URL, json=payload, timeout=REQUEST_TIMEOUT_SEC)
    resp.raise_for_status()
    body = resp.json()
    raw = body.get("response")
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError("Ollama response missing non-empty 'response' text")
    return _parse_llm_json(raw, source="ollama")


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
    """
    Structured extraction: **Gemini** when ``GEMINI_API_KEY`` or ``GOOGLE_API_KEY`` is set
    (unless ``SAATHI_LLM=ollama``), otherwise **Ollama** at ``OLLAMA_BASE_URL``.

    Raises:
        requests.HTTPError: Upstream HTTP error.
        ValueError: Invalid or empty model output.
        requests.RequestException: Connection / timeout errors.
    """
    backend = _active_llm_backend()
    if backend == "gemini":
        key = _gemini_api_key()
        if not key:
            raise ValueError("SAATHI_LLM=gemini but GEMINI_API_KEY / GOOGLE_API_KEY is missing")
        logger.info("llm.extract backend=gemini model=%s", GEMINI_MODEL)
        return _extract_with_gemini(transcript, key)
    logger.info("llm.extract backend=ollama model=%s", OLLAMA_MODEL)
    return _extract_with_ollama(transcript)
