"""Rule-based gap detection for ambient session — one short TTS line, language-aware."""

from __future__ import annotations

import re


def _norm_session_language(session_language: str | None) -> str:
    if not session_language:
        return "hi"
    s = session_language.strip().lower()
    if s.startswith("en") or s in ("2", "english"):
        return "en"
    return "hi"


def suggest_gap_prompt(transcript: str, session_language: str | None = None) -> str | None:
    """
    If critical fields seem missing from the raw transcript, return one short line for TTS (else None).
    ``session_language`` should match the ANM's keypad choice (``en`` / ``hi`` / …); English vs Hindi phrasing.
    """
    raw = transcript.strip()
    if len(raw) < 20:
        return None
    t = raw.lower()
    lang = _norm_session_language(session_language)

    def has_bp() -> bool:
        if re.search(r"\b\d{2,3}\s*[/\s-]\s*\d{2,3}\b", raw):
            return True
        if re.search(r"\bbp\b", t):
            return True
        if "blood pressure" in t:
            return True
        if re.search(r"\bby\s*\d{2,3}\b", t):
            return True
        return False

    def has_hb() -> bool:
        if re.search(r"\bhb\b", t):
            return True
        if re.search(r"\bhemoglobin\b", t):
            return True
        if re.search(r"\b\d+(\.\d+)?\s*(g\s*/\s*d\s*l|g/dl)\b", t):
            return True
        return False

    def has_weight() -> bool:
        return bool(re.search(r"\b\d{1,3}\s*kg\b", t)) or "weight" in t or "vajan" in t or "wazan" in t

    def has_pregnancy_months() -> bool:
        if re.search(r"\b\d+\s*mahine?\b", t):
            return True
        if re.search(r"\b\d+\s*months?\s+pregnant", t):
            return True
        if re.search(r"\b\d+\s*weeks?\s+pregnant", t):
            return True
        if re.search(r"\b\d+\s*maas\b", t):
            return True
        return False

    def has_complaint() -> bool:
        keys = (
            "shikayat", "complaint", "dard", "fever", "bukhar", "khansi", "cough",
            "sir dard", "headache", "opd", "checkup", "problem", "takleef",
        )
        return any(k in t for k in keys)

    def has_vaccine_hint() -> bool:
        keys = (
            "vaccine", "teeka", "tikka", "dpt", "bcg", "measles", "hep", "polio",
            "immunization", "booster", "pentavalent",
        )
        return any(k in t for k in keys)

    pregnancy = any(
        x in t
        for x in (
            "pregnant",
            "garbh",
            "garbhavati",
            " anc",
            "anc ",
            "mahine pregnant",
            "months pregnant",
            "pregnancy",
        )
    ) or ("mahine" in t and ("pregnant" in t or "garbh" in t or "mahila" in t))

    child = any(
        x in t
        for x in (
            "bachcha",
            "bacha",
            "baby",
            "months ka",
            "mahine ka",
            "saal ka",
            "infant",
        )
    )

    # Internal slot keys for bilingual formatting
    slots: list[str] = []

    if pregnancy:
        if not has_pregnancy_months():
            slots.append("preg_months")
        if not has_bp():
            slots.append("bp")
        if not has_hb():
            slots.append("hb")
    elif child:
        if not has_weight():
            slots.append("weight")
        if not has_vaccine_hint():
            slots.append("vaccine")
    else:
        if not has_bp() and not has_complaint():
            slots.extend(["bp", "complaint"])
        elif not has_bp():
            slots.append("bp")
        elif not has_complaint() and not has_hb():
            slots.append("complaint")

    if not slots:
        return None

    labels_hi: dict[str, str] = {
        "preg_months": "garbh ke kitne mahine",
        "bp": "BP",
        "hb": "Hb",
        "weight": "weight",
        "vaccine": "vaccine/teeka",
        "complaint": "mukhya shikayat",
    }
    labels_en: dict[str, str] = {
        "preg_months": "months of pregnancy",
        "bp": "blood pressure",
        "hb": "hemoglobin",
        "weight": "weight",
        "vaccine": "vaccination details",
        "complaint": "main complaint",
    }

    if lang == "en":
        parts = [labels_en[s] for s in slots]
        if len(parts) == 1:
            return f"Please confirm {parts[0]}."
        if len(parts) == 2:
            return f"Please confirm {parts[0]} and {parts[1]}."
        return "Please confirm " + ", ".join(parts[:-1]) + f", and {parts[-1]}."

    parts_hi = [labels_hi[s] for s in slots]
    if len(parts_hi) == 1:
        return f"{parts_hi[0]} confirm karein."
    return " aur ".join(parts_hi[:-1]) + " aur " + parts_hi[-1] + " confirm karein."
