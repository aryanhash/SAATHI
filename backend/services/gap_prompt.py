"""Rule-based gap detection for ambient session — suggests one short Hindi prompt."""

from __future__ import annotations

import re


def suggest_gap_prompt(transcript: str) -> str | None:
    """
    After ~2s silence, if critical fields seem missing from the raw transcript,
    return a single short line for TTS (else None).
    """
    raw = transcript.strip()
    if len(raw) < 20:
        return None
    t = raw.lower()

    def has_bp() -> bool:
        if re.search(r"\b\d{2,3}\s*[/\s-]\s*\d{2,3}\b", raw):
            return True
        if re.search(r"\bbp\b", t):
            return True
        if "blood pressure" in t:
            return True
        if re.search(r"\bby\s*\d{2,3}\b", t):  # "150 by 95"
            return True
        return False

    def has_hb() -> bool:
        if re.search(r"\bhb\b", t):
            return True
        if re.search(r"hemoglobin", t):
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

    missing: list[str] = []

    if pregnancy:
        if not has_pregnancy_months():
            missing.append("garbh ke kitne mahine")
        if not has_bp():
            missing.append("BP")
        if not has_hb():
            missing.append("Hb")
    elif child:
        if not has_weight():
            missing.append("weight")
        if not has_vaccine_hint():
            missing.append("vaccine/teeka")
    else:
        if not has_bp() and not has_complaint():
            missing.extend(["BP", "mukhya shikayat"])
        elif not has_bp():
            missing.append("BP")
        elif not has_complaint() and not has_hb():
            missing.append("mukhya shikayat")

    if not missing:
        return None

    if len(missing) == 1:
        return f"{missing[0]} confirm karein."

    return " aur ".join(missing[:-1]) + " aur " + missing[-1] + " confirm karein."
