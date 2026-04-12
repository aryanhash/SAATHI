"""Clinical risk triage and emergency detection from extracted visit fields."""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any


def _parse_bp_string(bp: str | None) -> tuple[float | None, float | None]:
    if not bp or not isinstance(bp, str):
        return None, None
    m = re.search(r"(\d+(?:\.\d+)?)\s*/\s*(\d+(?:\.\d+)?)", bp.strip())
    if not m:
        return None, None
    return float(m.group(1)), float(m.group(2))


def _systolic_diastolic(data: dict[str, Any]) -> tuple[float | None, float | None]:
    sys = data.get("blood_pressure_systolic")
    dia = data.get("blood_pressure_diastolic")
    if sys is not None:
        try:
            s = float(sys)
        except (TypeError, ValueError):
            s = None
    else:
        s = None
    if dia is not None:
        try:
            d = float(dia)
        except (TypeError, ValueError):
            d = None
    else:
        d = None
    if s is None or d is None:
        ps, pd = _parse_bp_string(data.get("blood_pressure"))
        if s is None and ps is not None:
            s = ps
        if d is None and pd is not None:
            d = pd
    return s, d


def _safe_float(val: Any) -> float | None:
    if val is None:
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


EMERGENCY_KEYWORDS = {
    "convulsion", "convulsions", "seizure", "seizures",
    "unconscious", "unconsciousness", "behosh",
    "uncontrolled bleeding", "heavy bleeding", "hemorrhage",
    "severe breathlessness", "breathless", "saans nahi",
    "chest pain", "seene mein dard",
    "blurred vision", "nazar dhundhli",
    "severe headache", "bahut tez sir dard",
    "swelling face", "haath pair suj gaye",
    "reduced fetal movement", "bachcha hil nahi raha",
    "leaking fluid", "paani aa raha",
    "cord prolapse",
}


def detect_emergency(data: dict[str, Any]) -> dict[str, Any]:
    """
    Returns emergency assessment:
    - is_emergency: bool
    - severity: "critical" | "urgent" | "none"
    - reasons: list[str]
    - recommended_action: str
    """
    reasons: list[str] = []
    severity = "none"

    sys, dia = _systolic_diastolic(data)
    hb = _safe_float(data.get("hemoglobin_g_dl"))
    temp = _safe_float(data.get("temperature_f"))
    spo2 = _safe_float(data.get("spo2_percent"))
    sugar = _safe_float(data.get("random_blood_sugar_mg_dl"))
    pulse = _safe_float(data.get("pulse_rate"))

    # --- CRITICAL (life-threatening) ---
    if sys is not None and sys >= 180:
        reasons.append(f"Hypertensive crisis: BP {sys}/{dia or '?'}")
        severity = "critical"
    if sys is not None and dia is not None and sys >= 160 and dia >= 110:
        reasons.append(f"Severe pre-eclampsia range: BP {sys}/{dia}")
        severity = "critical"
    if hb is not None and hb < 5:
        reasons.append(f"Severe anemia: Hb {hb} g/dL (life-threatening)")
        severity = "critical"
    if temp is not None and temp >= 104:
        reasons.append(f"Very high fever: {temp}°F")
        severity = "critical"
    if spo2 is not None and spo2 < 90:
        reasons.append(f"Severe hypoxia: SpO2 {spo2}%")
        severity = "critical"
    if sugar is not None and sugar > 500:
        reasons.append(f"Diabetic emergency: RBS {sugar} mg/dL")
        severity = "critical"
    if pulse is not None and (pulse < 40 or pulse > 150):
        reasons.append(f"Dangerous pulse rate: {pulse} bpm")
        severity = "critical"

    emergency_signs = data.get("emergency_signs") or []
    if isinstance(emergency_signs, list) and emergency_signs:
        reasons.extend([f"Emergency sign: {s}" for s in emergency_signs])
        severity = "critical"

    notes_blob = " ".join([
        str(data.get("notes") or ""),
        str(data.get("symptoms") or ""),
        str(data.get("diagnosis") or ""),
    ]).lower()
    for kw in EMERGENCY_KEYWORDS:
        if kw in notes_blob:
            reasons.append(f"Emergency keyword detected: {kw}")
            if severity != "critical":
                severity = "critical"

    referral = data.get("referral_needed")
    if referral is True and data.get("referral_reason"):
        ref_reason = str(data.get("referral_reason", "")).lower()
        critical_refs = ["eclampsia", "hemorrhage", "bleeding", "unconscious", "convulsion"]
        if any(c in ref_reason for c in critical_refs):
            reasons.append(f"Critical referral: {data['referral_reason']}")
            severity = "critical"

    # --- URGENT (needs attention within hours) ---
    if severity != "critical":
        if sys is not None and sys >= 160:
            reasons.append(f"Very high BP: {sys}/{dia or '?'}")
            severity = "urgent"
        if hb is not None and hb < 7:
            reasons.append(f"Severe anemia: Hb {hb} g/dL")
            severity = "urgent"
        if temp is not None and temp >= 102:
            reasons.append(f"High fever: {temp}°F")
            severity = "urgent"
        if spo2 is not None and spo2 < 94:
            reasons.append(f"Low SpO2: {spo2}%")
            severity = "urgent"
        if sugar is not None and sugar > 300:
            reasons.append(f"Very high sugar: RBS {sugar} mg/dL")
            severity = "urgent"

    is_emergency = severity in ("critical", "urgent")

    if severity == "critical":
        action = "IMMEDIATE: Call 108/ambulance. Refer to nearest PHC/CHC/District Hospital. Do not delay."
    elif severity == "urgent":
        action = "URGENT: Refer to PHC within 2 hours. Monitor vitals. Keep patient stable."
    else:
        action = "Routine follow-up as scheduled."

    return {
        "is_emergency": is_emergency,
        "severity": severity,
        "reasons": reasons,
        "recommended_action": action,
        "detected_at": datetime.now(timezone.utc).isoformat(),
    }


def calculate_risk(data: dict[str, Any]) -> dict[str, Any]:
    """
    Risk levels: red / amber / green.
    Also runs emergency detection and attaches it.
    """
    out = dict(data)
    sys, _ = _systolic_diastolic(out)
    hb = _safe_float(out.get("hemoglobin_g_dl"))
    sugar = _safe_float(out.get("random_blood_sugar_mg_dl"))
    spo2 = _safe_float(out.get("spo2_percent"))
    temp = _safe_float(out.get("temperature_f"))

    risk = "green"
    risk_factors: list[str] = []

    # Red triggers
    if sys is not None and sys > 140:
        risk = "red"
        risk_factors.append(f"High BP systolic: {sys}")
    if hb is not None and hb < 7:
        risk = "red"
        risk_factors.append(f"Severe anemia: Hb {hb}")
    if sugar is not None and sugar > 300:
        risk = "red"
        risk_factors.append(f"Very high sugar: {sugar}")
    if spo2 is not None and spo2 < 92:
        risk = "red"
        risk_factors.append(f"Low oxygen: SpO2 {spo2}%")
    if temp is not None and temp >= 103:
        risk = "red"
        risk_factors.append(f"High fever: {temp}°F")

    emergency_signs = out.get("emergency_signs")
    if isinstance(emergency_signs, list) and emergency_signs:
        risk = "red"
        risk_factors.append(f"Emergency signs: {', '.join(emergency_signs)}")

    # Amber triggers (only if not already red)
    if risk != "red":
        if sys is not None and 130 <= sys <= 140:
            risk = "amber"
            risk_factors.append(f"Borderline BP: {sys}")
        if hb is not None and 7 <= hb < 9:
            risk = "amber"
            risk_factors.append(f"Moderate anemia: Hb {hb}")
        if sugar is not None and 200 <= sugar <= 300:
            risk = "amber"
            risk_factors.append(f"High sugar: {sugar}")

    out["risk_level"] = risk
    out["risk_factors"] = risk_factors

    emergency = detect_emergency(out)
    out["emergency"] = emergency

    return out
