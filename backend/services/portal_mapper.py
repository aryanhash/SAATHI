"""Map extracted patient records to realistic ANMOL/RCH, U-WIN, NCD portal pre-fill payloads.

Field names and structure mirror India's actual government health IT portals:
- ANMOL (ANM Online) / RCH (Reproductive Child Health) portal
- U-WIN (Universal Immunization) portal
- NCD (Non-Communicable Disease) screening app under NPCDCS
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

VisitProgram = Literal["ANC", "PNC", "IMMUNIZATION", "NCD", "GENERAL"]

_INDIA_NCD_SCREENING_FIELDS = [
    "hypertension", "diabetes", "oral_cancer", "breast_cancer", "cervical_cancer",
]

IMMUNIZATION_SCHEDULE_INDIA = {
    "birth": ["BCG", "OPV-0", "Hep-B Birth"],
    "6_weeks": ["OPV-1", "Pentavalent-1", "RVV-1", "fIPV-1", "PCV-1"],
    "10_weeks": ["OPV-2", "Pentavalent-2", "RVV-2"],
    "14_weeks": ["OPV-3", "Pentavalent-3", "RVV-3", "fIPV-2", "PCV-2"],
    "9_months": ["MR-1", "JE-1", "PCV-Booster", "Vitamin A-1"],
    "16_months": ["MR-2", "JE-2", "DPT-B1", "OPV-Booster"],
    "5_years": ["DPT-B2"],
    "10_years": ["Td"],
    "16_years": ["Td"],
}


def infer_visit_program(data: dict[str, Any]) -> VisitProgram:
    explicit = (data.get("visit_type") or "").strip().upper()
    if explicit in ("ANC", "PNC", "NCD", "IMMUNIZATION"):
        return explicit  # type: ignore[return-value]

    name = (data.get("patient_name") or "").lower()
    notes = (data.get("notes") or "").lower()
    blob = f"{name} {notes}"

    if data.get("pregnancy_months") is not None:
        try:
            if float(data["pregnancy_months"]) > 0:
                return "ANC"
        except (TypeError, ValueError):
            pass

    vaccines = data.get("vaccines_given") or data.get("vaccines_due") or []
    if vaccines or "baby" in blob or "immunization" in blob or "vaccine" in blob or "bcg" in blob or "opv" in blob:
        return "IMMUNIZATION"

    if data.get("child_age_months") is not None or "pnc" in blob:
        return "PNC"

    if data.get("random_blood_sugar_mg_dl") or data.get("fasting_blood_sugar_mg_dl") or data.get("bmi"):
        return "NCD"

    age = data.get("age_years")
    if age is not None:
        try:
            if float(age) >= 30:
                return "NCD"
        except (TypeError, ValueError):
            pass

    return "GENERAL"


def _infer_immunization_due(data: dict[str, Any]) -> list[str]:
    """Infer next due vaccines based on child's age in months."""
    child_months = data.get("child_age_months")
    if child_months is None:
        age_years = data.get("age_years")
        if age_years is not None:
            try:
                ym = float(age_years)
                if ym < 2:
                    child_months = int(ym * 12)
            except (TypeError, ValueError):
                pass

    if child_months is None:
        return data.get("vaccines_due") or ["BCG", "OPV-0", "Hep-B Birth"]

    try:
        m = float(child_months)
    except (TypeError, ValueError):
        return data.get("vaccines_due") or []

    given = set(v.strip().upper() for v in (data.get("vaccines_given") or []))

    due: list[str] = []
    schedule_map = [
        (0, "birth"), (1.5, "6_weeks"), (2.5, "10_weeks"), (3.5, "14_weeks"),
        (9, "9_months"), (16, "16_months"), (60, "5_years"), (120, "10_years"),
    ]
    for threshold, slot in schedule_map:
        if m >= threshold:
            for v in IMMUNIZATION_SCHEDULE_INDIA[slot]:
                if v.strip().upper() not in given:
                    due.append(v)

    return due or (data.get("vaccines_due") or [])


def map_to_anmol_rch(data: dict[str, Any]) -> dict[str, Any]:
    """ANMOL / RCH Portal realistic field structure."""
    visit_prog = infer_visit_program(data)
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    result: dict[str, Any] = {
        "rch_id": data.get("patient_id"),
        "registration_date": now,
        "facility_type": data.get("facility_type", "Sub Centre"),
        "woman_name": data.get("patient_name"),
        "husband_name": data.get("husband_name"),
        "age_years": data.get("age_years"),
        "phone": data.get("phone"),
        "address": data.get("location"),
        "blood_group": data.get("blood_group"),
    }

    if visit_prog == "ANC":
        result["anc_registration"] = {
            "lmp_date": data.get("lmp_date"),
            "edd_date": data.get("edd_date"),
            "gestation_weeks": (data.get("pregnancy_months") or 0) * 4 if data.get("pregnancy_months") else None,
            "gestation_months": data.get("pregnancy_months"),
            "gravida": data.get("gravida"),
            "para": data.get("para"),
            "high_risk": data.get("risk_level") == "red",
            "risk_factors": data.get("risk_factors", []),
        }
        result["anc_visit"] = {
            "visit_date": now,
            "weight_kg": data.get("weight_kg"),
            "height_cm": data.get("height_cm"),
            "blood_pressure": data.get("blood_pressure"),
            "bp_systolic": data.get("blood_pressure_systolic"),
            "bp_diastolic": data.get("blood_pressure_diastolic"),
            "hemoglobin_g_dl": data.get("hemoglobin_g_dl"),
            "urine_albumin": data.get("urine_albumin"),
            "urine_sugar": data.get("urine_sugar"),
            "fetal_heart_sound": "present",
            "tt_dose": data.get("tt_doses"),
            "ifa_tablets": data.get("ifa_tablets_given"),
            "calcium_tablets": data.get("calcium_tablets_given"),
            "danger_signs_counseling": bool(data.get("counseling_done")),
            "birth_preparedness_counseling": bool(data.get("counseling_done")),
            "institutional_delivery_counseling": True,
            "referral_needed": data.get("referral_needed", False),
            "referral_facility": data.get("referral_facility"),
            "referral_reason": data.get("referral_reason"),
            "next_visit_date": data.get("follow_up_date"),
        }
    elif visit_prog == "PNC":
        result["pnc_visit"] = {
            "visit_date": now,
            "mother_bp": data.get("blood_pressure"),
            "mother_temperature_f": data.get("temperature_f"),
            "mother_hemoglobin": data.get("hemoglobin_g_dl"),
            "breastfeeding_initiated": data.get("breastfeeding_status"),
            "newborn_weight_kg": data.get("child_weight_kg"),
            "danger_signs": data.get("emergency_signs", []),
            "medicines": data.get("medicines_given", []),
            "referral_needed": data.get("referral_needed", False),
        }

    result["visit_summary"] = data.get("notes")
    return result


def map_to_uwin(data: dict[str, Any]) -> dict[str, Any]:
    """U-WIN (Universal Immunization) portal realistic field structure."""
    due_vaccines = _infer_immunization_due(data)
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    return {
        "beneficiary_id": data.get("patient_id"),
        "child_name": data.get("patient_name"),
        "mother_name": data.get("mother_name"),
        "father_name": data.get("father_name") or data.get("husband_name"),
        "date_of_birth": data.get("date_of_birth"),
        "gender": data.get("gender"),
        "age_months": data.get("child_age_months"),
        "weight_kg": data.get("child_weight_kg") or data.get("weight_kg"),
        "address": data.get("location"),
        "phone": data.get("phone"),
        "vaccination_session": {
            "session_date": now,
            "session_site": data.get("location") or "Sub Centre",
            "vaccinator_name": None,
            "vaccines_administered": data.get("vaccines_given", []),
            "adverse_event_observed": False,
        },
        "immunization_card": {
            "vaccines_completed": data.get("vaccines_given", []),
            "vaccines_due": due_vaccines,
            "next_due_date": data.get("follow_up_date"),
        },
        "last_visit_note": data.get("notes"),
    }


def map_to_ncd(data: dict[str, Any]) -> dict[str, Any]:
    """NCD Screening (NPCDCS / CBAC) portal realistic field structure."""
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    sys = data.get("blood_pressure_systolic")
    sugar = data.get("random_blood_sugar_mg_dl") or data.get("fasting_blood_sugar_mg_dl")

    hypertension_status = "normal"
    if sys is not None:
        try:
            s = float(sys)
            if s >= 140:
                hypertension_status = "hypertensive"
            elif s >= 130:
                hypertension_status = "pre-hypertensive"
        except (TypeError, ValueError):
            pass

    diabetes_status = "normal"
    if sugar is not None:
        try:
            sv = float(sugar)
            if sv >= 200:
                diabetes_status = "diabetic"
            elif sv >= 140:
                diabetes_status = "pre-diabetic"
        except (TypeError, ValueError):
            pass

    bmi = data.get("bmi")
    if bmi is None and data.get("weight_kg") and data.get("height_cm"):
        try:
            w = float(data["weight_kg"])
            h = float(data["height_cm"]) / 100
            bmi = round(w / (h * h), 1)
        except (TypeError, ValueError, ZeroDivisionError):
            pass

    bmi_category = None
    if bmi is not None:
        try:
            b = float(bmi)
            if b < 18.5:
                bmi_category = "underweight"
            elif b < 25:
                bmi_category = "normal"
            elif b < 30:
                bmi_category = "overweight"
            else:
                bmi_category = "obese"
        except (TypeError, ValueError):
            pass

    return {
        "screening_id": data.get("patient_id"),
        "screening_date": now,
        "patient_name": data.get("patient_name"),
        "age_years": data.get("age_years"),
        "gender": data.get("gender"),
        "phone": data.get("phone"),
        "address": data.get("location"),
        "vitals": {
            "blood_pressure": data.get("blood_pressure"),
            "bp_systolic": data.get("blood_pressure_systolic"),
            "bp_diastolic": data.get("blood_pressure_diastolic"),
            "pulse_rate": data.get("pulse_rate"),
            "temperature_f": data.get("temperature_f"),
            "spo2_percent": data.get("spo2_percent"),
            "weight_kg": data.get("weight_kg"),
            "height_cm": data.get("height_cm"),
            "bmi": bmi,
            "bmi_category": bmi_category,
            "waist_circumference_cm": data.get("waist_circumference_cm"),
        },
        "lab_results": {
            "random_blood_sugar_mg_dl": data.get("random_blood_sugar_mg_dl"),
            "fasting_blood_sugar_mg_dl": data.get("fasting_blood_sugar_mg_dl"),
            "hemoglobin_g_dl": data.get("hemoglobin_g_dl"),
        },
        "screening_results": {
            "hypertension": hypertension_status,
            "diabetes": diabetes_status,
            "oral_cancer_screening": data.get("oral_cancer_screening"),
            "breast_cancer_screening": data.get("breast_cancer_screening"),
            "cervical_cancer_screening": data.get("cervical_cancer_screening"),
        },
        "risk_level": data.get("risk_level"),
        "risk_factors": data.get("risk_factors", []),
        "medicines_prescribed": data.get("medicines_given", []),
        "counseling": data.get("counseling_done", []),
        "referral": {
            "needed": data.get("referral_needed", False),
            "facility": data.get("referral_facility"),
            "reason": data.get("referral_reason"),
        },
        "follow_up_date": data.get("follow_up_date"),
        "symptoms": data.get("symptoms", []),
        "diagnosis": data.get("diagnosis"),
    }


def build_portal_prefill(data: dict[str, Any]) -> dict[str, Any]:
    program = infer_visit_program(data)
    return {
        "visit_program": program,
        "anmol": map_to_anmol_rch(data),
        "uwin": map_to_uwin(data),
        "ncd": map_to_ncd(data),
    }
