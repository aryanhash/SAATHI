"""Map extracted patient records to ANMOL / U-WIN / NCD style pre-fill payloads."""

from __future__ import annotations

from typing import Any, Literal

VisitProgram = Literal["ANC", "IMMUNIZATION", "NCD"]


def infer_visit_program(data: dict[str, Any]) -> VisitProgram:
    """Route ANC → ANMOL, Immunization → U-WIN, else NCD."""
    name = (data.get("patient_name") or "") or ""
    notes = (data.get("notes") or "") or ""
    blob = f"{name} {notes}".lower()
    if data.get("pregnancy_months") is not None:
        try:
            if float(data["pregnancy_months"]) > 0:
                return "ANC"
        except (TypeError, ValueError):
            pass
    if "baby" in blob or "immunization" in blob or "vaccine" in blob or "bcg" in blob or "opv" in blob:
        return "IMMUNIZATION"
    return "NCD"


def map_to_anmol(data: dict[str, Any]) -> dict[str, Any]:
    return {
        "antenatal_registration": {
            "woman_name": data.get("patient_name"),
            "age_years": data.get("age_years"),
            "gestation_months": data.get("pregnancy_months"),
            "blood_pressure": data.get("blood_pressure"),
            "hemoglobin_g_dl": data.get("hemoglobin_g_dl"),
            "high_risk_flag": data.get("risk_level") == "red",
        },
        "visit_summary": data.get("notes"),
    }


def map_to_uwin(data: dict[str, Any]) -> dict[str, Any]:
    return {
        "child_name": data.get("patient_name"),
        "age_months_approx": None,
        "next_due_vaccines": ["BCG", "OPV-0"] if data.get("patient_name") else [],
        "last_visit_note": data.get("notes"),
    }


def map_to_ncd(data: dict[str, Any]) -> dict[str, Any]:
    return {
        "patient_name": data.get("patient_name"),
        "age_years": data.get("age_years"),
        "blood_pressure": data.get("blood_pressure"),
        "blood_pressure_systolic": data.get("blood_pressure_systolic"),
        "blood_pressure_diastolic": data.get("blood_pressure_diastolic"),
        "random_blood_sugar_mg_dl": data.get("random_blood_sugar_mg_dl"),
        "bmi": data.get("bmi"),
        "risk_level": data.get("risk_level"),
    }


def build_portal_prefill(data: dict[str, Any]) -> dict[str, Any]:
    program = infer_visit_program(data)
    return {
        "visit_program": program,
        "anmol": map_to_anmol(data),
        "uwin": map_to_uwin(data),
        "ncd": map_to_ncd(data),
    }
