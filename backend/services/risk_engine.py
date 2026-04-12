"""Clinical risk triage from extracted visit fields."""

from __future__ import annotations

import re
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


def calculate_risk(data: dict[str, Any]) -> dict[str, Any]:
    """
    Rules:
    - systolic BP > 140 → red
    - Hb < 7 g/dL → red
    - systolic BP 130–140 (inclusive) → amber (only if not already red)
    - else → green
    """
    out = dict(data)
    sys, _ = _systolic_diastolic(out)

    hb_raw = out.get("hemoglobin_g_dl")
    try:
        hb = float(hb_raw) if hb_raw is not None else None
    except (TypeError, ValueError):
        hb = None

    risk = "green"
    if (sys is not None and sys > 140) or (hb is not None and hb < 7):
        risk = "red"
    elif sys is not None and 130 <= sys <= 140:
        risk = "amber"

    out["risk_level"] = risk
    return out
