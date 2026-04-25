"""Twilio SMS + WhatsApp (optional). Daily follow-up SMS and end-of-day register."""

from __future__ import annotations

import logging
import os
import re
import threading
import time
from datetime import date, datetime
from typing import Any
from zoneinfo import ZoneInfo

import requests

from db.qdrant_client import get_all_patients, update_patient_fields

logger = logging.getLogger(__name__)

IST = ZoneInfo("Asia/Kolkata")
_TWILIO_MESSAGES = "https://api.twilio.com/2010-04-01/Accounts/{sid}/Messages.json"


def _twilio_configured() -> bool:
    return bool(
        (os.environ.get("TWILIO_ACCOUNT_SID") or "").strip()
        and (os.environ.get("TWILIO_AUTH_TOKEN") or "").strip()
    )


def _normalize_phone_e164(raw: str | None) -> str | None:
    if not raw:
        return None
    digits = re.sub(r"\D", "", str(raw).strip())
    if not digits:
        return None
    if len(digits) == 10 and digits[0] in "6789":
        return "+91" + digits
    if len(digits) == 12 and digits.startswith("91"):
        return "+" + digits
    if raw.strip().startswith("+") and len(digits) >= 10:
        return "+" + digits
    return None


def send_sms(to_e164: str, body: str) -> dict[str, Any]:
    """Send SMS via Twilio REST. Raises on HTTP error."""
    sid = (os.environ.get("TWILIO_ACCOUNT_SID") or "").strip()
    token = (os.environ.get("TWILIO_AUTH_TOKEN") or "").strip()
    from_num = (os.environ.get("TWILIO_SMS_FROM") or "").strip()
    if not sid or not token or not from_num:
        raise RuntimeError("Twilio SMS not configured (TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_SMS_FROM)")
    url = _TWILIO_MESSAGES.format(sid=sid)
    r = requests.post(
        url,
        auth=(sid, token),
        data={"From": from_num, "To": to_e164, "Body": body[:1500]},
        timeout=30,
    )
    if not r.ok:
        raise RuntimeError(f"Twilio SMS failed: {r.status_code} {r.text[:200]}")
    return r.json()


def send_whatsapp(to_whatsapp: str, body: str) -> dict[str, Any]:
    """
    Send WhatsApp via Twilio. `to_whatsapp` should be like whatsapp:+9198xxxx
    TWILIO_WHATSAPP_FROM must be whatsapp:+1... (sandbox or approved sender).
    """
    sid = (os.environ.get("TWILIO_ACCOUNT_SID") or "").strip()
    token = (os.environ.get("TWILIO_AUTH_TOKEN") or "").strip()
    from_wa = (os.environ.get("TWILIO_WHATSAPP_FROM") or "").strip()
    if not sid or not token or not from_wa:
        raise RuntimeError(
            "Twilio WhatsApp not configured (TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_WHATSAPP_FROM)",
        )
    to = to_whatsapp if to_whatsapp.startswith("whatsapp:") else f"whatsapp:{to_whatsapp}"
    url = _TWILIO_MESSAGES.format(sid=sid)
    r = requests.post(
        url,
        auth=(sid, token),
        data={"From": from_wa, "To": to, "Body": body[:1500]},
        timeout=30,
    )
    if not r.ok:
        raise RuntimeError(f"Twilio WhatsApp failed: {r.status_code} {r.text[:200]}")
    return r.json()


def _parse_iso_date(s: Any) -> date | None:
    if s is None or s == "":
        return None
    text = str(s).strip()[:32]
    try:
        return date.fromisoformat(text[:10])
    except ValueError:
        return None


def _patient_last_visit_date_ist(p: dict[str, Any]) -> date | None:
    lu = p.get("last_updated")
    if not lu:
        return None
    try:
        if isinstance(lu, str):
            return datetime.fromisoformat(lu.replace("Z", "+00:00")).astimezone(IST).date()
    except ValueError:
        return None
    return None


def build_daily_register_text(patients: list[dict[str, Any]], day: date) -> str:
    """Plain-text end-of-day register for WhatsApp."""
    lines: list[str] = [
        f"SAATHI — daily register {day.isoformat()} (IST)",
        f"Total records in system: {len(patients)}",
        "",
    ]
    todays: list[dict[str, Any]] = []
    for p in patients:
        vd = _patient_last_visit_date_ist(p)
        if vd == day:
            todays.append(p)
    lines.append(f"Visits recorded today: {len(todays)}")
    lines.append("")
    if not todays:
        lines.append("(No visits with last_updated matching today.)")
        return "\n".join(lines)
    for i, p in enumerate(todays[:40], 1):
        name = p.get("patient_name") or "—"
        pid = (p.get("patient_id") or "")[:8]
        risk = p.get("risk_level") or "—"
        vt = p.get("visit_type") or "—"
        bp = p.get("blood_pressure") or "—"
        lines.append(f"{i}. {name} | {vt} | risk:{risk} | BP:{bp} | id:{pid}")
    if len(todays) > 40:
        lines.append(f"... and {len(todays) - 40} more.")
    return "\n".join(lines)


def run_follow_up_reminders_for_today() -> dict[str, Any]:
    """
    Send SMS to patients whose follow_up_date is today (IST) and phone is set.
    Sets payload follow_up_reminder_sent_for to the follow_up_date string after success.
    """
    today = datetime.now(IST).date()
    today_s = today.isoformat()
    patients = get_all_patients()
    sent = 0
    skipped = 0
    errors: list[str] = []

    if not _twilio_configured():
        logger.info("notifications.follow_up skip twilio_not_configured")
        return {"ok": False, "reason": "twilio_not_configured", "today": today_s}

    for p in patients:
        pid = p.get("patient_id")
        if not pid:
            continue
        fu = _parse_iso_date(p.get("follow_up_date"))
        if fu is None or fu != today:
            continue
        fu_key = p.get("follow_up_date")
        if str(p.get("follow_up_reminder_sent_for") or "") == str(fu_key):
            skipped += 1
            continue
        phone = _normalize_phone_e164(p.get("phone"))
        if not phone:
            skipped += 1
            continue
        name = (p.get("patient_name") or "Patient").strip()[:40]
        msg = (
            f"Namaskar {name}, aaj ({today.strftime('%d-%b-%Y')}) aapki follow-up visit PHC par scheduled hai. "
            f"Kripya samay par aayein. — SAATHI"
        )
        try:
            send_sms(phone, msg)
            update_patient_fields(str(pid), {"follow_up_reminder_sent_for": str(fu_key)})
            sent += 1
            logger.info("notifications.follow_up sent patient_id=%s to=%s", pid, phone[:6] + "***")
        except Exception as e:
            errors.append(f"{pid}: {e}")
            logger.warning("notifications.follow_up fail patient_id=%s err=%s", pid, e)

    return {"ok": True, "today": today_s, "sent": sent, "skipped": skipped, "errors": errors}


def send_daily_register_whatsapp() -> dict[str, Any]:
    """Build today's register from Qdrant and send to ANM WhatsApp (env)."""
    to = (os.environ.get("SAATHI_ANM_WHATSAPP_TO") or "").strip()
    if not to:
        raise RuntimeError("Set SAATHI_ANM_WHATSAPP_TO e.g. whatsapp:+9198xxxxxxxx")
    today = datetime.now(IST).date()
    body = build_daily_register_text(get_all_patients(), today)
    out = send_whatsapp(to, body)
    return {"ok": True, "to": to, "sid": out.get("sid"), "chars": len(body)}


def notification_status() -> dict[str, Any]:
    return {
        "twilio_sms_ready": bool(
            (os.environ.get("TWILIO_ACCOUNT_SID") or "").strip()
            and (os.environ.get("TWILIO_AUTH_TOKEN") or "").strip()
            and (os.environ.get("TWILIO_SMS_FROM") or "").strip(),
        ),
        "twilio_whatsapp_ready": bool(
            (os.environ.get("TWILIO_ACCOUNT_SID") or "").strip()
            and (os.environ.get("TWILIO_AUTH_TOKEN") or "").strip()
            and (os.environ.get("TWILIO_WHATSAPP_FROM") or "").strip()
            and (os.environ.get("SAATHI_ANM_WHATSAPP_TO") or "").strip(),
        ),
        "anm_whatsapp_to_set": bool((os.environ.get("SAATHI_ANM_WHATSAPP_TO") or "").strip()),
    }


def start_reminder_background_thread() -> None:
    """Hourly wake: run follow-up SMS job (IST day match inside)."""

    def loop() -> None:
        time.sleep(30)  # let Qdrant + app finish boot
        while True:
            try:
                run_follow_up_reminders_for_today()
            except Exception:
                logger.exception("notifications.reminder_loop")
            time.sleep(3600)

    t = threading.Thread(target=loop, name="saathi-follow-up-reminders", daemon=True)
    t.start()
    logger.info("notifications.reminder_thread started")
