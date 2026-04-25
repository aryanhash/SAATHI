"""Twilio SMS + WhatsApp (optional). Daily follow-up SMS and end-of-day register."""

from __future__ import annotations

import logging
import os
import re
import threading
import time
from datetime import date, datetime
from typing import Any
from urllib.parse import quote
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


def _twilio_sms_send_ready() -> bool:
    return bool(
        _twilio_configured()
        and (os.environ.get("TWILIO_SMS_FROM") or "").strip()
    )


def _twilio_whatsapp_send_ready() -> bool:
    return bool(
        _twilio_configured()
        and (os.environ.get("TWILIO_WHATSAPP_FROM") or "").strip()
    )


def _env_truthy(name: str) -> bool:
    v = (os.environ.get(name) or "").strip().lower()
    return v in ("1", "true", "yes", "on")


def notifications_demo_mode() -> bool:
    """
    - Set ``SAATHI_NOTIFICATIONS_DEMO=1`` to force demo (log + wa.me) even with Twilio.
    - Or leave Twilio unset and set ``SAATHI_DEMO_PHONE`` — no carrier; SMS is logged only;
      daily register returns a click-to-chat URL for that number.
    """
    if _env_truthy("SAATHI_NOTIFICATIONS_DEMO"):
        return True
    if not _twilio_configured():
        return bool((os.environ.get("SAATHI_DEMO_PHONE") or "").strip())
    return False


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


def _normalize_whatsapp_to(raw: str | None) -> str | None:
    """
    Normalize a WhatsApp recipient/sender into Twilio format: ``whatsapp:+E164``.
    Accepts: ``+91...``, ``9198...``, ``whatsapp:+91...``, or a 10-digit Indian mobile.
    """
    if not raw:
        return None
    s = str(raw).strip()
    if not s:
        return None
    if s.lower().startswith("whatsapp:"):
        s2 = s.split(":", 1)[1].strip()
        p = _normalize_phone_e164(s2)
        return f"whatsapp:{p}" if p else None
    p = _normalize_phone_e164(s)
    return f"whatsapp:{p}" if p else None


def _demo_e164_or_none() -> str | None:
    return _normalize_phone_e164(os.environ.get("SAATHI_DEMO_PHONE"))


def _wa_me_url(phone_e164: str, text: str) -> str:
    """Click-to-chat link (opens WhatsApp on a device with the draft message). No Twilio."""
    digits = re.sub(r"\D", "", phone_e164)
    if not digits:
        return ""
    t = text[:3000]
    return f"https://wa.me/{digits}?text={quote(t, safe='')}"


def _log_demo_sms(
    intended_patient_to: str,
    body: str,
    *,
    patient_id: str,
    demo_recipient: str,
) -> None:
    logger.info(
        "notifications.DEMO_SMS (not sent — no Twilio) digest_recipient=%s patient_id=%s would_send_to=%s\n%s",
        demo_recipient,
        patient_id,
        intended_patient_to,
        body,
    )


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
    In demo mode (no Twilio, SAATHI_DEMO_PHONE set) messages are logged only, not sent.
    """
    today = datetime.now(IST).date()
    today_s = today.isoformat()
    patients = get_all_patients()
    sent = 0
    skipped = 0
    errors: list[str] = []

    if notifications_demo_mode():
        demo_to = _demo_e164_or_none()
        if not demo_to:
            logger.info("notifications.follow_up skip demo mode but SAATHI_DEMO_PHONE missing")
            return {
                "ok": False,
                "reason": "demo_phone_required",
                "today": today_s,
                "detail": "Set SAATHI_DEMO_PHONE=+91XXXXXXXXXX in .env for demo, or add Twilio for real SMS.",
            }
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
            _log_demo_sms(phone, msg, patient_id=str(pid), demo_recipient=demo_to)
            update_patient_fields(str(pid), {"follow_up_reminder_sent_for": str(fu_key)})
            sent += 1
            logger.info("notifications.follow_up demo_logged patient_id=%s", pid)
        return {
            "ok": True,
            "demo": True,
            "demo_recipient": demo_to,
            "today": today_s,
            "simulated": sent,
            "skipped": skipped,
            "errors": errors,
            "note": "SMS not delivered without Twilio — see server log lines notifications.DEMO_SMS.",
        }

    if not _twilio_sms_send_ready():
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
    """
    Legacy single-recipient register sender.
    Builds today's register from Qdrant and sends to ``SAATHI_ANM_WHATSAPP_TO`` (env).
    """
    today = datetime.now(IST).date()
    body = build_daily_register_text(get_all_patients(), today)

    if notifications_demo_mode():
        p = _demo_e164_or_none()
        if not p:
            raise RuntimeError(
                "Demo mode: set SAATHI_DEMO_PHONE=+91XXXXXXXXXX (E.164), or add Twilio + SAATHI_ANM_WHATSAPP_TO for real WhatsApp.",
            )
        url = _wa_me_url(p, body)
        logger.info(
            "notifications.DEMO_WA (not sent via Twilio). Open on your phone: %s\n-----\n%s\n-----",
            url,
            body[:2000],
        )
        return {
            "ok": True,
            "demo": True,
            "to": f"whatsapp:{p}",
            "sid": "demo",
            "chars": len(body),
            "whatsapp_open_url": url,
            "note": "Use whatsapp_open_url in a browser to open your WhatsApp with this draft (no API send).",
        }

    to = _normalize_whatsapp_to(os.environ.get("SAATHI_ANM_WHATSAPP_TO"))
    if not to:
        raise RuntimeError("Set SAATHI_ANM_WHATSAPP_TO e.g. whatsapp:+9198xxxxxxxx")
    out = send_whatsapp(to, body)
    return {"ok": True, "to": to, "sid": out.get("sid"), "chars": len(body)}


def send_daily_register_whatsapp_all() -> dict[str, Any]:
    """
    Per-ANM daily register:
    - Groups patients by ``registered_by_anm_whatsapp`` stored on each patient record.
    - Sends one WhatsApp register per ANM containing only patients they registered.
    """
    today = datetime.now(IST).date()
    patients = get_all_patients()

    groups: dict[str, list[dict[str, Any]]] = {}
    unassigned = 0
    for p in patients:
        to = _normalize_whatsapp_to(p.get("registered_by_anm_whatsapp"))
        if not to:
            unassigned += 1
            continue
        groups.setdefault(to, []).append(p)

    if not groups:
        raise RuntimeError(
            "No ANM WhatsApp numbers found on patient records. "
            "Ensure the UI sends anm_whatsapp when saving visits so patients get tagged with registered_by_anm_whatsapp."
        )

    # Demo mode: generate click-to-chat drafts (no Twilio send).
    if notifications_demo_mode():
        demo_to = _demo_e164_or_none()
        if not demo_to:
            raise RuntimeError(
                "Demo mode: set SAATHI_DEMO_PHONE=+91XXXXXXXXXX (E.164), or configure Twilio WhatsApp for real sending.",
            )
        drafts: list[dict[str, Any]] = []
        for intended_to, plist in sorted(groups.items(), key=lambda kv: kv[0]):
            body = build_daily_register_text(plist, today)
            url = _wa_me_url(demo_to, body)
            drafts.append(
                {
                    "intended_to": intended_to,
                    "demo_to": f"whatsapp:{demo_to}",
                    "whatsapp_open_url": url,
                    "chars": len(body),
                    "patient_total_for_anm": len(plist),
                }
            )
        return {
            "ok": True,
            "demo": True,
            "date": today.isoformat(),
            "anm_count": len(groups),
            "unassigned_patients": unassigned,
            "drafts": drafts,
            "note": "Demo mode: open each whatsapp_open_url to send manually (no API send).",
        }

    # Real Twilio send
    results: list[dict[str, Any]] = []
    errors: list[str] = []
    for to, plist in sorted(groups.items(), key=lambda kv: kv[0]):
        body = build_daily_register_text(plist, today)
        try:
            out = send_whatsapp(to, body)
            results.append(
                {
                    "to": to,
                    "sid": out.get("sid"),
                    "chars": len(body),
                    "patient_total_for_anm": len(plist),
                }
            )
        except Exception as e:
            errors.append(f"{to}: {e}")

    return {
        "ok": len(errors) == 0,
        "date": today.isoformat(),
        "anm_count": len(groups),
        "sent": len(results),
        "unassigned_patients": unassigned,
        "results": results,
        "errors": errors,
    }


def notification_status() -> dict[str, Any]:
    demo = notifications_demo_mode()
    demo_phone = _demo_e164_or_none()
    return {
        "demo_mode": demo,
        "demo_phone_set": bool(demo_phone),
        "twilio_sms_ready": _twilio_sms_send_ready(),
        # WhatsApp sender readiness (per-ANM recipients are stored on patient records).
        "twilio_whatsapp_sender_ready": _twilio_whatsapp_send_ready(),
        # Legacy single-recipient config (still supported).
        "twilio_whatsapp_legacy_ready": bool(_twilio_whatsapp_send_ready() and _normalize_whatsapp_to(os.environ.get("SAATHI_ANM_WHATSAPP_TO"))),
        "anm_whatsapp_to_set": bool(_normalize_whatsapp_to(os.environ.get("SAATHI_ANM_WHATSAPP_TO"))),
        "notifications_usable": bool(
            (demo and demo_phone)
            or _twilio_sms_send_ready()
            or (
                _twilio_whatsapp_send_ready()
            )
        ),
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
