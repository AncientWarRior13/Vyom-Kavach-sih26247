"""Alerting hooks: dispatch a simulated notification fan-out when a track's
threat level goes CRITICAL, and (optionally) a real webhook call.

IMPORTANT -- what this does and does NOT do:
- The "recipient roster" below is entirely fictional/demo data: generic
  role titles (not real named officials or real agencies), RFC 2606
  reserved example.com addresses, and 555-01xx phone numbers (the
  universally-recognized fake-number convention). NO real SMS or email is
  ever sent to anyone -- every dispatch is logged as SIMULATED.
- The one REAL side effect this module can have is an optional generic
  webhook POST, and only if the operator sets RADARSENSE_ALERT_WEBHOOK_URL
  themselves (e.g. to their own Slack/Discord incoming-webhook URL) -- that
  is the actual integration point a real deployment would wire up, kept
  separate from the simulated roster so the two are never confused.

Rate-limited per track (COOLDOWN_S) so one persistent CRITICAL track doesn't
spam the log every polling tick.

Real personal email delivery (separate from the simulated roster above): if
RADARSENSE_SMTP_PASS is set (a Gmail App Password -- myaccount.google.com ->
Security -> App Passwords -- generated and stored by the operator, never
handled by this code beyond reading the env var), dispatch_alert also sends
a real email via Gmail SMTP to RADARSENSE_ALERT_EMAIL_TO. SMS to a real
number is intentionally NOT implemented -- it needs a paid third-party
provider account (Twilio etc.) the operator would have to set up themselves;
until then the personal phone number is shown in the response as a labeled,
explicitly-simulated entry, same as the fictional roster.
"""

import os
import smtplib
import time
import uuid
from collections import OrderedDict
from email.message import EmailMessage

import httpx
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter(prefix="/alert", tags=["alert"])

COOLDOWN_S = 30
MAX_HISTORY = 200
WEBHOOK_URL = os.environ.get("RADARSENSE_ALERT_WEBHOOK_URL", "").strip()
WEBHOOK_TIMEOUT_S = 4.0

SMTP_HOST = os.environ.get("RADARSENSE_SMTP_HOST", "smtp.gmail.com")
SMTP_PORT = int(os.environ.get("RADARSENSE_SMTP_PORT", "587"))
SMTP_USER = os.environ.get("RADARSENSE_SMTP_USER", "trivikramnarayanan@gmail.com")
SMTP_PASS = os.environ.get("RADARSENSE_SMTP_PASS", "").strip()
ALERT_EMAIL_TO = os.environ.get("RADARSENSE_ALERT_EMAIL_TO", "trivikramnarayanan@gmail.com")
PERSONAL_PHONE = os.environ.get("RADARSENSE_ALERT_PHONE", "7358084532")

ROSTER = [
    {"role": "Regional Air Defense Command", "channel": "sms",
     "contact": "+1-555-0101 (demo)"},
    {"role": "Regional Air Defense Command", "channel": "email",
     "contact": "air-defense-command@alerts.example.com"},
    {"role": "Counter-UAS Coordination Cell", "channel": "sms",
     "contact": "+1-555-0102 (demo)"},
    {"role": "Counter-UAS Coordination Cell", "channel": "email",
     "contact": "counter-uas-cell@alerts.example.com"},
    {"role": "Civil Administration Liaison", "channel": "email",
     "contact": "civil-liaison@alerts.example.com"},
    {"role": "Local Law Enforcement Control Room", "channel": "sms",
     "contact": "+1-555-0103 (demo)"},
]

_history: "OrderedDict[str, dict]" = OrderedDict()
_last_dispatch: dict = {}


class DispatchRequest(BaseModel):
    track_id: str
    predicted_class: str
    threat_level: str
    threat_score: float
    range_m: float
    closing_mps: float
    tti_sec: float | None = None
    rationale: str = ""


@router.get("/roster")
def get_roster():
    return {"roster": ROSTER,
            "note": "Simulated demo roster -- fictional roles, RFC2606 example.com "
                    "addresses and 555-01xx numbers. No real SMS/email is ever sent."}


@router.post("/dispatch")
def dispatch_alert(req: DispatchRequest):
    now = time.time()
    last = _last_dispatch.get(req.track_id, 0)
    if now - last < COOLDOWN_S:
        return {"dispatched": False,
                "reason": f"cooldown active ({COOLDOWN_S - (now - last):.0f}s remaining)"}
    _last_dispatch[req.track_id] = now

    alert_id = uuid.uuid4().hex[:12]
    recipients = [
        {**r, "status": "simulated_sent", "simulated": True}
        for r in ROSTER
    ]

    webhook_result = None
    if WEBHOOK_URL:
        webhook_result = _fire_webhook(req, alert_id)

    email_result = _send_email(req, alert_id) if SMTP_PASS else \
        {"attempted": False, "reason": "RADARSENSE_SMTP_PASS not set -- see api/alerting.py docstring"}

    personal_contacts = [
        {"role": "Operator (personal)", "channel": "email", "contact": ALERT_EMAIL_TO,
         "simulated": False, "status": "sent" if email_result.get("ok") else "not_sent",
         "detail": email_result},
        {"role": "Operator (personal)", "channel": "sms", "contact": PERSONAL_PHONE,
         "simulated": True, "status": "simulated_sent",
         "detail": {"reason": "real SMS needs a provider account (Twilio etc.) -- not configured"}},
    ]

    record = {
        "alert_id": alert_id,
        "timestamp": now,
        "track_id": req.track_id,
        "predicted_class": req.predicted_class,
        "threat_level": req.threat_level,
        "threat_score": req.threat_score,
        "range_m": req.range_m,
        "closing_mps": req.closing_mps,
        "tti_sec": req.tti_sec,
        "rationale": req.rationale,
        "recipients": recipients,
        "personal_contacts": personal_contacts,
        "webhook": webhook_result,
    }
    _history[alert_id] = record
    while len(_history) > MAX_HISTORY:
        _history.popitem(last=False)

    return {"dispatched": True, "alert": record}


def _send_email(req: DispatchRequest, alert_id: str) -> dict:
    """Real email via Gmail SMTP + App Password. Only runs if SMTP_PASS is
    set (see module docstring) -- never attempted otherwise."""
    msg = EmailMessage()
    msg["Subject"] = f"[RadarSense] CRITICAL threat — {req.predicted_class.upper()} track {req.track_id}"
    msg["From"] = SMTP_USER
    msg["To"] = ALERT_EMAIL_TO
    lines = [
        f"RadarSense threat alert (alert_id {alert_id})",
        "",
        f"Track:            {req.track_id}",
        f"Predicted class:  {req.predicted_class}",
        f"Threat level:     {req.threat_level} (score {req.threat_score:.0f})",
        f"Range:            {req.range_m:.0f} m",
        f"Closing speed:    {req.closing_mps:.1f} m/s",
    ]
    if req.tti_sec is not None:
        lines.append(f"Time to intercept: {req.tti_sec:.0f}s")
    lines.append(f"Rationale:        {req.rationale}")
    msg.set_content("\n".join(lines) + "\n")
    try:
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=10) as server:
            server.starttls()
            server.login(SMTP_USER, SMTP_PASS)
            server.send_message(msg)
        return {"attempted": True, "ok": True}
    except Exception as e:
        return {"attempted": True, "ok": False, "error": str(e)}


def _fire_webhook(req: DispatchRequest, alert_id: str) -> dict:
    payload = {
        "text": f"[RadarSense] CRITICAL threat — track {req.track_id} "
                f"({req.predicted_class}), range {req.range_m:.0f}m, "
                f"closing {req.closing_mps:.1f}m/s. {req.rationale}",
        "alert_id": alert_id,
    }
    try:
        resp = httpx.post(WEBHOOK_URL, json=payload, timeout=WEBHOOK_TIMEOUT_S)
        return {"attempted": True, "status_code": resp.status_code, "ok": resp.is_success}
    except Exception as e:
        return {"attempted": True, "ok": False, "error": str(e)}


@router.get("/history")
def get_history(limit: int = 50):
    items = list(_history.values())[-limit:]
    items.reverse()
    return {"count": len(_history), "alerts": items,
            "webhook_configured": bool(WEBHOOK_URL)}


@router.delete("/history")
def clear_history():
    _history.clear()
    _last_dispatch.clear()
    return {"cleared": True}
