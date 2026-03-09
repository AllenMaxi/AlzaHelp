"""
Centralised configuration — every os.environ.get() lives here.
Import constants from this module instead of reading env vars directly.
"""

import os
import logging
import secrets
from pathlib import Path
from dotenv import load_dotenv

# ── Load .env ────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(ROOT_DIR / ".env")

logger = logging.getLogger(__name__)

# ── JWT / Auth ───────────────────────────────────────────────────
_jwt_secret = os.environ.get("JWT_SECRET_KEY", "").strip()
REQUIRE_JWT_SECRET = os.environ.get("REQUIRE_JWT_SECRET", "false").strip().lower() == "true"
if REQUIRE_JWT_SECRET and not _jwt_secret:
    raise RuntimeError("JWT_SECRET_KEY must be set when REQUIRE_JWT_SECRET=true")
if not _jwt_secret:
    _jwt_secret = secrets.token_hex(32)
    logger.warning("JWT_SECRET_KEY not set — generated ephemeral key. Tokens will not survive restarts.")
SECRET_KEY = _jwt_secret
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24  # 24 hours
REFRESH_TOKEN_EXPIRE_DAYS = 30
COOKIE_SAMESITE = os.environ.get("COOKIE_SAMESITE", "lax")
COOKIE_SECURE = os.environ.get("COOKIE_SECURE", "true").lower() == "true"
ENFORCE_HTTPS = os.environ.get("ENFORCE_HTTPS", "false").strip().lower() == "true"

# ── Voice ────────────────────────────────────────────────────────
VOICE_CONFIRMATION_TTL_SECONDS = max(60, int(os.environ.get("VOICE_CONFIRMATION_TTL_SECONDS", "600") or "600"))

# ── Notification delivery queue ──────────────────────────────────
NOTIFICATION_MAX_ATTEMPTS = int(os.environ.get("NOTIFICATION_MAX_ATTEMPTS", "5"))
NOTIFICATION_BACKOFF_BASE_SECONDS = int(os.environ.get("NOTIFICATION_BACKOFF_BASE_SECONDS", "30"))
NOTIFICATION_BACKOFF_MAX_SECONDS = int(os.environ.get("NOTIFICATION_BACKOFF_MAX_SECONDS", "1800"))
NOTIFICATION_LEASE_SECONDS = 60

CHANNEL_DISPATCH_POLICY = {
    "critical": {
        "immediate": ["push", "sms", "telegram"],
        "if_policy_allows": ["whatsapp"],
    },
    "high": {
        "immediate": ["push", "telegram"],
        "on_escalation": ["sms"],
        "if_policy_allows": ["whatsapp"],
    },
    "medium": {
        "immediate": ["push", "telegram"],
        "on_escalation_threshold": {"sms": 2},
    },
    "low": {
        "immediate": ["push"],
        "if_linked": ["telegram"],
    },
}

WHATSAPP_ALERT_TEMPLATES = {
    "sos_trigger": os.environ.get("TWILIO_WA_TEMPLATE_SOS", "").strip() or None,
    "fall_detected": os.environ.get("TWILIO_WA_TEMPLATE_FALL", "").strip() or None,
    "geofence_exit": os.environ.get("TWILIO_WA_TEMPLATE_GEOFENCE", "").strip() or None,
    "missed_medication_dose": os.environ.get("TWILIO_WA_TEMPLATE_MISSED_MED", "").strip() or None,
}

TRANSIENT_HTTP_CODES = {429, 500, 502, 503, 504}

# ── Write rate limiting ──────────────────────────────────────────
WRITE_RATE_LIMIT_ENABLED = os.environ.get("WRITE_RATE_LIMIT_ENABLED", "true").strip().lower() == "true"
WRITE_RATE_LIMIT_WINDOW_SECONDS = max(10, int(os.environ.get("WRITE_RATE_LIMIT_WINDOW_SECONDS", "60") or "60"))
WRITE_RATE_LIMIT_MAX_REQUESTS = max(10, int(os.environ.get("WRITE_RATE_LIMIT_MAX_REQUESTS", "120") or "120"))

WRITE_RATE_LIMIT_EXEMPT_PREFIXES = (
    "/health",
    "/api/auth/login",
    "/api/auth/register",
    "/api/auth/refresh",
    "/api/auth/logout",
    "/api/auth/demo",
    "/api/voice-command",
    "/api/voice/transcribe",
    "/api/tts",
    "/api/safety/location-ping",
    "/api/safety/escalations/run",
    "/api/webhooks/",
)

# ── Stripe ───────────────────────────────────────────────────────
try:
    import stripe as stripe_lib
    _stripe_key = os.environ.get("STRIPE_SECRET_KEY", "").strip()
    if _stripe_key:
        stripe_lib.api_key = _stripe_key
        _HAS_STRIPE = True
    else:
        _HAS_STRIPE = False
except ImportError:
    stripe_lib = None
    _HAS_STRIPE = False

STRIPE_WEBHOOK_SECRET = os.environ.get("STRIPE_WEBHOOK_SECRET", "").strip()
STRIPE_PRICE_ID = os.environ.get("STRIPE_PRICE_ID", "").strip()

# ── Subscription limits ─────────────────────────────────────────
PATIENT_LIMITS = {"free": 3, "premium": 20}

# ── Navigation / Geocoding ───────────────────────────────────────
NAV_HTTP_USER_AGENT = os.environ.get("NAV_HTTP_USER_AGENT", "AlzaHelp/1.0 (care companion)").strip() or "AlzaHelp/1.0 (care companion)"
GEOCODING_PROVIDER = os.environ.get("NAV_GEOCODING_PROVIDER", "nominatim").strip().lower() or "nominatim"
GEOCODING_SEARCH_URL = os.environ.get("NAV_GEOCODING_SEARCH_URL", "https://nominatim.openstreetmap.org/search").strip()
ROUTING_PROVIDER = os.environ.get("NAV_ROUTING_PROVIDER", "osrm").strip().lower() or "osrm"
ROUTING_BASE_URL = os.environ.get("NAV_ROUTING_BASE_URL", "https://router.project-osrm.org/route/v1").strip().rstrip("/")
LOCATION_STALE_SAMPLE_SECONDS = max(30, int(os.environ.get("NAV_LOCATION_STALE_SAMPLE_SECONDS", "180") or "180"))
NOMINATIM_MIN_INTERVAL_SECONDS = max(1.0, float(os.environ.get("NAV_NOMINATIM_MIN_INTERVAL_SECONDS", "1.0") or "1.0"))

# ── AI / Embedding ───────────────────────────────────────────────
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "text-embedding-3-small")
RAG_CHUNK_SIZE_CHARS = int(os.environ.get("RAG_CHUNK_SIZE_CHARS", "900"))
RAG_CHUNK_OVERLAP_CHARS = int(os.environ.get("RAG_CHUNK_OVERLAP_CHARS", "160"))

# ── Safety ───────────────────────────────────────────────────────
SEVERITY_ORDER = {
    "low": 1,
    "medium": 2,
    "high": 3,
    "critical": 4,
}

SAFETY_EVENT_TYPES = {
    "all",
    "geofence_exit",
    "sos_trigger",
    "fall_detected",
    "missed_medication_dose",
    "location_share",
}

DEFAULT_ESCALATION_RULE_TEMPLATES = [
    {"event_type": "geofence_exit", "min_severity": "high", "intervals_minutes": [5, 15, 30], "enabled": True},
    {"event_type": "sos_trigger", "min_severity": "critical", "intervals_minutes": [2, 5, 15], "enabled": True},
    {"event_type": "fall_detected", "min_severity": "high", "intervals_minutes": [3, 10, 20], "enabled": True},
    {"event_type": "missed_medication_dose", "min_severity": "high", "intervals_minutes": [30, 120], "enabled": True},
]

# ── BPSD ─────────────────────────────────────────────────────────
BPSD_SYMPTOM_TAXONOMY = [
    "agitation", "sundowning", "wandering", "sleep_disturbance",
    "appetite_change", "anxiety", "depression", "apathy",
    "confusion", "aggression", "hallucinations", "repetitive_questions",
]
BPSD_TIME_OF_DAY = ["morning", "afternoon", "evening", "night"]

# ── External bots ────────────────────────────────────────────────
EXTERNAL_BOT_CHANNELS = {"telegram", "whatsapp"}
EXTERNAL_BOT_ALLOWED_ROLES = {"caregiver", "clinician", "admin"}
DOCTOR_BOT_INTENTS = {
    "progress_summary", "medications_today", "missed_doses",
    "safety_alerts", "mood_behavior", "today_instructions",
    "compliance_check", "full_report", "add_medication",
    "update_medication", "deactivate_medication",
    "add_care_instruction", "log_patient_intake", "unknown",
}
DOCTOR_BOT_WRITE_INTENTS = {
    "add_medication", "update_medication", "deactivate_medication",
    "add_care_instruction", "log_patient_intake",
}

# ── File uploads ─────────────────────────────────────────────────
ALLOWED_MIME_TYPES = {
    "image/jpeg", "image/png", "image/gif", "image/webp",
    "audio/mpeg", "audio/wav", "audio/webm", "audio/ogg", "audio/mp4",
    "video/mp4", "video/webm",
    "application/pdf",
}
ALLOWED_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".gif", ".webp",
    ".mp3", ".wav", ".webm", ".ogg", ".m4a",
    ".mp4",
    ".pdf",
}
MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024  # 10 MB

# ── Twilio ───────────────────────────────────────────────────────
TWILIO_SID = os.environ.get("TWILIO_ACCOUNT_SID", "").strip()
TWILIO_FROM = os.environ.get("TWILIO_FROM_NUMBER", "").strip()

# ── Telegram ─────────────────────────────────────────────────────
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_BOT_WEBHOOK_SECRET = os.environ.get("TELEGRAM_BOT_WEBHOOK_SECRET", "").strip()

# ── Push notifications (VAPID) ───────────────────────────────────
VAPID_PRIVATE_KEY = os.environ.get("VAPID_PRIVATE_KEY", "").strip()
VAPID_CLAIMS = {"sub": os.environ.get("VAPID_CLAIMS_SUB", "mailto:admin@alzahelp.com").strip()}

# ── Webhook URLs ─────────────────────────────────────────────────
ALERT_PUSH_WEBHOOK_URL = os.environ.get("ALERT_PUSH_WEBHOOK_URL", "").strip()
ALERT_EMAIL_WEBHOOK_URL = os.environ.get("ALERT_EMAIL_WEBHOOK_URL", "").strip()
ALERT_SMS_WEBHOOK_URL = os.environ.get("ALERT_SMS_WEBHOOK_URL", "").strip()

# ── Admin ────────────────────────────────────────────────────────
ADMIN_BOOTSTRAP_EMAILS_RAW = os.environ.get("ADMIN_BOOTSTRAP_EMAILS", "")

# ── CORS ─────────────────────────────────────────────────────────
CORS_ORIGINS = os.environ.get("CORS_ORIGINS", "").strip() or "http://localhost:3000"
