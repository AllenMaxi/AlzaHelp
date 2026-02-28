from fastapi import FastAPI, APIRouter, HTTPException, Depends, UploadFile, File, Form, Request, Response
from fastapi.responses import StreamingResponse
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorGridFSBucket
import os
import logging
from pathlib import Path
from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional, Any, Tuple
import asyncio
import uuid
from datetime import datetime, timezone, timedelta, date
import httpx
import json
import re
import io
import base64
import math
import secrets
import hmac
import hashlib
import zipfile
import xml.etree.ElementTree as ET
from xml.sax.saxutils import escape as xml_escape
from openai import AsyncOpenAI
from passlib.context import CryptContext
from jose import JWTError, jwt

try:
    from slowapi import Limiter, _rate_limit_exceeded_handler
    from slowapi.util import get_remote_address
    from slowapi.errors import RateLimitExceeded
    _HAS_SLOWAPI = True
except ImportError:
    _HAS_SLOWAPI = False

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

# Patient limits per subscription tier
PATIENT_LIMITS = {"free": 3, "premium": 20}

try:
    from pypdf import PdfReader
except Exception:  # pragma: no cover - optional dependency
    PdfReader = None

try:
    from docx import Document as DocxDocument
except Exception:  # pragma: no cover - optional dependency
    DocxDocument = None

try:
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    from reportlab.lib import colors
    _HAS_REPORTLAB = True
except ImportError:
    _HAS_REPORTLAB = False

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# Auth Configuration
_jwt_secret = os.environ.get("JWT_SECRET_KEY", "").strip()
if not _jwt_secret:
    _jwt_secret = secrets.token_hex(32)
    logging.getLogger(__name__).warning("JWT_SECRET_KEY not set — generated ephemeral key. Tokens will not survive restarts.")
SECRET_KEY = _jwt_secret
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24  # 24 hours
REFRESH_TOKEN_EXPIRE_DAYS = 30
COOKIE_SAMESITE = os.environ.get("COOKIE_SAMESITE", "lax")
COOKIE_SECURE = os.environ.get("COOKIE_SECURE", "true").lower() == "true"

# Password hashing configuration
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Initialize OpenAI client (lazy initialization)
openai_client = None

def get_openai_client():
    global openai_client
    if openai_client is None:
        openai_client = AsyncOpenAI(api_key=os.environ.get('OPENAI_API_KEY', ''))
    return openai_client

# MongoDB connection
mongo_url = os.environ['MONGO_URL']
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ['DB_NAME']]

# GridFS for file storage (safer than local storage)
fs_bucket = AsyncIOMotorGridFSBucket(db)

# Create the main app without a prefix
app = FastAPI(title="AlzaHelp API", version="1.0.0")

# Rate limiting
if _HAS_SLOWAPI:
    limiter = Limiter(key_func=get_remote_address)
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

def rate_limit(limit_string: str):
    """Apply rate limit if slowapi is available, otherwise no-op."""
    if _HAS_SLOWAPI:
        return limiter.limit(limit_string)
    return lambda f: f

# Create a router with the /api prefix
api_router = APIRouter(prefix="/api")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "text-embedding-3-small")
RAG_CHUNK_SIZE_CHARS = int(os.environ.get("RAG_CHUNK_SIZE_CHARS", "900"))
RAG_CHUNK_OVERLAP_CHARS = int(os.environ.get("RAG_CHUNK_OVERLAP_CHARS", "160"))

SEVERITY_ORDER = {
    "low": 1,
    "medium": 2,
    "high": 3,
    "critical": 4
}

SAFETY_EVENT_TYPES = {
    "all",
    "geofence_exit",
    "sos_trigger",
    "fall_detected",
    "missed_medication_dose",
    "location_share"
}

DEFAULT_ESCALATION_RULE_TEMPLATES = [
    {"event_type": "geofence_exit", "min_severity": "high", "intervals_minutes": [5, 15, 30], "enabled": True},
    {"event_type": "sos_trigger", "min_severity": "critical", "intervals_minutes": [2, 5, 15], "enabled": True},
    {"event_type": "fall_detected", "min_severity": "high", "intervals_minutes": [3, 10, 20], "enabled": True},
    {"event_type": "missed_medication_dose", "min_severity": "high", "intervals_minutes": [30, 120], "enabled": True},
]

BPSD_SYMPTOM_TAXONOMY = [
    "agitation",
    "sundowning",
    "wandering",
    "sleep_disturbance",
    "appetite_change",
    "anxiety",
    "depression",
    "apathy",
    "confusion",
    "aggression",
    "hallucinations",
    "repetitive_questions"
]

BPSD_TIME_OF_DAY = ["morning", "afternoon", "evening", "night"]
EXTERNAL_BOT_CHANNELS = {"telegram", "whatsapp"}
EXTERNAL_BOT_ALLOWED_ROLES = {"caregiver", "clinician", "admin"}
DOCTOR_BOT_INTENTS = {
    "progress_summary",
    "medications_today",
    "missed_doses",
    "safety_alerts",
    "mood_behavior",
    "today_instructions",
    "compliance_check",
    "full_report",
    "add_medication",
    "update_medication",
    "deactivate_medication",
    "add_care_instruction",
    "log_patient_intake",
    "unknown"
}
DOCTOR_BOT_WRITE_INTENTS = {
    "add_medication", "update_medication", "deactivate_medication",
    "add_care_instruction", "log_patient_intake"
}

# File upload security
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

def validate_upload(file: UploadFile, content: bytes):
    """Validate file type, extension, and size. Raises HTTPException on failure."""
    if len(content) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(status_code=413, detail=f"File too large. Maximum size is {MAX_FILE_SIZE_BYTES // (1024*1024)}MB.")
    content_type = file.content_type or "application/octet-stream"
    if content_type not in ALLOWED_MIME_TYPES:
        raise HTTPException(status_code=415, detail=f"File type '{content_type}' not allowed.")
    ext = Path(file.filename).suffix.lower() if file.filename else ""
    if ext and ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=415, detail=f"File extension '{ext}' not allowed.")
    return ext or ".bin"

# ==================== HELPERS ====================

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)

def validate_password_strength(password: str):
    """Enforce minimum password requirements."""
    if len(password) < 8:
        raise HTTPException(status_code=400, detail="Password must be at least 8 characters long.")
    if not re.search(r'[A-Z]', password):
        raise HTTPException(status_code=400, detail="Password must contain at least one uppercase letter.")
    if not re.search(r'\d', password):
        raise HTTPException(status_code=400, detail="Password must contain at least one number.")

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    to_encode["iat"] = int(datetime.now(timezone.utc).timestamp())
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def create_refresh_token(user_id: str) -> str:
    expire = datetime.now(timezone.utc) + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
    return jwt.encode({"sub_refresh": user_id, "exp": expire}, SECRET_KEY, algorithm=ALGORITHM)

def parse_admin_bootstrap_emails() -> set:
    raw = os.environ.get("ADMIN_BOOTSTRAP_EMAILS", "")
    return {e.strip().lower() for e in raw.split(",") if e and e.strip()}

def is_bootstrap_admin_email(email: str) -> bool:
    return email.strip().lower() in parse_admin_bootstrap_emails()

def normalize_external_bot_channel(value: Optional[str]) -> str:
    if not value:
        return ""
    cleaned = value.strip().lower()
    return cleaned if cleaned in EXTERNAL_BOT_CHANNELS else ""

def role_can_use_external_bot(user: "User") -> bool:
    return (user.role or "").strip().lower() in EXTERNAL_BOT_ALLOWED_ROLES

def create_external_bot_link_code() -> str:
    # Friendly and short code for messaging apps.
    return secrets.token_hex(4).upper()

def normalize_external_peer_id(channel: str, raw_peer_id: Optional[str]) -> str:
    value = (raw_peer_id or "").strip()
    if channel == "whatsapp":
        value = value.lower()
    return value

def verify_twilio_signature(
    auth_token: str,
    request_url: str,
    params: dict,
    signature: str
) -> bool:
    """Validate Twilio webhook signature using X-Twilio-Signature."""
    if not auth_token:
        return True
    if not signature:
        return False
    base = request_url
    for key in sorted(params.keys()):
        base += f"{key}{params[key]}"
    digest = hmac.new(auth_token.encode("utf-8"), base.encode("utf-8"), hashlib.sha1).digest()
    expected = base64.b64encode(digest).decode("utf-8")
    return hmac.compare_digest(expected, signature)

def classify_doctor_bot_intent_heuristic(text: str) -> str:
    lowered = (text or "").strip().lower()
    if not lowered:
        return "unknown"
    # Write intents (check first — higher priority than reads)
    if any(k in lowered for k in ["add medication", "add medicine", "new medication", "new medicine",
                                   "prescribe", "new prescription", "create medication", "add pill",
                                   "add a medication", "add a medicine", "add a pill"]):
        return "add_medication"
    if any(k in lowered for k in ["add instruction", "add care instruction", "create instruction",
                                   "new instruction", "new care plan", "add protocol",
                                   "upload instruction", "add procedure", "create care plan"]):
        return "add_care_instruction"
    if any(k in lowered for k in ["stop medication", "stop medicine", "deactivate medication",
                                   "discontinue", "cancel medication", "remove medication",
                                   "stop pill", "deactivate medicine"]):
        return "deactivate_medication"
    if any(k in lowered for k in ["update medication", "change medication", "change dosage",
                                   "update dosage", "modify medication", "adjust medication",
                                   "change dose", "update dose", "change times",
                                   "update schedule", "change frequency"]):
        return "update_medication"
    if (any(k in lowered for k in ["mark as taken", "mark taken", "log intake", "patient took",
                                    "mark dose taken", "confirm intake", "record intake",
                                    "log dose", "she took", "he took", "as taken"])
            or re.search(r"\bmark\b.*\btaken\b", lowered)):
        return "log_patient_intake"
    # Read intents
    if any(k in lowered for k in ["full report", "complete report", "everything", "all details"]):
        return "full_report"
    if any(k in lowered for k in ["missed", "skipped", "not taken", "overdue dose"]):
        return "missed_doses"
    if any(k in lowered for k in ["alert", "sos", "fall", "geofence", "safety"]):
        return "safety_alerts"
    if any(k in lowered for k in ["mood", "behavior", "agitation", "sleep", "bpsd", "anxiety", "depression"]):
        return "mood_behavior"
    if any(k in lowered for k in ["instruction", "procedure", "protocol", "regimen", "care plan", "steps"]) and any(
        c in lowered for c in ["today", "tonight", "now", "read", "what are", "show"]
    ):
        return "today_instructions"
    if any(k in lowered for k in ["requirement", "compliance", "fulfilled", "all done", "completed all"]):
        return "compliance_check"
    if any(k in lowered for k in ["medication", "medicine", "pill", "dose", "dosage", "what to take", "when to take"]):
        return "medications_today"
    if any(k in lowered for k in ["progress", "status", "summary", "how is", "report", "update"]):
        return "progress_summary"
    return "unknown"

def normalize_hhmm(value: str) -> str:
    """Normalize time values into HH:MM."""
    if not value:
        return ""
    value = value.strip()
    match = re.match(r"^(\d{1,2}):(\d{2})$", value)
    if not match:
        return value
    hour = int(match.group(1))
    minute = int(match.group(2))
    if hour < 0 or hour > 23 or minute < 0 or minute > 59:
        return value
    return f"{hour:02d}:{minute:02d}"

def parse_hhmm_to_datetime(time_str: str, base_date: datetime) -> Optional[datetime]:
    """Build datetime for today using HH:MM time string."""
    try:
        normalized = normalize_hhmm(time_str)
        hour, minute = normalized.split(":")
        return base_date.replace(hour=int(hour), minute=int(minute), second=0, microsecond=0)
    except Exception:
        return None

def format_hhmm_for_voice(value: str) -> str:
    """Render HH:MM in human voice format, e.g. 08:00 -> 8:00 AM."""
    normalized = normalize_hhmm(value)
    match = re.match(r"^(\d{2}):(\d{2})$", normalized)
    if not match:
        return value
    hour = int(match.group(1))
    minute = int(match.group(2))
    meridiem = "AM" if hour < 12 else "PM"
    hour_12 = hour % 12 or 12
    return f"{hour_12}:{minute:02d} {meridiem}"

def format_datetime_for_voice(dt: datetime) -> str:
    return format_hhmm_for_voice(f"{dt.hour:02d}:{dt.minute:02d}")

def join_voice_items(items: List[str]) -> str:
    cleaned = [i for i in items if i]
    if not cleaned:
        return ""
    if len(cleaned) == 1:
        return cleaned[0]
    if len(cleaned) == 2:
        return f"{cleaned[0]} and {cleaned[1]}"
    return f"{', '.join(cleaned[:-1])}, and {cleaned[-1]}"

def is_medication_schedule_question(text: str) -> bool:
    lowered = (text or "").strip().lower()
    if not lowered:
        return False
    # Keep explicit navigation commands on the navigation path.
    if re.search(r"\b(open|show|go to|take me to|navigate)\b.*\b(medication|medications|medicine|medicines|pills?)\b", lowered):
        return False

    med_terms = ["medication", "medications", "medicine", "medicines", "pill", "pills", "dose", "dosage"]
    if not any(term in lowered for term in med_terms):
        return False

    explicit_schedule_phrases = [
        "what time",
        "at what time",
        "when should i take",
        "what should i take",
        "which should i take",
        "have to take",
        "need to take",
        "what do i take",
        "which do i take"
    ]
    has_explicit_schedule_phrase = any(phrase in lowered for phrase in explicit_schedule_phrases)
    has_today_schedule_phrase = (
        any(token in lowered for token in ["today", "tonight", "now"]) and
        any(token in lowered for token in ["take", "dose", "dosage", "time", "when"])
    )
    return has_explicit_schedule_phrase or has_today_schedule_phrase

def is_medication_taken_report(text: str) -> bool:
    """Detect when patient reports they took their medication."""
    lowered = (text or "").strip().lower()
    if not lowered:
        return False
    # Must not match schedule questions
    if is_medication_schedule_question(text):
        return False
    taken_phrases = [
        "i took", "i've taken", "i have taken", "already took", "just took",
        "took my", "finished my medicine", "finished my medication",
        "done with my pills", "done with my medicine", "done with my medication",
        "i had my medication", "i had my medicine", "i had my pill",
        "i took the", "taken my medicine", "taken my medication", "taken my pill",
        "i swallowed", "medicine taken", "medication taken", "pill taken",
        "took the pill", "took the medicine", "took the medication",
        "i already had my", "completed my dose", "had my dose"
    ]
    return any(phrase in lowered for phrase in taken_phrases)


def is_today_instruction_question(text: str) -> bool:
    lowered = (text or "").strip().lower()
    if not lowered:
        return False

    instruction_terms = [
        "instruction",
        "instructions",
        "procedure",
        "procedures",
        "protocol",
        "regimen",
        "care plan",
        "steps"
    ]
    if not any(term in lowered for term in instruction_terms):
        return False

    cue_terms = [
        "today",
        "tonight",
        "now",
        "read",
        "tell me",
        "what are",
        "what should",
        "what do i"
    ]
    return any(cue in lowered for cue in cue_terms)

async def classify_special_voice_intent(text: str) -> str:
    """
    Lightweight classifier for high-priority intents that need deterministic handling.
    Returns: medication_schedule | today_instructions | unsupported_chess | none
    """
    if not os.environ.get("OPENAI_API_KEY"):
        return "none"
    prompt = (
        "Classify the user's utterance into one intent.\n"
        "Allowed intents:\n"
        "- medication_schedule: asking what/when medicines to take today or now.\n"
        "- medication_taken: patient reports they took their medicine, a pill, a dose, or confirms intake.\n"
        "- mood_report: patient describes how they feel, their mood, energy, or emotional state (e.g. 'I feel happy', 'I'm sad', 'feeling tired').\n"
        "- today_instructions: asking for today's care procedures/instructions/regimen.\n"
        "- unsupported_chess: asking to play chess.\n"
        "- none: everything else.\n"
        "Return strict JSON only: {\"intent\":\"...\"}."
    )
    try:
        client = get_openai_client()
        completion = await client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0,
            max_tokens=40,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": text}
            ]
        )
        raw = (completion.choices[0].message.content or "").strip()
        parsed = json.loads(raw) if raw else {}
        intent = str(parsed.get("intent", "none")).strip().lower()
        if intent in {"medication_schedule", "medication_taken", "mood_report", "today_instructions", "unsupported_chess", "none"}:
            return intent
    except Exception as e:
        logger.warning(f"Special voice intent classification failed: {e}")
    return "none"

def haversine_distance_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Distance in meters between coordinates."""
    r = 6371000  # Earth radius (m)
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return r * c

def bearing_degrees(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Initial bearing in degrees from point A to B."""
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dlon = math.radians(lon2 - lon1)
    y = math.sin(dlon) * math.cos(phi2)
    x = math.cos(phi1) * math.sin(phi2) - math.sin(phi1) * math.cos(phi2) * math.cos(dlon)
    return (math.degrees(math.atan2(y, x)) + 360) % 360

def cardinal_direction(bearing: float) -> str:
    directions = ["north", "north-east", "east", "south-east", "south", "south-west", "west", "north-west"]
    idx = round(bearing / 45) % 8
    return directions[idx]

def normalize_frequency(value: Optional[str], fallback: str = "daily") -> str:
    allowed = {"daily", "weekly", "as_needed"}
    if not value:
        return fallback
    cleaned = value.strip().lower()
    return cleaned if cleaned in allowed else fallback

def normalize_day_of_week(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    cleaned = value.strip().lower()
    allowed = {
        "monday", "tuesday", "wednesday", "thursday",
        "friday", "saturday", "sunday"
    }
    return cleaned if cleaned in allowed else None

def normalize_policy_type(value: Optional[str], fallback: str = "general") -> str:
    allowed = {"general", "medication"}
    if not value:
        return fallback
    cleaned = value.strip().lower()
    return cleaned if cleaned in allowed else fallback

def normalize_signoff_status(value: Optional[str], required: bool) -> str:
    if not required:
        return "not_required"
    allowed = {"pending", "signed_off", "rejected"}
    if not value:
        return "pending"
    cleaned = value.strip().lower()
    return cleaned if cleaned in allowed else "pending"

def parse_yyyy_mm_dd(value: Optional[str]) -> Optional[date]:
    if not value:
        return None
    try:
        return datetime.strptime(value.strip(), "%Y-%m-%d").date()
    except Exception:
        return None

def normalize_yyyy_mm_dd(value: Optional[str]) -> Optional[str]:
    parsed = parse_yyyy_mm_dd(value)
    return parsed.isoformat() if parsed else None

def clean_phone_for_dial(phone: Optional[str]) -> Optional[str]:
    if not phone:
        return None
    cleaned = re.sub(r"[^\d+]", "", phone)
    return cleaned if cleaned else None

def normalize_severity(value: Optional[str], fallback: str = "high") -> str:
    if not value:
        return fallback
    cleaned = value.strip().lower()
    return cleaned if cleaned in SEVERITY_ORDER else fallback

def normalize_safety_event_type(value: Optional[str], fallback: str = "all") -> str:
    if not value:
        return fallback
    cleaned = value.strip().lower()
    return cleaned if cleaned in SAFETY_EVENT_TYPES else fallback

def severity_at_least(actual: str, minimum: str) -> bool:
    return SEVERITY_ORDER.get(normalize_severity(actual), 0) >= SEVERITY_ORDER.get(normalize_severity(minimum), 0)

def normalize_escalation_intervals(values: Optional[List[int]]) -> List[int]:
    if not isinstance(values, list):
        return []
    cleaned = []
    for item in values:
        try:
            minutes = int(item)
        except Exception:
            continue
        if 1 <= minutes <= 24 * 60:
            cleaned.append(minutes)
    # Keep bounded number of escalation steps and ensure deterministic order.
    unique = sorted(set(cleaned))
    return unique[:5]

def parse_iso_to_utc(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        return None

def normalize_bpsd_symptom(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    cleaned = value.strip().lower().replace(" ", "_")
    return cleaned if cleaned in BPSD_SYMPTOM_TAXONOMY else None

def normalize_bpsd_time_of_day(value: Optional[str], fallback: str = "evening") -> str:
    if not value:
        return fallback
    cleaned = value.strip().lower()
    return cleaned if cleaned in BPSD_TIME_OF_DAY else fallback

def clamp_score(value: Optional[int], default_value: int = 3) -> int:
    try:
        score = int(value)
    except Exception:
        return default_value
    return max(1, min(score, 5))

def normalize_appetite(value: Optional[str], fallback: str = "normal") -> str:
    if not value:
        return fallback
    cleaned = value.strip().lower()
    return cleaned if cleaned in {"low", "normal", "high"} else fallback

def medication_active_on_day(medication: dict, target_day: date) -> bool:
    if medication.get("active") is False:
        return False
    start_d = parse_yyyy_mm_dd(medication.get("start_date"))
    end_d = parse_yyyy_mm_dd(medication.get("end_date"))
    if start_d and target_day < start_d:
        return False
    if end_d and target_day > end_d:
        return False
    return True

def medication_due_on_day(medication: dict, target_day: date) -> bool:
    if not medication_active_on_day(medication, target_day):
        return False
    frequency = (medication.get("frequency") or "daily").strip().lower()
    start_d = parse_yyyy_mm_dd(medication.get("start_date"))

    if frequency == "weekly":
        weekday = (start_d.weekday() if start_d else 0)
        return target_day.weekday() == weekday

    if frequency == "every_other_day":
        if not start_d:
            return True
        delta_days = (target_day - start_d).days
        return delta_days >= 0 and delta_days % 2 == 0

    # daily, custom, twice_daily, three_times_daily default to daily due slots
    return True

def medication_schedule_times(medication: dict) -> List[str]:
    explicit = [normalize_hhmm(t) for t in (medication.get("scheduled_times") or []) if t]
    if explicit:
        return explicit

    frequency = (medication.get("frequency") or "daily").strip().lower()
    if frequency == "twice_daily":
        return ["08:00", "20:00"]
    if frequency == "three_times_daily":
        return ["08:00", "14:00", "20:00"]

    count = max(1, int(medication.get("times_per_day", 1)))
    default_slots = ["08:00", "12:00", "18:00", "21:00"]
    return default_slots[:min(count, len(default_slots))]

def extract_instruction_text_from_upload(
    content: bytes,
    content_type: Optional[str],
    filename: Optional[str]
) -> str:
    """Best-effort text extraction for txt/md/csv/json/pdf/docx."""
    ctype = (content_type or "").lower()
    ext = (Path(filename).suffix.lower() if filename else "")

    if ctype.startswith("text/") or ext in {".txt", ".md", ".csv", ".json"}:
        return content.decode("utf-8", errors="ignore").strip()

    if ext == ".pdf" or "application/pdf" in ctype:
        if PdfReader is None:
            return ""
        try:
            reader = PdfReader(io.BytesIO(content))
            parts = []
            for page in reader.pages:
                page_text = page.extract_text() or ""
                if page_text.strip():
                    parts.append(page_text.strip())
            return "\n\n".join(parts).strip()
        except Exception:
            return ""

    if ext == ".docx" or "officedocument.wordprocessingml.document" in ctype:
        if DocxDocument is not None:
            try:
                doc = DocxDocument(io.BytesIO(content))
                text = "\n".join([p.text.strip() for p in doc.paragraphs if p.text and p.text.strip()])
                if text.strip():
                    return text.strip()
            except Exception:
                pass
        # Fallback parser using DOCX XML payload.
        try:
            with zipfile.ZipFile(io.BytesIO(content), "r") as zf:
                xml_raw = zf.read("word/document.xml")
            root = ET.fromstring(xml_raw)
            ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
            texts = [node.text for node in root.findall(".//w:t", ns) if node.text]
            return " ".join(texts).strip()
        except Exception:
            return ""

    return ""

def build_instruction_search_text(doc: dict) -> str:
    tags_text = " ".join(doc.get("tags") or [])
    return (
        f"{doc.get('title', '')} {doc.get('summary', '')} {doc.get('instruction_text', '')} "
        f"{doc.get('frequency', '')} {doc.get('day_of_week', '') or ''} {doc.get('time_of_day', '') or ''} {tags_text}"
    ).lower()

def is_instruction_effective_on_date(instruction: dict, target_date: date) -> bool:
    start_d = parse_yyyy_mm_dd(instruction.get("effective_start_date"))
    end_d = parse_yyyy_mm_dd(instruction.get("effective_end_date"))
    if start_d and target_date < start_d:
        return False
    if end_d and target_date > end_d:
        return False
    return True

def instruction_allowed_for_patient_use(instruction: dict, target_date: date) -> bool:
    if not instruction.get("active", True):
        return False
    if instruction.get("status") in {"archived", "rejected"}:
        return False
    signoff_required = bool(
        instruction.get("signoff_required", instruction.get("policy_type") == "medication")
    )
    signoff_status = normalize_signoff_status(instruction.get("signoff_status"), signoff_required)
    if signoff_required and signoff_status != "signed_off":
        return False
    return is_instruction_effective_on_date(instruction, target_date)

def chunk_text_for_rag(
    text: str,
    chunk_size_chars: int = RAG_CHUNK_SIZE_CHARS,
    overlap_chars: int = RAG_CHUNK_OVERLAP_CHARS
) -> List[str]:
    cleaned = re.sub(r"\s+", " ", (text or "").strip())
    if not cleaned:
        return []
    if len(cleaned) <= chunk_size_chars:
        return [cleaned]

    chunks = []
    start = 0
    text_len = len(cleaned)
    while start < text_len:
        end = min(text_len, start + chunk_size_chars)
        window = cleaned[start:end]
        # Prefer sentence-like boundary.
        if end < text_len:
            boundary = max(window.rfind(". "), window.rfind("! "), window.rfind("? "))
            if boundary > int(chunk_size_chars * 0.55):
                end = start + boundary + 1
                window = cleaned[start:end]
        chunks.append(window.strip())
        if end >= text_len:
            break
        start = max(0, end - overlap_chars)
    return [c for c in chunks if c]

def cosine_similarity(vec_a: Optional[List[float]], vec_b: Optional[List[float]]) -> float:
    if not vec_a or not vec_b or len(vec_a) != len(vec_b):
        return 0.0
    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0
    for a, b in zip(vec_a, vec_b):
        dot += a * b
        norm_a += a * a
        norm_b += b * b
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (math.sqrt(norm_a) * math.sqrt(norm_b))

async def generate_embeddings(texts: List[str]) -> Optional[List[List[float]]]:
    if not texts:
        return []
    if not os.environ.get("OPENAI_API_KEY"):
        return None
    try:
        client = get_openai_client()
        response = await client.embeddings.create(model=EMBEDDING_MODEL, input=texts)
        vectors = [item.embedding for item in response.data]
        if len(vectors) != len(texts):
            logger.warning("Embedding count mismatch: %s != %s", len(vectors), len(texts))
        return vectors
    except Exception as exc:
        logger.error("Embedding generation failed: %s", exc)
        return None

async def get_query_embedding(query: str) -> Optional[List[float]]:
    embeddings = await generate_embeddings([query])
    if not embeddings:
        return None
    return embeddings[0]

def _build_memory_search_text(mem: dict) -> str:
    people_str = ", ".join(mem.get("people", [])) if mem.get("people") else ""
    return f"{mem.get('title', '')} {mem.get('date', '')} {mem.get('location', '')} {mem.get('description', '')} {people_str}"

def _build_family_search_text(fam: dict) -> str:
    return f"{fam.get('name', '')} {fam.get('relationship', '')} {fam.get('relationship_label', '')} {fam.get('notes', '')} {fam.get('category', '')}"

async def _embed_and_store_memory(doc: dict):
    text = _build_memory_search_text(doc)
    embeddings = await generate_embeddings([text])
    if embeddings and embeddings[0]:
        await db.memories.update_one(
            {"id": doc["id"], "user_id": doc["user_id"]},
            {"$set": {"embedding": embeddings[0]}}
        )

async def _embed_and_store_family(doc: dict):
    text = _build_family_search_text(doc)
    embeddings = await generate_embeddings([text])
    if embeddings and embeddings[0]:
        await db.family_members.update_one(
            {"id": doc["id"], "user_id": doc["user_id"]},
            {"$set": {"embedding": embeddings[0]}}
        )

async def upsert_instruction_chunks(instruction_doc: dict) -> int:
    """Regenerate chunk documents + embeddings for an instruction."""
    instruction_id = instruction_doc["id"]
    user_id = instruction_doc["user_id"]
    await db.care_instruction_chunks.delete_many({"instruction_id": instruction_id, "user_id": user_id})

    rag_source_text = "\n\n".join([
        instruction_doc.get("title", "").strip(),
        (instruction_doc.get("summary") or "").strip(),
        (instruction_doc.get("instruction_text") or "").strip()
    ]).strip()
    chunks = chunk_text_for_rag(rag_source_text)
    if not chunks:
        await db.care_instructions.update_one(
            {"id": instruction_id, "user_id": user_id},
            {"$set": {"chunk_count": 0, "embedding_updated_at": datetime.now(timezone.utc).isoformat()}}
        )
        return 0

    embeddings = await generate_embeddings(chunks)
    now_iso = datetime.now(timezone.utc).isoformat()
    docs = []
    for idx, chunk_text in enumerate(chunks):
        docs.append({
            "id": f"instrchunk_{uuid.uuid4().hex[:12]}",
            "user_id": user_id,
            "instruction_id": instruction_id,
            "instruction_version": int(instruction_doc.get("version", 1)),
            "policy_type": instruction_doc.get("policy_type", "general"),
            "regimen_key": instruction_doc.get("regimen_key"),
            "chunk_index": idx,
            "chunk_text": chunk_text,
            "snippet": chunk_text[:260],
            "embedding": embeddings[idx] if embeddings and idx < len(embeddings) else None,
            "created_at": now_iso
        })
    if docs:
        await db.care_instruction_chunks.insert_many(docs)
    await db.care_instructions.update_one(
        {"id": instruction_id, "user_id": user_id},
        {"$set": {"chunk_count": len(docs), "embedding_updated_at": now_iso}}
    )
    return len(docs)

def build_citation_snippet_block(citations: List[dict], max_items: int = 4) -> str:
    if not citations:
        return ""
    lines = ["Citations:"]
    for item in citations[:max_items]:
        lines.append(
            f"[{item['id']}] {item.get('title', 'Instruction')} (v{item.get('version', 1)}): {item.get('snippet', '')}"
        )
    return "\n".join(lines)

def select_latest_medication_regimens(instructions: List[dict]) -> List[dict]:
    med_by_key = {}
    for inst in instructions:
        if inst.get("policy_type") != "medication":
            continue
        key = inst.get("regimen_key") or re.sub(r"\s+", "_", inst.get("title", "").lower())
        prev = med_by_key.get(key)
        if not prev:
            med_by_key[key] = inst
            continue
        prev_version = int(prev.get("version", 1))
        next_version = int(inst.get("version", 1))
        if next_version > prev_version or (
            next_version == prev_version and inst.get("updated_at", "") > prev.get("updated_at", "")
        ):
            med_by_key[key] = inst
    return list(med_by_key.values())

# ==================== MODELS ====================

class User(BaseModel):
    model_config = ConfigDict(extra="ignore")
    user_id: str
    email: str
    name: str
    role: str = "patient"  # patient, caregiver, clinician, admin
    account_status: str = "active"  # active, pending, suspended, rejected
    clinician_approval_status: str = "not_applicable"  # not_applicable, pending, approved, rejected, suspended
    requested_role: Optional[str] = None
    clinician_profile: Optional[dict] = None
    approved_by_admin_user_id: Optional[str] = None
    approved_at: Optional[str] = None
    approval_notes: Optional[str] = None
    hashed_password: Optional[str] = None
    picture: Optional[str] = None
    linked_patient_ids: List[str] = []
    subscription_tier: str = "free"  # "free" | "premium"
    stripe_customer_id: Optional[str] = None
    stripe_subscription_id: Optional[str] = None
    subscription_expires_at: Optional[datetime] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class UserCreate(BaseModel):
    email: str
    password: str
    name: str
    role: str = "patient"
    license_number: Optional[str] = None
    medical_organization: Optional[str] = None
    jurisdiction: Optional[str] = None
    referral_code: Optional[str] = None

class UserLogin(BaseModel):
    email: str
    password: str

class UserSession(BaseModel):
    model_config = ConfigDict(extra="ignore")
    user_id: str
    session_token: str
    expires_at: datetime
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class AdminClinicianDecision(BaseModel):
    notes: Optional[str] = None

class FamilyMember(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str = Field(default_factory=lambda: f"family_{uuid.uuid4().hex[:12]}")
    user_id: str
    name: str
    relationship: str
    relationship_label: str
    phone: Optional[str] = None
    address: Optional[str] = None
    birthday: Optional[str] = None
    photos: List[str] = []
    voice_notes: List[str] = []
    notes: Optional[str] = None
    category: str  # spouse, children, grandchildren, friends, other
    embedding: Optional[List[float]] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class FamilyMemberCreate(BaseModel):
    name: str
    relationship: str
    relationship_label: str
    phone: Optional[str] = None
    address: Optional[str] = None
    birthday: Optional[str] = None
    photos: List[str] = []
    voice_notes: List[str] = []
    notes: Optional[str] = None
    category: str

class FamilyMemberUpdate(BaseModel):
    name: Optional[str] = None
    relationship: Optional[str] = None
    relationship_label: Optional[str] = None
    phone: Optional[str] = None
    address: Optional[str] = None
    birthday: Optional[str] = None
    photos: Optional[List[str]] = None
    voice_notes: Optional[List[str]] = None
    notes: Optional[str] = None
    category: Optional[str] = None

class Memory(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str = Field(default_factory=lambda: f"memory_{uuid.uuid4().hex[:12]}")
    user_id: str
    title: str
    date: str
    year: int
    location: Optional[str] = None
    description: str
    people: List[str] = []
    photos: List[str] = []
    category: str  # milestone, family, travel, celebration, other
    embedding: Optional[List[float]] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class MemoryCreate(BaseModel):
    title: str
    date: str
    year: int
    location: Optional[str] = None
    description: str
    people: List[str] = []
    photos: List[str] = []
    category: str

class MemoryUpdate(BaseModel):
    title: Optional[str] = None
    date: Optional[str] = None
    year: Optional[int] = None
    location: Optional[str] = None
    description: Optional[str] = None
    people: Optional[List[str]] = None
    photos: Optional[List[str]] = None
    category: Optional[str] = None

class Reminder(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str = Field(default_factory=lambda: f"reminder_{uuid.uuid4().hex[:12]}")
    user_id: str
    title: str
    time: str
    period: str  # morning, afternoon, evening, night
    category: str  # health, meals, activity
    completed: bool = False
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class ReminderCreate(BaseModel):
    title: str
    time: str
    period: str
    category: str

class Destination(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str = Field(default_factory=lambda: f"destination_{uuid.uuid4().hex[:12]}")
    user_id: str
    name: str
    address: str
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    visit_time: Optional[str] = None
    notes: Optional[str] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class DestinationCreate(BaseModel):
    name: str
    address: str
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    visit_time: Optional[str] = None
    notes: Optional[str] = None

class DestinationUpdate(BaseModel):
    name: Optional[str] = None
    address: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    visit_time: Optional[str] = None
    notes: Optional[str] = None

class NavigationGuideRequest(BaseModel):
    destination_id: str
    current_latitude: float
    current_longitude: float

class SafetyZone(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str = Field(default_factory=lambda: f"zone_{uuid.uuid4().hex[:12]}")
    user_id: str
    name: str
    center_latitude: float
    center_longitude: float
    radius_meters: int = 500
    active: bool = True
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class SafetyZoneCreate(BaseModel):
    name: str
    center_latitude: float
    center_longitude: float
    radius_meters: int = 500

class SafetyZoneUpdate(BaseModel):
    name: Optional[str] = None
    center_latitude: Optional[float] = None
    center_longitude: Optional[float] = None
    radius_meters: Optional[int] = None
    active: Optional[bool] = None

class SafetyLocationPing(BaseModel):
    latitude: float
    longitude: float

class EmergencyContact(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str = Field(default_factory=lambda: f"emergency_{uuid.uuid4().hex[:12]}")
    user_id: str
    name: str
    relationship: str = "caregiver"
    phone: str
    is_primary: bool = False
    receive_call: bool = True
    receive_sms: bool = True
    notes: Optional[str] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class EmergencyContactCreate(BaseModel):
    name: str
    relationship: str = "caregiver"
    phone: str
    is_primary: bool = False
    receive_call: bool = True
    receive_sms: bool = True
    notes: Optional[str] = None

class EmergencyContactUpdate(BaseModel):
    name: Optional[str] = None
    relationship: Optional[str] = None
    phone: Optional[str] = None
    is_primary: Optional[bool] = None
    receive_call: Optional[bool] = None
    receive_sms: Optional[bool] = None
    notes: Optional[str] = None

class SOSTriggerRequest(BaseModel):
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    message: Optional[str] = None
    auto_call_primary: bool = True

class LocationShareRequest(BaseModel):
    latitude: float
    longitude: float
    reason: str = "manual_share"

class FallEventCreate(BaseModel):
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    detected_by: str = "manual"  # manual, device_motion, wearable
    severity: str = "high"       # medium, high, critical
    confidence: Optional[float] = None
    notes: Optional[str] = None

class SafetyEscalationRule(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str = Field(default_factory=lambda: f"esc_rule_{uuid.uuid4().hex[:12]}")
    user_id: str
    event_type: str = "all"
    min_severity: str = "high"
    intervals_minutes: List[int] = [5, 15, 30]
    enabled: bool = True
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class SafetyEscalationRuleCreate(BaseModel):
    event_type: str = "all"
    min_severity: str = "high"
    intervals_minutes: List[int] = [5, 15, 30]
    enabled: bool = True

class SafetyEscalationRuleUpdate(BaseModel):
    event_type: Optional[str] = None
    min_severity: Optional[str] = None
    intervals_minutes: Optional[List[int]] = None
    enabled: Optional[bool] = None

class CareInstruction(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str = Field(default_factory=lambda: f"instruction_{uuid.uuid4().hex[:12]}")
    user_id: str
    title: str
    instruction_text: str
    summary: Optional[str] = None
    frequency: str = "daily"  # daily, weekly, as_needed
    day_of_week: Optional[str] = None
    time_of_day: Optional[str] = None
    tags: List[str] = []
    policy_type: str = "general"  # general, medication
    regimen_key: Optional[str] = None
    version: int = 1
    status: str = "active"  # draft, active, archived, rejected
    effective_start_date: Optional[str] = None  # YYYY-MM-DD
    effective_end_date: Optional[str] = None    # YYYY-MM-DD
    signoff_required: bool = False
    signoff_status: str = "not_required"  # not_required, pending, signed_off, rejected
    signed_off_by_name: Optional[str] = None
    signed_off_by_user_id: Optional[str] = None
    signed_off_at: Optional[str] = None
    signed_off_notes: Optional[str] = None
    supersedes_instruction_id: Optional[str] = None
    chunk_count: int = 0
    embedding_updated_at: Optional[str] = None
    source_type: str = "text"  # text or file
    source_filename: Optional[str] = None
    source_file_url: Optional[str] = None
    active: bool = True
    uploaded_by_user_id: str
    uploaded_by_role: str = "patient"
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class CareInstructionCreate(BaseModel):
    title: str
    instruction_text: str
    summary: Optional[str] = None
    frequency: str = "daily"
    day_of_week: Optional[str] = None
    time_of_day: Optional[str] = None
    tags: List[str] = []
    active: bool = True
    policy_type: str = "general"
    regimen_key: Optional[str] = None
    effective_start_date: Optional[str] = None
    effective_end_date: Optional[str] = None
    signoff_required: Optional[bool] = None

class CareInstructionUpdate(BaseModel):
    title: Optional[str] = None
    instruction_text: Optional[str] = None
    summary: Optional[str] = None
    frequency: Optional[str] = None
    day_of_week: Optional[str] = None
    time_of_day: Optional[str] = None
    tags: Optional[List[str]] = None
    active: Optional[bool] = None
    policy_type: Optional[str] = None
    regimen_key: Optional[str] = None
    status: Optional[str] = None
    effective_start_date: Optional[str] = None
    effective_end_date: Optional[str] = None
    signoff_required: Optional[bool] = None

class CareInstructionSignoffRequest(BaseModel):
    approved: bool = True
    signed_by_name: Optional[str] = None
    notes: Optional[str] = None
    effective_start_date: Optional[str] = None
    effective_end_date: Optional[str] = None

class Medication(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str = Field(default_factory=lambda: f"med_{uuid.uuid4().hex[:12]}")
    user_id: str
    name: str
    dosage: str
    frequency: str  # daily, twice_daily, three_times_daily, every_other_day, weekly, custom
    times_per_day: int = 1
    scheduled_times: List[str] = []
    prescribing_doctor: Optional[str] = None
    instructions: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    active: bool = True
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class MedicationCreate(BaseModel):
    name: str
    dosage: str
    frequency: str
    times_per_day: int = 1
    scheduled_times: List[str] = []
    prescribing_doctor: Optional[str] = None
    instructions: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None

class MedicationUpdate(BaseModel):
    name: Optional[str] = None
    dosage: Optional[str] = None
    frequency: Optional[str] = None
    times_per_day: Optional[int] = None
    scheduled_times: Optional[List[str]] = None
    prescribing_doctor: Optional[str] = None
    instructions: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    active: Optional[bool] = None

class MedicationIntakeLog(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str = Field(default_factory=lambda: f"intake_{uuid.uuid4().hex[:12]}")
    user_id: str
    medication_id: str
    status: str  # taken, missed
    scheduled_for: Optional[datetime] = None
    confirmed_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    source: str = "manual"
    notes: Optional[str] = None
    recorded_by_user_id: Optional[str] = None
    delay_minutes: Optional[int] = None

class MedicationIntakeCreate(BaseModel):
    status: str = "taken"
    scheduled_for: Optional[str] = None
    source: Optional[str] = "manual"
    notes: Optional[str] = None

class MoodCheckin(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str = Field(default_factory=lambda: f"mood_{uuid.uuid4().hex[:12]}")
    user_id: str
    mood_score: int = 3
    energy_score: int = 3
    anxiety_score: int = 3
    sleep_quality: int = 3
    appetite: str = "normal"
    notes: Optional[str] = None
    source: str = "patient"
    created_by_user_id: Optional[str] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class MoodCheckinCreate(BaseModel):
    mood_score: int = 3
    energy_score: int = 3
    anxiety_score: int = 3
    sleep_quality: int = 3
    appetite: str = "normal"
    notes: Optional[str] = None
    source: Optional[str] = None

class BPSDObservation(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str = Field(default_factory=lambda: f"bpsd_{uuid.uuid4().hex[:12]}")
    user_id: str
    symptom: str
    severity: int = 3
    time_of_day: str = "evening"
    duration_minutes: Optional[int] = None
    trigger_tags: List[str] = []
    notes: Optional[str] = None
    observed_by_role: str = "caregiver"
    observed_by_user_id: Optional[str] = None
    observed_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class BPSDObservationCreate(BaseModel):
    symptom: str
    severity: int = 3
    time_of_day: str = "evening"
    duration_minutes: Optional[int] = None
    trigger_tags: List[str] = []
    notes: Optional[str] = None
    observed_at: Optional[str] = None

class CareInviteCreate(BaseModel):
    role: str = "caregiver"  # caregiver, clinician
    permission: str = "edit"  # edit, read_only
    note: Optional[str] = None
    expires_in_days: int = 14

class CareInviteAccept(BaseModel):
    code: str

class ExternalBotLinkCodeCreate(BaseModel):
    channel: str
    patient_user_id: Optional[str] = None
    expires_in_minutes: int = 20

class ExternalBotLinkPatientUpdate(BaseModel):
    patient_user_id: Optional[str] = None

class DoctorBotQueryRequest(BaseModel):
    text: str
    patient_user_id: Optional[str] = None
    prefer_voice: bool = False

class CaregiverReminderCreate(BaseModel):
    title: str
    time: str
    period: str
    category: str

class ShareLinkCreate(BaseModel):
    expires_in_days: int = 14

class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str = Field(default_factory=lambda: f"msg_{uuid.uuid4().hex[:12]}")
    user_id: str
    session_id: str
    role: str  # user or assistant
    content: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class ChatRequest(BaseModel):
    message: str
    session_id: str

# ==================== AUTHENTICATION ====================

async def get_current_user(request: Request) -> User:
    """Get current user from JWT token in cookie or Authorization header"""
    token = request.cookies.get("access_token")
    
    # Fallback to Authorization header
    if not token:
        auth_header = request.headers.get("Authorization")
        if auth_header and auth_header.startswith("Bearer "):
            token = auth_header.split(" ")[1]
    
    if not token:
        raise HTTPException(status_code=401, detail="Not authenticated")
    
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub")
        token_iat = payload.get("iat")
        if email is None:
            raise HTTPException(status_code=401, detail="Invalid authentication token")
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid authentication token")
    
    # Get user
    user_doc = await db.users.find_one(
        {"email": email},
        {"_id": 0}
    )
    
    if not user_doc:
        raise HTTPException(status_code=401, detail="User not found")

    account_status = user_doc.get("account_status", "active")
    if account_status == "suspended":
        raise HTTPException(status_code=403, detail="Account suspended. Contact support.")
    if account_status == "rejected":
        raise HTTPException(status_code=403, detail="Account rejected.")

    if user_doc.get("role") == "clinician":
        approval_status = user_doc.get("clinician_approval_status", "not_applicable")
        if approval_status == "pending":
            raise HTTPException(status_code=403, detail="Clinician account pending admin approval.")
        if approval_status in {"rejected", "suspended"}:
            raise HTTPException(status_code=403, detail="Clinician account not approved.")

    # Token revocation: reject tokens issued before password change
    pwd_changed = user_doc.get("password_changed_at")
    if pwd_changed and token_iat:
        if isinstance(pwd_changed, str):
            pwd_changed_ts = datetime.fromisoformat(pwd_changed).timestamp()
        elif isinstance(pwd_changed, datetime):
            pwd_changed_ts = pwd_changed.timestamp()
        else:
            pwd_changed_ts = 0
        if token_iat < pwd_changed_ts:
            raise HTTPException(status_code=401, detail="Session expired. Please login again.")

    return User(**user_doc)

def require_admin_user(current_user: User):
    if current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    if current_user.account_status != "active":
        raise HTTPException(status_code=403, detail="Admin account is not active")

async def resolve_target_user_id(
    current_user: User,
    target_user_id: Optional[str] = None,
    require_write: bool = False
) -> str:
    """Resolve patient user_id with caregiver/clinician access checks."""
    if current_user.role == "admin":
        return target_user_id or current_user.user_id

    if not target_user_id or target_user_id == current_user.user_id:
        return current_user.user_id

    link = await db.care_links.find_one(
        {
            "patient_id": target_user_id,
            "caregiver_id": current_user.user_id,
            "status": "accepted"
        },
        {"_id": 0}
    )
    if not link:
        raise HTTPException(status_code=403, detail="Access denied for this patient")

    if require_write and link.get("permission") != "edit":
        raise HTTPException(status_code=403, detail="Read-only access for this patient")

    return target_user_id

async def get_alert_recipients(patient_user_id: str) -> List[dict]:
    """Collect patient + accepted caregiver recipients for notifications."""
    recipients = []
    patient = await db.users.find_one(
        {"user_id": patient_user_id},
        {"_id": 0, "user_id": 1, "name": 1, "email": 1, "role": 1}
    )
    if patient:
        recipients.append(patient)

    links = await db.care_links.find(
        {"patient_id": patient_user_id, "status": "accepted"},
        {"_id": 0, "caregiver_id": 1}
    ).to_list(200)
    caregiver_ids = [l["caregiver_id"] for l in links]
    if caregiver_ids:
        caregivers = await db.users.find(
            {"user_id": {"$in": caregiver_ids}},
            {"_id": 0, "user_id": 1, "name": 1, "email": 1, "role": 1}
        ).to_list(200)
        recipients.extend(caregivers)

    return recipients

async def dispatch_proactive_hooks(
    patient_user_id: str,
    event_type: str,
    severity: str,
    payload: dict
) -> List[dict]:
    """
    Dispatch alert payload to configured push/email/SMS webhooks.
    Hook URLs:
      ALERT_PUSH_WEBHOOK_URL
      ALERT_EMAIL_WEBHOOK_URL
      ALERT_SMS_WEBHOOK_URL
    """
    recipients = await get_alert_recipients(patient_user_id)
    channels = [
        ("push", os.environ.get("ALERT_PUSH_WEBHOOK_URL", "").strip()),
        ("email", os.environ.get("ALERT_EMAIL_WEBHOOK_URL", "").strip()),
        ("sms", os.environ.get("ALERT_SMS_WEBHOOK_URL", "").strip()),
    ]

    envelope = {
        "event_type": event_type,
        "severity": severity,
        "patient_user_id": patient_user_id,
        "occurred_at": datetime.now(timezone.utc).isoformat(),
        "recipients": recipients,
        "payload": payload
    }

    results = []
    for channel, url in channels:
        if not url:
            results.append({"channel": channel, "sent": False, "reason": "hook_not_configured"})
            continue
        try:
            async with httpx.AsyncClient(timeout=8.0) as client:
                response = await client.post(url, json={**envelope, "channel": channel})
            ok = 200 <= response.status_code < 300
            result = {
                "channel": channel,
                "sent": ok,
                "status_code": response.status_code
            }
            if not ok:
                result["reason"] = (response.text or "non_2xx")[:240]
            results.append(result)
        except Exception as exc:
            results.append({"channel": channel, "sent": False, "reason": str(exc)[:240]})

    log_doc = {
        "id": f"notify_{uuid.uuid4().hex[:12]}",
        "patient_user_id": patient_user_id,
        "event_type": event_type,
        "severity": severity,
        "payload": payload,
        "recipients": recipients,
        "results": results,
        "created_at": datetime.now(timezone.utc).isoformat()
    }
    await db.alert_notification_logs.insert_one(log_doc)
    return results

async def ensure_default_safety_escalation_rules(user_id: str):
    existing = await db.safety_escalation_rules.count_documents({"user_id": user_id}, limit=1)
    if existing:
        return
    now_iso = datetime.now(timezone.utc).isoformat()
    docs = []
    for template in DEFAULT_ESCALATION_RULE_TEMPLATES:
        docs.append({
            "id": f"esc_rule_{uuid.uuid4().hex[:12]}",
            "user_id": user_id,
            "event_type": normalize_safety_event_type(template.get("event_type"), "all"),
            "min_severity": normalize_severity(template.get("min_severity"), "high"),
            "intervals_minutes": normalize_escalation_intervals(template.get("intervals_minutes")),
            "enabled": bool(template.get("enabled", True)),
            "created_at": now_iso,
            "updated_at": now_iso
        })
    if docs:
        await db.safety_escalation_rules.insert_many(docs)

async def resolve_escalation_rule(user_id: str, event_type: str, severity: str) -> Optional[dict]:
    await ensure_default_safety_escalation_rules(user_id)
    rules = await db.safety_escalation_rules.find(
        {
            "user_id": user_id,
            "enabled": True,
            "event_type": {"$in": [normalize_safety_event_type(event_type, "all"), "all"]}
        },
        {"_id": 0}
    ).to_list(100)
    if not rules:
        return None

    eligible = [r for r in rules if severity_at_least(severity, r.get("min_severity", "high"))]
    if not eligible:
        return None

    def _rule_sort_key(rule: dict) -> Tuple[int, int]:
        exact = 1 if rule.get("event_type") == event_type else 0
        min_sev = SEVERITY_ORDER.get(rule.get("min_severity", "high"), 0)
        return exact, min_sev

    eligible.sort(key=_rule_sort_key, reverse=True)
    return eligible[0]

async def create_safety_alert_with_escalation(
    user_id: str,
    event_type: str,
    severity: str,
    message: str,
    payload: dict,
    extra_fields: Optional[dict] = None
) -> Tuple[dict, List[dict]]:
    now = datetime.now(timezone.utc)
    normalized_event = normalize_safety_event_type(event_type, "geofence_exit")
    normalized_severity = normalize_severity(severity, "high")

    rule = await resolve_escalation_rule(user_id, normalized_event, normalized_severity)
    intervals = normalize_escalation_intervals((rule or {}).get("intervals_minutes") or [])
    next_at = (now + timedelta(minutes=intervals[0])).isoformat() if intervals else None

    alert_doc = {
        "id": f"alert_{uuid.uuid4().hex[:12]}",
        "user_id": user_id,
        "message": message,
        "event_type": normalized_event,
        "severity": normalized_severity,
        "acknowledged": False,
        "triggered_at": now.isoformat(),
        "escalation_rule_id": rule.get("id") if rule else None,
        "escalation_intervals_minutes": intervals,
        "escalation_stage": 0,
        "escalation_status": "active" if intervals else "none",
        "escalation_next_at": next_at,
        "last_escalated_at": None,
        "escalated_count": 0
    }
    if extra_fields:
        alert_doc.update(extra_fields)

    await db.safety_alerts.insert_one(alert_doc)
    hook_results = await dispatch_proactive_hooks(
        patient_user_id=user_id,
        event_type=normalized_event,
        severity=normalized_severity,
        payload={**payload, "alert_id": alert_doc["id"], "escalation_stage": 0}
    )
    await db.safety_escalation_events.insert_one({
        "id": f"escevt_{uuid.uuid4().hex[:12]}",
        "user_id": user_id,
        "alert_id": alert_doc["id"],
        "event_type": normalized_event,
        "stage": 0,
        "action": "initial_dispatch",
        "severity": normalized_severity,
        "hook_results": hook_results,
        "created_at": now.isoformat()
    })
    return alert_doc, hook_results

async def process_due_alert_escalations(
    user_id: str,
    max_alerts: int = 20,
    trigger_source: str = "manual"
) -> dict:
    now = datetime.now(timezone.utc)
    safe_limit = max(1, min(int(max_alerts), 100))
    due_alerts = await db.safety_alerts.find(
        {
            "user_id": user_id,
            "acknowledged": False,
            "escalation_status": "active",
            "escalation_next_at": {"$lte": now.isoformat()}
        },
        {"_id": 0}
    ).sort("escalation_next_at", 1).to_list(safe_limit)

    processed = []
    for alert in due_alerts:
        intervals = normalize_escalation_intervals(alert.get("escalation_intervals_minutes") or [])
        current_stage = int(alert.get("escalation_stage", 0))
        if not intervals or current_stage >= len(intervals):
            await db.safety_alerts.update_one(
                {"id": alert["id"], "user_id": user_id},
                {"$set": {"escalation_status": "max_reached", "escalation_next_at": None}}
            )
            continue

        next_stage = current_stage + 1
        escalated_severity = normalize_severity(alert.get("severity"), "high")
        if next_stage >= 2 and escalated_severity == "high":
            escalated_severity = "critical"
        elif next_stage >= 1 and escalated_severity == "medium":
            escalated_severity = "high"

        escalation_payload = {
            "alert_id": alert["id"],
            "original_event_type": alert.get("event_type"),
            "message": alert.get("message"),
            "escalation_stage": next_stage,
            "trigger_source": trigger_source
        }
        hook_results = await dispatch_proactive_hooks(
            patient_user_id=user_id,
            event_type=f"{alert.get('event_type', 'alert')}_escalation",
            severity=escalated_severity,
            payload=escalation_payload
        )

        next_at = None
        next_status = "max_reached"
        if next_stage < len(intervals):
            next_at = (now + timedelta(minutes=intervals[next_stage])).isoformat()
            next_status = "active"

        await db.safety_alerts.update_one(
            {"id": alert["id"], "user_id": user_id},
            {
                "$set": {
                    "severity": escalated_severity,
                    "escalation_stage": next_stage,
                    "escalation_status": next_status,
                    "escalation_next_at": next_at,
                    "last_escalated_at": now.isoformat(),
                    "escalated_count": int(alert.get("escalated_count", 0)) + 1
                }
            }
        )
        event_doc = {
            "id": f"escevt_{uuid.uuid4().hex[:12]}",
            "user_id": user_id,
            "alert_id": alert["id"],
            "event_type": alert.get("event_type"),
            "stage": next_stage,
            "action": "escalation_dispatch",
            "severity": escalated_severity,
            "hook_results": hook_results,
            "created_at": now.isoformat()
        }
        await db.safety_escalation_events.insert_one(event_doc)
        processed.append({
            "alert_id": alert["id"],
            "stage": next_stage,
            "severity": escalated_severity,
            "next_at": next_at
        })

    return {
        "processed_count": len(processed),
        "processed": processed,
        "run_at": now.isoformat()
    }

async def log_admin_audit(
    admin_user: User,
    action: str,
    target_user_id: str,
    details: Optional[dict] = None
):
    doc = {
        "id": f"admaudit_{uuid.uuid4().hex[:12]}",
        "admin_user_id": admin_user.user_id,
        "admin_name": admin_user.name,
        "action": action,
        "target_user_id": target_user_id,
        "details": details or {},
        "created_at": datetime.now(timezone.utc).isoformat()
    }
    await db.admin_audit_logs.insert_one(doc)

# ==================== AUTH ROUTES ====================

@api_router.post("/auth/register")
@rate_limit("5/minute")
async def register(request: Request, user_data: UserCreate):
    """Register a new user"""
    # Check if user exists
    existing_user = await db.users.find_one({"email": user_data.email})
    if existing_user:
        raise HTTPException(status_code=400, detail="Email already registered")
    
    requested_role = user_data.role if user_data.role in {"patient", "caregiver", "clinician", "admin"} else "patient"

    # Create user
    user_id = f"user_{uuid.uuid4().hex[:12]}"
    validate_password_strength(user_data.password)
    hashed_password = get_password_hash(user_data.password)
    role = requested_role
    account_status = "active"
    clinician_approval_status = "not_applicable"
    clinician_profile = None
    approved_by_admin_user_id = None
    approved_at = None
    approval_notes = None

    if requested_role == "admin":
        if not is_bootstrap_admin_email(user_data.email):
            raise HTTPException(status_code=403, detail="Admin accounts cannot be self-registered.")
    elif requested_role == "clinician":
        account_status = "pending"
        clinician_approval_status = "pending"
        clinician_profile = {
            "license_number": user_data.license_number,
            "medical_organization": user_data.medical_organization,
            "jurisdiction": user_data.jurisdiction
        }
    
    new_user = {
        "user_id": user_id,
        "email": user_data.email,
        "name": user_data.name,
        "role": role,
        "requested_role": requested_role,
        "account_status": account_status,
        "clinician_approval_status": clinician_approval_status,
        "clinician_profile": clinician_profile,
        "approved_by_admin_user_id": approved_by_admin_user_id,
        "approved_at": approved_at,
        "approval_notes": approval_notes,
        "hashed_password": hashed_password,
        "picture": None,
        "linked_patient_ids": [],
        "password_changed_at": datetime.now(timezone.utc).isoformat(),
        "created_at": datetime.now(timezone.utc),
        "updated_at": datetime.now(timezone.utc).isoformat()
    }

    if user_data.referral_code:
        referrer = await db.users.find_one({"referral_code": user_data.referral_code})
        if referrer:
            new_user["referred_by"] = user_data.referral_code

    await db.users.insert_one(new_user)
    if requested_role == "clinician":
        return {
            "message": "Clinician registration submitted. Waiting for admin approval.",
            "user_id": user_id,
            "requires_approval": True
        }
    return {"message": "User registered successfully", "user_id": user_id, "requires_approval": False}

@api_router.post("/auth/login")
@rate_limit("10/minute")
async def login(request: Request, response: Response, form_data: UserLogin):
    """Login user and set JWT cookie"""
    # Find user
    user_doc = await db.users.find_one({"email": form_data.email})
    if not user_doc or not verify_password(form_data.password, user_doc["hashed_password"]):
        raise HTTPException(status_code=401, detail="Incorrect email or password")

    account_status = user_doc.get("account_status", "active")
    if account_status == "pending":
        raise HTTPException(status_code=403, detail="Account pending approval.")
    if account_status == "suspended":
        raise HTTPException(status_code=403, detail="Account suspended.")
    if account_status == "rejected":
        raise HTTPException(status_code=403, detail="Account rejected.")

    if user_doc.get("role") == "clinician":
        approval_status = user_doc.get("clinician_approval_status", "not_applicable")
        if approval_status == "pending":
            raise HTTPException(status_code=403, detail="Clinician account pending admin approval.")
        if approval_status in {"rejected", "suspended"}:
            raise HTTPException(status_code=403, detail="Clinician account not approved.")
    
    # Create access token
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={
            "sub": user_doc["email"],
            "user_id": user_doc["user_id"],
            "role": user_doc.get("role", "patient")
        },
        expires_delta=access_token_expires
    )
    
    # Set cookies
    response.set_cookie(
        key="access_token",
        value=access_token,
        httponly=True,
        secure=COOKIE_SECURE,
        samesite=COOKIE_SAMESITE,
        path="/",
        max_age=ACCESS_TOKEN_EXPIRE_MINUTES * 60
    )
    refresh_token = create_refresh_token(user_doc["user_id"])
    response.set_cookie(
        key="refresh_token",
        value=refresh_token,
        httponly=True,
        secure=COOKIE_SECURE,
        samesite=COOKIE_SAMESITE,
        path="/api/auth/refresh",
        max_age=REFRESH_TOKEN_EXPIRE_DAYS * 24 * 60 * 60
    )

    return {"message": "Login successful", "user": User(**user_doc)}

@api_router.get("/auth/me")
async def get_me(current_user: User = Depends(get_current_user)):
    """Get current user info"""
    return current_user.model_dump()

@api_router.post("/auth/logout")
async def logout(response: Response):
    """Logout user"""
    response.delete_cookie(key="access_token", path="/")
    response.delete_cookie(key="refresh_token", path="/api/auth/refresh")
    return {"message": "Logged out"}

@api_router.delete("/auth/account")
async def delete_account(current_user: User = Depends(get_current_user)):
    """Delete user account and all associated data (GDPR right to be forgotten)."""
    guard_demo_write(current_user)
    uid = current_user.user_id
    collections = [
        "memories", "family_members", "reminders", "medications",
        "medication_intake_logs", "destinations", "chat_messages",
        "safety_zones", "safety_alerts", "safety_fall_events",
        "safety_escalation_events", "safety_escalation_rules",
        "safety_emergency_contacts", "safety_location_shares",
        "mood_checkins", "bpsd_observations",
        "care_instructions", "care_instruction_chunks",
        "care_links", "care_invites",
        "push_subscriptions", "external_bot_links",
        "external_bot_message_logs", "alert_notification_logs",
        "admin_audit_logs"
    ]
    for coll in collections:
        await db[coll].delete_many({"user_id": uid})
    await db.users.delete_one({"user_id": uid})
    return {"message": "Account and all data deleted"}

@api_router.get("/auth/export")
async def export_account_data(current_user: User = Depends(get_current_user)):
    """Export all user data as JSON (GDPR data portability)."""
    uid = current_user.user_id
    export = {}
    for coll_name in ["memories", "family_members", "reminders", "medications",
                       "medication_intake_logs", "chat_messages", "mood_checkins",
                       "bpsd_observations", "care_instructions", "destinations",
                       "safety_zones", "safety_emergency_contacts"]:
        docs = await db[coll_name].find({"user_id": uid}, {"_id": 0, "embedding": 0}).to_list(5000)
        # Convert datetime objects to strings for JSON serialization
        for doc in docs:
            for k, v in doc.items():
                if isinstance(v, datetime):
                    doc[k] = v.isoformat()
        export[coll_name] = docs
    user_doc = await db.users.find_one({"user_id": uid}, {"_id": 0, "hashed_password": 0})
    if user_doc:
        for k, v in user_doc.items():
            if isinstance(v, datetime):
                user_doc[k] = v.isoformat()
    export["profile"] = user_doc
    return export

@api_router.post("/auth/refresh")
@rate_limit("20/minute")
async def refresh_access_token(request: Request, response: Response):
    """Refresh access token using refresh token cookie"""
    token = request.cookies.get("refresh_token")
    if not token:
        raise HTTPException(status_code=401, detail="No refresh token")
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id = payload.get("sub_refresh")
        if not user_id:
            raise HTTPException(status_code=401, detail="Invalid refresh token")
    except JWTError:
        raise HTTPException(status_code=401, detail="Expired refresh token")
    user_doc = await db.users.find_one({"user_id": user_id})
    if not user_doc:
        raise HTTPException(status_code=401, detail="User not found")
    new_access = create_access_token(
        data={"sub": user_doc["email"], "user_id": user_id, "role": user_doc.get("role", "patient")},
        expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    )
    response.set_cookie(
        key="access_token", value=new_access, httponly=True,
        secure=COOKIE_SECURE, samesite=COOKIE_SAMESITE, path="/",
        max_age=ACCESS_TOKEN_EXPIRE_MINUTES * 60
    )
    return {"message": "Token refreshed"}


# ==================== DEMO MODE ====================

DEMO_USER_ID = "demo_patient_001"
DEMO_USER_EMAIL = "demo@alzahelp.app"

async def _seed_demo_account():
    """Create or refresh the demo account with sample data."""
    existing = await db.users.find_one({"user_id": DEMO_USER_ID})
    if existing:
        return

    await db.users.insert_one({
        "user_id": DEMO_USER_ID,
        "email": DEMO_USER_EMAIL,
        "name": "Maria Garcia",
        "role": "patient",
        "account_status": "active",
        "is_demo": True,
        "hashed_password": get_password_hash("DemoAccount1!"),
        "linked_patient_ids": [],
        "subscription_tier": "premium",
        "created_at": datetime.now(timezone.utc),
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "password_changed_at": datetime.now(timezone.utc).isoformat(),
    })

    family = [
        {"id": "fam_demo_01", "user_id": DEMO_USER_ID, "name": "Carlos Garcia", "relationship": "Son", "phone": "+1-555-0101", "notes": "Visits every Sunday"},
        {"id": "fam_demo_02", "user_id": DEMO_USER_ID, "name": "Sofia Garcia", "relationship": "Daughter", "phone": "+1-555-0102", "notes": "Lives nearby, helps with groceries"},
        {"id": "fam_demo_03", "user_id": DEMO_USER_ID, "name": "Pedro Garcia", "relationship": "Husband", "phone": "+1-555-0100", "notes": "Primary caregiver"},
    ]
    for fm in family:
        fm["created_at"] = datetime.now(timezone.utc).isoformat()
        await db.family_members.insert_one(fm)

    meds = [
        {"id": "med_demo_01", "user_id": DEMO_USER_ID, "name": "Donepezil", "dosage": "10mg", "frequency": "daily", "scheduled_times": ["08:00"], "active": True, "notes": "Take with breakfast"},
        {"id": "med_demo_02", "user_id": DEMO_USER_ID, "name": "Memantine", "dosage": "20mg", "frequency": "daily", "scheduled_times": ["09:00"], "active": True, "notes": "For cognitive symptoms"},
        {"id": "med_demo_03", "user_id": DEMO_USER_ID, "name": "Vitamin D", "dosage": "2000 IU", "frequency": "daily", "scheduled_times": ["08:00"], "active": True, "notes": "With food"},
        {"id": "med_demo_04", "user_id": DEMO_USER_ID, "name": "Melatonin", "dosage": "3mg", "frequency": "daily", "scheduled_times": ["21:00"], "active": True, "notes": "Before bed for sleep"},
    ]
    for med in meds:
        med["created_at"] = datetime.now(timezone.utc).isoformat()
        await db.medications.insert_one(med)

    memories = [
        {"id": "mem_demo_01", "user_id": DEMO_USER_ID, "title": "Wedding Day", "date": "1985-06-15", "year": 1985, "location": "Santiago, Chile", "description": "Pedro and I got married at the beautiful church on Avenida Providencia. Sofia was the flower girl and Carlos carried the rings.", "people": ["Pedro Garcia", "Sofia Garcia", "Carlos Garcia"], "photos": [], "category": "milestone"},
        {"id": "mem_demo_02", "user_id": DEMO_USER_ID, "title": "Carlos's Graduation", "date": "2010-12-20", "year": 2010, "location": "University of Chile", "description": "Carlos graduated with honors in engineering. The whole family was there. We celebrated at that restaurant by the park.", "people": ["Carlos Garcia", "Pedro Garcia"], "photos": [], "category": "milestone"},
        {"id": "mem_demo_03", "user_id": DEMO_USER_ID, "title": "Summer at the Beach", "date": "2019-01-10", "year": 2019, "location": "Viña del Mar", "description": "We spent two weeks at the coast. The grandchildren loved building sandcastles. Sofia made her famous empanadas.", "people": ["Sofia Garcia", "Pedro Garcia"], "photos": [], "category": "vacation"},
        {"id": "mem_demo_04", "user_id": DEMO_USER_ID, "title": "Morning Walk Routine", "date": "2025-11-01", "year": 2025, "location": "Neighborhood Park", "description": "Pedro and I walk every morning around the park. We feed the pigeons near the fountain. It helps me feel calm.", "people": ["Pedro Garcia"], "photos": [], "category": "routine"},
        {"id": "mem_demo_05", "user_id": DEMO_USER_ID, "title": "Cooking with Sofia", "date": "2025-12-25", "year": 2025, "location": "Home", "description": "Sofia came over for Christmas and we made pastel de choclo together. She said it tasted just like abuela's recipe.", "people": ["Sofia Garcia"], "photos": [], "category": "family"},
    ]
    for mem in memories:
        mem["created_at"] = datetime.now(timezone.utc).isoformat()
        mem["updated_at"] = datetime.now(timezone.utc).isoformat()
        mem["search_text"] = f"{mem['title']} {mem['date']} {mem.get('location', '')} {mem['description']} {', '.join(mem['people'])}".lower()
        await db.memories.insert_one(mem)

    for i in range(7):
        day = datetime.now(timezone.utc) - timedelta(days=i)
        await db.mood_checkins.insert_one({
            "id": f"mood_demo_{i:02d}",
            "user_id": DEMO_USER_ID,
            "mood_score": [3, 2, 3, 2, 3, 2, 3][i],
            "energy_level": [2, 2, 3, 1, 2, 3, 2][i],
            "anxiety_level": [1, 2, 1, 2, 1, 1, 2][i],
            "sleep_quality": [3, 2, 3, 2, 2, 3, 3][i],
            "appetite": [2, 3, 2, 2, 3, 2, 3][i],
            "notes": ["Good day", "Felt a bit confused", "Enjoyed the walk", "Tired", "Happy to see Carlos", "Quiet day", "Slept well"][i],
            "source": "patient",
            "created_at": day.isoformat(),
        })

    reminders = [
        {"id": "rem_demo_01", "user_id": DEMO_USER_ID, "title": "Morning walk with Pedro", "time": "09:00", "active": True},
        {"id": "rem_demo_02", "user_id": DEMO_USER_ID, "title": "Call Sofia", "time": "15:00", "active": True},
        {"id": "rem_demo_03", "user_id": DEMO_USER_ID, "title": "Water the plants", "time": "10:00", "active": True},
    ]
    for rem in reminders:
        rem["created_at"] = datetime.now(timezone.utc).isoformat()
        await db.reminders.insert_one(rem)


def is_demo_user(current_user) -> bool:
    """Check if the current user is the demo account."""
    uid = getattr(current_user, 'user_id', '') if not isinstance(current_user, dict) else current_user.get('user_id', '')
    return uid == DEMO_USER_ID


def guard_demo_write(current_user):
    """Block write operations for demo users."""
    if is_demo_user(current_user):
        raise HTTPException(status_code=403, detail="Demo mode is read-only. Sign up to save your own data!")


@api_router.post("/auth/demo")
@rate_limit("20/minute")
async def start_demo(request: Request, response: Response):
    """Start a demo session with pre-populated data."""
    await _seed_demo_account()
    access_token_expires = timedelta(hours=2)
    access_token = create_access_token(
        data={"sub": DEMO_USER_EMAIL, "user_id": DEMO_USER_ID, "role": "patient"},
        expires_delta=access_token_expires
    )
    response.set_cookie(
        key="access_token", value=access_token, httponly=True,
        secure=COOKIE_SECURE, samesite=COOKIE_SAMESITE, path="/",
        max_age=2 * 60 * 60
    )
    return {"message": "Demo started", "user": {"user_id": DEMO_USER_ID, "name": "Maria Garcia", "role": "patient", "is_demo": True}}


# ==================== BILLING / SUBSCRIPTIONS ====================

def require_premium(user):
    """Raise 403 if user is not on the premium tier (or within grace period)."""
    doc = user if isinstance(user, dict) else user.dict()
    tier = doc.get("subscription_tier", "free")
    if tier == "premium":
        return
    expires = doc.get("subscription_expires_at")
    if expires and expires > datetime.now(timezone.utc):
        return  # still in grace period
    raise HTTPException(
        status_code=403,
        detail="This feature requires AlzaHelp Premium. Upgrade at your dashboard.",
    )


def _is_premium(user) -> bool:
    """Check if user has premium without raising."""
    doc = user if isinstance(user, dict) else user.dict()
    if doc.get("subscription_tier") == "premium":
        return True
    expires = doc.get("subscription_expires_at")
    return bool(expires and expires > datetime.now(timezone.utc))


@api_router.get("/billing/status")
async def billing_status(current_user: User = Depends(get_current_user)):
    """Return current subscription status."""
    user_doc = await db.users.find_one({"user_id": current_user.user_id}, {"_id": 0})
    expires = user_doc.get("subscription_expires_at")
    tier = user_doc.get("subscription_tier", "free")
    patient_limit = PATIENT_LIMITS.get(tier, 3)
    patient_count = await db.care_links.count_documents(
        {"caregiver_id": current_user.user_id, "status": "accepted"}
    )
    return {
        "tier": tier,
        "expires_at": expires.isoformat() if isinstance(expires, datetime) else expires,
        "has_stripe": _HAS_STRIPE,
        "patient_limit": patient_limit,
        "patient_count": patient_count,
    }


@api_router.post("/billing/create-checkout")
async def billing_create_checkout(current_user: User = Depends(get_current_user)):
    """Create a Stripe Checkout session for Premium subscription."""
    if not _HAS_STRIPE:
        raise HTTPException(status_code=503, detail="Payments not configured")
    if not STRIPE_PRICE_ID:
        raise HTTPException(status_code=503, detail="Stripe price not configured")

    frontend_url = os.environ.get("CORS_ORIGINS", "http://localhost:3000").split(",")[0].strip()

    # Find or create Stripe customer
    user_doc = await db.users.find_one({"user_id": current_user.user_id})
    stripe_cid = user_doc.get("stripe_customer_id")

    if not stripe_cid:
        customer = stripe_lib.Customer.create(
            email=current_user.email,
            metadata={"user_id": current_user.user_id},
        )
        stripe_cid = customer.id
        await db.users.update_one(
            {"user_id": current_user.user_id},
            {"$set": {"stripe_customer_id": stripe_cid}},
        )

    session = stripe_lib.checkout.Session.create(
        customer=stripe_cid,
        mode="subscription",
        line_items=[{"price": STRIPE_PRICE_ID, "quantity": 1}],
        success_url=f"{frontend_url}/dashboard?billing=success",
        cancel_url=f"{frontend_url}/dashboard?billing=cancel",
    )
    return {"url": session.url}


@api_router.post("/billing/create-portal")
async def billing_create_portal(current_user: User = Depends(get_current_user)):
    """Create a Stripe Customer Portal session for managing subscription."""
    if not _HAS_STRIPE:
        raise HTTPException(status_code=503, detail="Payments not configured")

    user_doc = await db.users.find_one({"user_id": current_user.user_id})
    stripe_cid = user_doc.get("stripe_customer_id")
    if not stripe_cid:
        raise HTTPException(status_code=400, detail="No billing account found")

    frontend_url = os.environ.get("CORS_ORIGINS", "http://localhost:3000").split(",")[0].strip()
    session = stripe_lib.billing_portal.Session.create(
        customer=stripe_cid,
        return_url=f"{frontend_url}/dashboard",
    )
    return {"url": session.url}


@api_router.post("/billing/webhook")
async def billing_webhook(request: Request):
    """Handle Stripe webhook events."""
    if not _HAS_STRIPE or not STRIPE_WEBHOOK_SECRET:
        raise HTTPException(status_code=503, detail="Webhooks not configured")

    payload = await request.body()
    sig = request.headers.get("stripe-signature")
    try:
        event = stripe_lib.Webhook.construct_event(payload, sig, STRIPE_WEBHOOK_SECRET)
    except (ValueError, stripe_lib.error.SignatureVerificationError):
        raise HTTPException(status_code=400, detail="Invalid webhook signature")

    event_type = event["type"]
    data_obj = event["data"]["object"]
    logger = logging.getLogger(__name__)

    if event_type == "checkout.session.completed":
        customer_id = data_obj.get("customer")
        subscription_id = data_obj.get("subscription")
        if customer_id:
            await db.users.update_one(
                {"stripe_customer_id": customer_id},
                {"$set": {
                    "subscription_tier": "premium",
                    "stripe_subscription_id": subscription_id,
                }},
            )
            logger.info("User upgraded to premium via checkout: customer=%s", customer_id)

    elif event_type == "customer.subscription.updated":
        customer_id = data_obj.get("customer")
        period_end = data_obj.get("current_period_end")
        if customer_id and period_end:
            await db.users.update_one(
                {"stripe_customer_id": customer_id},
                {"$set": {
                    "subscription_expires_at": datetime.fromtimestamp(period_end, tz=timezone.utc),
                }},
            )

    elif event_type == "customer.subscription.deleted":
        customer_id = data_obj.get("customer")
        if customer_id:
            grace = datetime.now(timezone.utc) + timedelta(days=3)
            await db.users.update_one(
                {"stripe_customer_id": customer_id},
                {"$set": {
                    "subscription_tier": "free",
                    "stripe_subscription_id": None,
                    "subscription_expires_at": grace,
                }},
            )
            logger.info("Subscription cancelled for customer=%s, grace until %s", customer_id, grace.isoformat())

    elif event_type == "invoice.payment_failed":
        customer_id = data_obj.get("customer")
        if customer_id:
            user_doc = await db.users.find_one({"stripe_customer_id": customer_id})
            if user_doc:
                try:
                    await send_push_to_user(
                        user_doc["user_id"],
                        "Payment Failed",
                        "Your AlzaHelp Premium payment failed. Please update your payment method.",
                        "/dashboard?billing=failed",
                    )
                except Exception:
                    pass

    return {"received": True}


# ==================== ADMIN (CLINICIAN GOVERNANCE) ====================

@api_router.get("/admin/clinicians/pending", response_model=List[dict])
async def admin_list_pending_clinicians(current_user: User = Depends(get_current_user)):
    require_admin_user(current_user)
    users = await db.users.find(
        {"role": "clinician", "clinician_approval_status": "pending"},
        {"_id": 0, "hashed_password": 0}
    ).sort("created_at", -1).to_list(500)
    return users

@api_router.get("/admin/clinicians", response_model=List[dict])
async def admin_list_clinicians(
    status: str = "all",
    current_user: User = Depends(get_current_user)
):
    require_admin_user(current_user)
    query = {"role": "clinician"}
    normalized = (status or "all").strip().lower()
    if normalized != "all":
        if normalized not in {"pending", "approved", "rejected", "suspended"}:
            raise HTTPException(status_code=400, detail="Invalid clinician status filter")
        query["clinician_approval_status"] = normalized
    users = await db.users.find(query, {"_id": 0, "hashed_password": 0}).sort("created_at", -1).to_list(1000)
    return users

@api_router.post("/admin/clinicians/{clinician_user_id}/approve", response_model=dict)
async def admin_approve_clinician(
    clinician_user_id: str,
    payload: AdminClinicianDecision,
    current_user: User = Depends(get_current_user)
):
    require_admin_user(current_user)
    clinician = await db.users.find_one({"user_id": clinician_user_id, "role": "clinician"}, {"_id": 0, "hashed_password": 0})
    if not clinician:
        raise HTTPException(status_code=404, detail="Clinician not found")

    now_iso = datetime.now(timezone.utc).isoformat()
    update = {
        "account_status": "active",
        "clinician_approval_status": "approved",
        "approved_by_admin_user_id": current_user.user_id,
        "approved_at": now_iso,
        "approval_notes": payload.notes,
        "updated_at": now_iso
    }
    await db.users.update_one({"user_id": clinician_user_id}, {"$set": update})
    await log_admin_audit(
        current_user,
        "approve_clinician",
        clinician_user_id,
        {"notes": payload.notes}
    )
    updated = await db.users.find_one({"user_id": clinician_user_id}, {"_id": 0, "hashed_password": 0})
    return updated

@api_router.post("/admin/clinicians/{clinician_user_id}/reject", response_model=dict)
async def admin_reject_clinician(
    clinician_user_id: str,
    payload: AdminClinicianDecision,
    current_user: User = Depends(get_current_user)
):
    require_admin_user(current_user)
    clinician = await db.users.find_one({"user_id": clinician_user_id, "role": "clinician"}, {"_id": 0, "hashed_password": 0})
    if not clinician:
        raise HTTPException(status_code=404, detail="Clinician not found")

    now_iso = datetime.now(timezone.utc).isoformat()
    update = {
        "account_status": "rejected",
        "clinician_approval_status": "rejected",
        "approved_by_admin_user_id": current_user.user_id,
        "approved_at": now_iso,
        "approval_notes": payload.notes,
        "updated_at": now_iso
    }
    await db.users.update_one({"user_id": clinician_user_id}, {"$set": update})
    await log_admin_audit(
        current_user,
        "reject_clinician",
        clinician_user_id,
        {"notes": payload.notes}
    )
    updated = await db.users.find_one({"user_id": clinician_user_id}, {"_id": 0, "hashed_password": 0})
    return updated

@api_router.post("/admin/clinicians/{clinician_user_id}/suspend", response_model=dict)
async def admin_suspend_clinician(
    clinician_user_id: str,
    payload: AdminClinicianDecision,
    current_user: User = Depends(get_current_user)
):
    require_admin_user(current_user)
    clinician = await db.users.find_one({"user_id": clinician_user_id, "role": "clinician"}, {"_id": 0, "hashed_password": 0})
    if not clinician:
        raise HTTPException(status_code=404, detail="Clinician not found")

    now_iso = datetime.now(timezone.utc).isoformat()
    update = {
        "account_status": "suspended",
        "clinician_approval_status": "suspended",
        "approval_notes": payload.notes,
        "updated_at": now_iso
    }
    await db.users.update_one({"user_id": clinician_user_id}, {"$set": update})
    await log_admin_audit(
        current_user,
        "suspend_clinician",
        clinician_user_id,
        {"notes": payload.notes}
    )
    updated = await db.users.find_one({"user_id": clinician_user_id}, {"_id": 0, "hashed_password": 0})
    return updated

@api_router.post("/admin/clinicians/{clinician_user_id}/reactivate", response_model=dict)
async def admin_reactivate_clinician(
    clinician_user_id: str,
    payload: AdminClinicianDecision,
    current_user: User = Depends(get_current_user)
):
    require_admin_user(current_user)
    clinician = await db.users.find_one({"user_id": clinician_user_id, "role": "clinician"}, {"_id": 0, "hashed_password": 0})
    if not clinician:
        raise HTTPException(status_code=404, detail="Clinician not found")

    now_iso = datetime.now(timezone.utc).isoformat()
    update = {
        "account_status": "active",
        "clinician_approval_status": "approved",
        "approval_notes": payload.notes if payload.notes is not None else clinician.get("approval_notes"),
        "updated_at": now_iso
    }
    await db.users.update_one({"user_id": clinician_user_id}, {"$set": update})
    await log_admin_audit(
        current_user,
        "reactivate_clinician",
        clinician_user_id,
        {"notes": payload.notes}
    )
    updated = await db.users.find_one({"user_id": clinician_user_id}, {"_id": 0, "hashed_password": 0})
    return updated

@api_router.get("/admin/audit", response_model=List[dict])
async def admin_audit_log(
    limit: int = 200,
    current_user: User = Depends(get_current_user)
):
    require_admin_user(current_user)
    safe_limit = max(1, min(limit, 1000))
    logs = await db.admin_audit_logs.find({}, {"_id": 0}).sort("created_at", -1).to_list(safe_limit)
    return logs

# ==================== FILE UPLOAD (MongoDB GridFS) ====================

@api_router.post("/upload")
async def upload_file(
    file: UploadFile = File(...),
    target_user_id: Optional[str] = None,
    current_user: User = Depends(get_current_user)
):
    """Upload a file (photo or voice note) to MongoDB GridFS"""
    guard_demo_write(current_user)
    owner_user_id = await resolve_target_user_id(
        current_user,
        target_user_id,
        require_write=True if target_user_id else False
    )

    # Read file content
    content = await file.read()
    ext = validate_upload(file, content)
    filename = f"{owner_user_id}_{uuid.uuid4().hex[:8]}{ext}"
    content_type = file.content_type or 'application/octet-stream'

    # Store in GridFS
    file_id = await fs_bucket.upload_from_stream(
        filename,
        io.BytesIO(content),
        metadata={
            "user_id": owner_user_id,
            "content_type": content_type,
            "original_filename": Path(file.filename).name if file.filename else filename,
            "uploaded_at": datetime.now(timezone.utc).isoformat()
        }
    )
    
    # Return URL to retrieve file
    return {"url": f"/api/files/{filename}", "filename": filename, "file_id": str(file_id)}

@api_router.post("/upload/multiple")
async def upload_multiple_files(
    files: List[UploadFile] = File(...),
    target_user_id: Optional[str] = None,
    current_user: User = Depends(get_current_user)
):
    """Upload multiple files to MongoDB GridFS"""
    guard_demo_write(current_user)
    owner_user_id = await resolve_target_user_id(
        current_user,
        target_user_id,
        require_write=True if target_user_id else False
    )

    urls = []
    for file in files:
        content = await file.read()
        ext = validate_upload(file, content)
        filename = f"{owner_user_id}_{uuid.uuid4().hex[:8]}{ext}"
        content_type = file.content_type or 'application/octet-stream'

        await fs_bucket.upload_from_stream(
            filename,
            io.BytesIO(content),
            metadata={
                "user_id": owner_user_id,
                "content_type": content_type,
                "original_filename": Path(file.filename).name if file.filename else filename,
                "uploaded_at": datetime.now(timezone.utc).isoformat()
            }
        )
        
        urls.append(f"/api/files/{filename}")
    
    return {"urls": urls}

@api_router.get("/files/{filename}")
async def get_file(filename: str, request: Request):
    """Retrieve a file from MongoDB GridFS with ownership verification"""
    try:
        grid_out = await fs_bucket.open_download_stream_by_name(filename)
        content = await grid_out.read()
        metadata = grid_out.metadata or {}
        content_type = metadata.get('content_type', 'application/octet-stream')
        file_owner = metadata.get('user_id', '')

        # Demo account files are public
        is_demo_file = file_owner.startswith("demo_")

        if not is_demo_file:
            current_user = await get_current_user(request)
            if current_user.user_id != file_owner and current_user.role != "admin":
                link = await db.care_links.find_one({
                    "patient_id": file_owner,
                    "caregiver_id": current_user.user_id,
                    "status": "accepted"
                })
                if not link:
                    raise HTTPException(status_code=403, detail="Access denied")

        return StreamingResponse(
            io.BytesIO(content),
            media_type=content_type,
            headers={
                "Content-Disposition": f"inline; filename={filename}",
                "Cache-Control": "private, max-age=3600"
            }
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving file {filename}: {e}")
        raise HTTPException(status_code=404, detail="File not found")

# ==================== FAMILY MEMBERS ====================

@api_router.get("/family", response_model=List[dict])
async def get_family_members(current_user: User = Depends(get_current_user)):
    """Get all family members for current user"""
    members = await db.family_members.find(
        {"user_id": current_user.user_id},
        {"_id": 0, "embedding": 0}
    ).to_list(100)
    return members

@api_router.post("/family", response_model=dict)
async def create_family_member(
    member: FamilyMemberCreate,
    current_user: User = Depends(get_current_user)
):
    """Create a new family member"""
    guard_demo_write(current_user)
    member_obj = FamilyMember(
        user_id=current_user.user_id,
        **member.model_dump()
    )
    
    doc = member_obj.model_dump()
    doc['created_at'] = doc['created_at'].isoformat()
    doc['updated_at'] = doc['updated_at'].isoformat()
    
    # Store searchable text for keyword search
    doc['search_text'] = f"{member.name} {member.relationship} {member.relationship_label} {member.notes or ''}".lower()
    
    await db.family_members.insert_one(doc)
    asyncio.create_task(_embed_and_store_family(doc))

    # Return without search_text and _id
    if 'search_text' in doc:
        del doc['search_text']
    if '_id' in doc:
        del doc['_id']
    return doc

@api_router.put("/family/{member_id}", response_model=dict)
async def update_family_member(
    member_id: str,
    member: FamilyMemberUpdate,
    current_user: User = Depends(get_current_user)
):
    """Update a family member"""
    guard_demo_write(current_user)
    update_data = {k: v for k, v in member.model_dump().items() if v is not None}
    update_data['updated_at'] = datetime.now(timezone.utc).isoformat()
    
    # Update search text if name or notes changed
    if 'name' in update_data or 'notes' in update_data or 'relationship' in update_data:
        existing = await db.family_members.find_one(
            {"id": member_id, "user_id": current_user.user_id},
            {"_id": 0}
        )
        if existing:
            name = update_data.get('name', existing.get('name', ''))
            relationship = update_data.get('relationship', existing.get('relationship', ''))
            relationship_label = update_data.get('relationship_label', existing.get('relationship_label', ''))
            notes = update_data.get('notes', existing.get('notes', ''))
            update_data['search_text'] = f"{name} {relationship} {relationship_label} {notes}".lower()
    
    result = await db.family_members.update_one(
        {"id": member_id, "user_id": current_user.user_id},
        {"$set": update_data}
    )
    
    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="Family member not found")

    updated = await db.family_members.find_one(
        {"id": member_id},
        {"_id": 0, "search_text": 0}
    )
    if updated:
        asyncio.create_task(_embed_and_store_family(updated))
    return updated

@api_router.delete("/family/{member_id}")
async def delete_family_member(
    member_id: str,
    current_user: User = Depends(get_current_user)
):
    """Delete a family member"""
    guard_demo_write(current_user)
    result = await db.family_members.delete_one(
        {"id": member_id, "user_id": current_user.user_id}
    )
    
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Family member not found")
    
    return {"message": "Family member deleted"}

# ==================== MEMORIES ====================

@api_router.get("/memories", response_model=List[dict])
async def get_memories(current_user: User = Depends(get_current_user)):
    """Get all memories for current user"""
    memories = await db.memories.find(
        {"user_id": current_user.user_id},
        {"_id": 0, "embedding": 0}
    ).sort("year", -1).to_list(500)
    return memories

@api_router.post("/memories", response_model=dict)
async def create_memory(
    memory: MemoryCreate,
    current_user: User = Depends(get_current_user)
):
    """Create a new memory"""
    guard_demo_write(current_user)
    memory_obj = Memory(
        user_id=current_user.user_id,
        **memory.model_dump()
    )
    
    doc = memory_obj.model_dump()
    doc['created_at'] = doc['created_at'].isoformat()
    doc['updated_at'] = doc['updated_at'].isoformat()
    
    # Store searchable text for keyword search
    people_str = ", ".join(memory.people) if memory.people else ""
    doc['search_text'] = f"{memory.title} {memory.date} {memory.location or ''} {memory.description} {people_str}".lower()
    
    await db.memories.insert_one(doc)
    asyncio.create_task(_embed_and_store_memory(doc))

    # Return without search_text and _id
    if 'search_text' in doc:
        del doc['search_text']
    if '_id' in doc:
        del doc['_id']
    return doc

@api_router.put("/memories/{memory_id}", response_model=dict)
async def update_memory(
    memory_id: str,
    memory: MemoryUpdate,
    current_user: User = Depends(get_current_user)
):
    """Update a memory"""
    guard_demo_write(current_user)
    update_data = {k: v for k, v in memory.model_dump().items() if v is not None}
    update_data['updated_at'] = datetime.now(timezone.utc).isoformat()
    
    # Update search text
    existing = await db.memories.find_one(
        {"id": memory_id, "user_id": current_user.user_id},
        {"_id": 0}
    )
    if existing:
        title = update_data.get('title', existing.get('title', ''))
        date = update_data.get('date', existing.get('date', ''))
        location = update_data.get('location', existing.get('location', ''))
        description = update_data.get('description', existing.get('description', ''))
        people = update_data.get('people', existing.get('people', []))
        people_str = ", ".join(people) if people else ""
        update_data['search_text'] = f"{title} {date} {location} {description} {people_str}".lower()
    
    result = await db.memories.update_one(
        {"id": memory_id, "user_id": current_user.user_id},
        {"$set": update_data}
    )
    
    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="Memory not found")

    updated = await db.memories.find_one(
        {"id": memory_id},
        {"_id": 0, "search_text": 0}
    )
    if updated:
        asyncio.create_task(_embed_and_store_memory(updated))
    return updated

@api_router.delete("/memories/{memory_id}")
async def delete_memory(
    memory_id: str,
    current_user: User = Depends(get_current_user)
):
    """Delete a memory"""
    guard_demo_write(current_user)
    result = await db.memories.delete_one(
        {"id": memory_id, "user_id": current_user.user_id}
    )
    
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Memory not found")
    
    return {"message": "Memory deleted"}

# ==================== REMINDERS ====================

@api_router.get("/reminders", response_model=List[dict])
async def get_reminders(current_user: User = Depends(get_current_user)):
    """Get all reminders for current user"""
    reminders = await db.reminders.find(
        {"user_id": current_user.user_id},
        {"_id": 0}
    ).to_list(100)
    return reminders

@api_router.post("/reminders", response_model=dict)
async def create_reminder(
    reminder: ReminderCreate,
    current_user: User = Depends(get_current_user)
):
    """Create a new reminder"""
    guard_demo_write(current_user)
    reminder_obj = Reminder(
        user_id=current_user.user_id,
        **reminder.model_dump()
    )
    
    doc = reminder_obj.model_dump()
    doc['created_at'] = doc['created_at'].isoformat()
    
    await db.reminders.insert_one(doc)
    
    # Return without _id
    if '_id' in doc:
        del doc['_id']
    return doc

@api_router.put("/reminders/{reminder_id}/toggle")
async def toggle_reminder(
    reminder_id: str,
    current_user: User = Depends(get_current_user)
):
    """Toggle reminder completion"""
    reminder = await db.reminders.find_one(
        {"id": reminder_id, "user_id": current_user.user_id},
        {"_id": 0}
    )
    
    if not reminder:
        raise HTTPException(status_code=404, detail="Reminder not found")
    
    new_completed = not reminder.get('completed', False)
    
    await db.reminders.update_one(
        {"id": reminder_id},
        {"$set": {"completed": new_completed}}
    )
    
    return {"id": reminder_id, "completed": new_completed}

@api_router.delete("/reminders/{reminder_id}")
async def delete_reminder(
    reminder_id: str,
    current_user: User = Depends(get_current_user)
):
    """Delete a reminder"""
    guard_demo_write(current_user)
    result = await db.reminders.delete_one(
        {"id": reminder_id, "user_id": current_user.user_id}
    )
    
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Reminder not found")
    
    return {"message": "Reminder deleted"}

@api_router.post("/reminders/reset")
async def reset_reminders(current_user: User = Depends(get_current_user)):
    """Reset all reminders to incomplete (for new day)"""
    await db.reminders.update_many(
        {"user_id": current_user.user_id},
        {"$set": {"completed": False}}
    )
    return {"message": "All reminders reset"}

# ==================== DESTINATIONS (GPS ROUTES) ====================

@api_router.get("/destinations", response_model=List[dict])
async def get_destinations(current_user: User = Depends(get_current_user)):
    """Get all saved destinations for current user"""
    destinations = await db.destinations.find(
        {"user_id": current_user.user_id},
        {"_id": 0}
    ).sort("created_at", -1).to_list(200)
    return destinations

@api_router.post("/destinations", response_model=dict)
async def create_destination(
    destination: DestinationCreate,
    current_user: User = Depends(get_current_user)
):
    """Create a new destination for route guidance"""
    destination_obj = Destination(
        user_id=current_user.user_id,
        **destination.model_dump()
    )

    doc = destination_obj.model_dump()
    doc["created_at"] = doc["created_at"].isoformat()
    doc["updated_at"] = doc["updated_at"].isoformat()

    await db.destinations.insert_one(doc)

    if "_id" in doc:
        del doc["_id"]
    return doc

@api_router.put("/destinations/{destination_id}", response_model=dict)
async def update_destination(
    destination_id: str,
    destination_update: DestinationUpdate,
    current_user: User = Depends(get_current_user)
):
    """Update destination information"""
    existing = await db.destinations.find_one(
        {"id": destination_id, "user_id": current_user.user_id},
        {"_id": 0}
    )

    if not existing:
        raise HTTPException(status_code=404, detail="Destination not found")

    update_data = {
        k: v for k, v in destination_update.model_dump().items() if v is not None
    }
    update_data["updated_at"] = datetime.now(timezone.utc).isoformat()

    await db.destinations.update_one(
        {"id": destination_id, "user_id": current_user.user_id},
        {"$set": update_data}
    )

    updated = await db.destinations.find_one(
        {"id": destination_id, "user_id": current_user.user_id},
        {"_id": 0}
    )
    return updated

@api_router.delete("/destinations/{destination_id}")
async def delete_destination(
    destination_id: str,
    current_user: User = Depends(get_current_user)
):
    """Delete a destination"""
    result = await db.destinations.delete_one(
        {"id": destination_id, "user_id": current_user.user_id}
    )

    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Destination not found")

    return {"message": "Destination deleted"}

# ==================== NAVIGATION GUIDANCE ====================

@api_router.post("/navigation/guide", response_model=dict)
async def build_navigation_guide(
    request: NavigationGuideRequest,
    current_user: User = Depends(get_current_user)
):
    """Generate simple in-app turn-by-turn style guidance."""
    destination = await db.destinations.find_one(
        {"id": request.destination_id, "user_id": current_user.user_id},
        {"_id": 0}
    )
    if not destination:
        raise HTTPException(status_code=404, detail="Destination not found")

    if destination.get("latitude") is None or destination.get("longitude") is None:
        steps = [
            "Step 1: Confirm your destination name and address.",
            f"Step 2: Head toward {destination['address']}.",
            "Step 3: Use landmarks and street signs to stay on course.",
            f"Step 4: You have arrived at {destination['name']}."
        ]
        return {
            "destination": destination["name"],
            "distance_meters": None,
            "eta_minutes": None,
            "direction": None,
            "steps": steps
        }

    distance_m = haversine_distance_m(
        request.current_latitude,
        request.current_longitude,
        destination["latitude"],
        destination["longitude"]
    )
    bearing = bearing_degrees(
        request.current_latitude,
        request.current_longitude,
        destination["latitude"],
        destination["longitude"]
    )
    direction = cardinal_direction(bearing)
    eta_minutes = max(1, int(round((distance_m / 1000) / 4.5 * 60)))  # 4.5 km/h walk

    steps = [
        f"Step 1: Start from your current location and face {direction}.",
        f"Step 2: Walk about {distance_m/1000:.2f} km in that direction.",
        "Step 3: Check your position every few minutes and keep heading toward the destination.",
        f"Step 4: When you are within 150 meters, look for {destination['name']} at {destination['address']}.",
        f"Step 5: You have arrived at {destination['name']}."
    ]
    if destination.get("notes"):
        steps.insert(4, f"Reminder: {destination['notes']}")

    return {
        "destination": destination["name"],
        "distance_meters": round(distance_m, 1),
        "eta_minutes": eta_minutes,
        "direction": direction,
        "steps": steps
    }

# ==================== SAFETY (GEOFENCING) ====================

@api_router.get("/safety/escalation-rules", response_model=List[dict])
async def get_safety_escalation_rules(
    target_user_id: Optional[str] = None,
    current_user: User = Depends(get_current_user)
):
    owner_id = await resolve_target_user_id(current_user, target_user_id)
    await ensure_default_safety_escalation_rules(owner_id)
    rules = await db.safety_escalation_rules.find(
        {"user_id": owner_id},
        {"_id": 0}
    ).sort([("event_type", 1), ("min_severity", 1)]).to_list(200)
    return rules

@api_router.post("/safety/escalation-rules", response_model=dict)
async def upsert_safety_escalation_rule(
    payload: SafetyEscalationRuleCreate,
    target_user_id: Optional[str] = None,
    current_user: User = Depends(get_current_user)
):
    owner_id = await resolve_target_user_id(current_user, target_user_id, require_write=True if target_user_id else False)
    event_type = normalize_safety_event_type(payload.event_type, "all")
    min_severity = normalize_severity(payload.min_severity, "high")
    intervals = normalize_escalation_intervals(payload.intervals_minutes)
    if not intervals:
        raise HTTPException(status_code=400, detail="intervals_minutes must include at least one value between 1 and 1440")

    existing = await db.safety_escalation_rules.find_one(
        {"user_id": owner_id, "event_type": event_type, "min_severity": min_severity},
        {"_id": 0}
    )
    now_iso = datetime.now(timezone.utc).isoformat()
    if existing:
        await db.safety_escalation_rules.update_one(
            {"id": existing["id"], "user_id": owner_id},
            {"$set": {"intervals_minutes": intervals, "enabled": bool(payload.enabled), "updated_at": now_iso}}
        )
        updated = await db.safety_escalation_rules.find_one({"id": existing["id"], "user_id": owner_id}, {"_id": 0})
        return updated

    rule_obj = SafetyEscalationRule(
        user_id=owner_id,
        event_type=event_type,
        min_severity=min_severity,
        intervals_minutes=intervals,
        enabled=bool(payload.enabled)
    )
    doc = rule_obj.model_dump()
    doc["created_at"] = doc["created_at"].isoformat()
    doc["updated_at"] = doc["updated_at"].isoformat()
    await db.safety_escalation_rules.insert_one(doc)
    doc.pop("_id", None)
    return doc

@api_router.put("/safety/escalation-rules/{rule_id}", response_model=dict)
async def update_safety_escalation_rule(
    rule_id: str,
    payload: SafetyEscalationRuleUpdate,
    target_user_id: Optional[str] = None,
    current_user: User = Depends(get_current_user)
):
    owner_id = await resolve_target_user_id(current_user, target_user_id, require_write=True if target_user_id else False)
    update_data = {}
    if payload.event_type is not None:
        update_data["event_type"] = normalize_safety_event_type(payload.event_type, "all")
    if payload.min_severity is not None:
        update_data["min_severity"] = normalize_severity(payload.min_severity, "high")
    if payload.intervals_minutes is not None:
        intervals = normalize_escalation_intervals(payload.intervals_minutes)
        if not intervals:
            raise HTTPException(status_code=400, detail="intervals_minutes must include at least one value between 1 and 1440")
        update_data["intervals_minutes"] = intervals
    if payload.enabled is not None:
        update_data["enabled"] = bool(payload.enabled)
    if not update_data:
        raise HTTPException(status_code=400, detail="No fields to update")
    update_data["updated_at"] = datetime.now(timezone.utc).isoformat()

    result = await db.safety_escalation_rules.update_one(
        {"id": rule_id, "user_id": owner_id},
        {"$set": update_data}
    )
    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="Escalation rule not found")
    updated = await db.safety_escalation_rules.find_one({"id": rule_id, "user_id": owner_id}, {"_id": 0})
    return updated

@api_router.post("/safety/escalations/run", response_model=dict)
async def run_safety_escalations(
    target_user_id: Optional[str] = None,
    max_alerts: int = 20,
    current_user: User = Depends(get_current_user)
):
    owner_id = await resolve_target_user_id(current_user, target_user_id, require_write=True if target_user_id else False)
    result = await process_due_alert_escalations(
        owner_id,
        max_alerts=max_alerts,
        trigger_source=f"manual_run_by_{current_user.user_id}"
    )
    return result

@api_router.get("/safety/escalations/history", response_model=List[dict])
async def get_safety_escalation_history(
    target_user_id: Optional[str] = None,
    limit: int = 50,
    current_user: User = Depends(get_current_user)
):
    owner_id = await resolve_target_user_id(current_user, target_user_id)
    safe_limit = max(1, min(limit, 200))
    events = await db.safety_escalation_events.find(
        {"user_id": owner_id},
        {"_id": 0}
    ).sort("created_at", -1).to_list(safe_limit)
    return events

@api_router.get("/safety/zones", response_model=List[dict])
async def get_safety_zones(
    target_user_id: Optional[str] = None,
    current_user: User = Depends(get_current_user)
):
    owner_id = await resolve_target_user_id(current_user, target_user_id)
    zones = await db.safety_zones.find(
        {"user_id": owner_id},
        {"_id": 0}
    ).to_list(200)
    return zones

@api_router.post("/safety/zones", response_model=dict)
async def create_safety_zone(
    zone: SafetyZoneCreate,
    target_user_id: Optional[str] = None,
    current_user: User = Depends(get_current_user)
):
    owner_id = await resolve_target_user_id(current_user, target_user_id, require_write=True)
    zone_obj = SafetyZone(user_id=owner_id, **zone.model_dump())
    doc = zone_obj.model_dump()
    doc["created_at"] = doc["created_at"].isoformat()
    doc["updated_at"] = doc["updated_at"].isoformat()
    await db.safety_zones.insert_one(doc)
    if "_id" in doc:
        del doc["_id"]
    return doc

@api_router.put("/safety/zones/{zone_id}", response_model=dict)
async def update_safety_zone(
    zone_id: str,
    zone_update: SafetyZoneUpdate,
    target_user_id: Optional[str] = None,
    current_user: User = Depends(get_current_user)
):
    owner_id = await resolve_target_user_id(current_user, target_user_id, require_write=True)
    update_data = {k: v for k, v in zone_update.model_dump().items() if v is not None}
    update_data["updated_at"] = datetime.now(timezone.utc).isoformat()
    result = await db.safety_zones.update_one(
        {"id": zone_id, "user_id": owner_id},
        {"$set": update_data}
    )
    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="Safety zone not found")
    updated = await db.safety_zones.find_one({"id": zone_id, "user_id": owner_id}, {"_id": 0})
    return updated

@api_router.delete("/safety/zones/{zone_id}")
async def delete_safety_zone(
    zone_id: str,
    target_user_id: Optional[str] = None,
    current_user: User = Depends(get_current_user)
):
    owner_id = await resolve_target_user_id(current_user, target_user_id, require_write=True)
    result = await db.safety_zones.delete_one({"id": zone_id, "user_id": owner_id})
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Safety zone not found")
    return {"message": "Safety zone deleted"}

@api_router.post("/safety/location-ping", response_model=dict)
async def safety_location_ping(
    ping: SafetyLocationPing,
    current_user: User = Depends(get_current_user)
):
    """Evaluate geofence state and create alerts if outside all active safe zones."""
    zones = await db.safety_zones.find(
        {"user_id": current_user.user_id, "active": True},
        {"_id": 0}
    ).to_list(200)

    evaluated = []
    outside_count = 0
    now = datetime.now(timezone.utc)
    new_alerts = []
    hook_results = []

    for zone in zones:
        distance = haversine_distance_m(
            ping.latitude,
            ping.longitude,
            zone["center_latitude"],
            zone["center_longitude"]
        )
        inside = distance <= zone["radius_meters"]
        if not inside:
            outside_count += 1

        evaluated.append({
            "zone_id": zone["id"],
            "zone_name": zone["name"],
            "inside": inside,
            "distance_meters": round(distance, 1),
            "radius_meters": zone["radius_meters"]
        })

        if not inside:
            existing_recent = await db.safety_alerts.find_one(
                {
                    "user_id": current_user.user_id,
                    "zone_id": zone["id"],
                    "acknowledged": False,
                    "triggered_at": {"$gte": (now - timedelta(minutes=20)).isoformat()}
                },
                {"_id": 0}
            )
            if not existing_recent:
                alert_doc, results = await create_safety_alert_with_escalation(
                    user_id=current_user.user_id,
                    event_type="geofence_exit",
                    severity="high",
                    message=f"Outside safe zone: {zone['name']}",
                    payload={
                        "zone_id": zone["id"],
                        "zone_name": zone["name"],
                        "distance_meters": round(distance, 1),
                        "location": {"latitude": ping.latitude, "longitude": ping.longitude}
                    },
                    extra_fields={
                        "zone_id": zone["id"],
                        "zone_name": zone["name"],
                        "latitude": ping.latitude,
                        "longitude": ping.longitude,
                        "distance_meters": round(distance, 1)
                    }
                )
                new_alerts.append(alert_doc)
                hook_results.append({"alert_id": alert_doc["id"], "results": results})

    return {
        "checked_at": now.isoformat(),
        "evaluated_zones": evaluated,
        "outside_count": outside_count,
        "new_alerts": new_alerts,
        "hook_results": hook_results
    }

@api_router.get("/safety/alerts", response_model=List[dict])
async def get_safety_alerts(
    target_user_id: Optional[str] = None,
    only_unacknowledged: bool = False,
    current_user: User = Depends(get_current_user)
):
    owner_id = await resolve_target_user_id(current_user, target_user_id)
    query = {"user_id": owner_id}
    if only_unacknowledged:
        query["acknowledged"] = False
    alerts = await db.safety_alerts.find(query, {"_id": 0}).sort("triggered_at", -1).to_list(200)
    return alerts

@api_router.patch("/safety/alerts/{alert_id}/ack")
async def acknowledge_safety_alert(
    alert_id: str,
    target_user_id: Optional[str] = None,
    current_user: User = Depends(get_current_user)
):
    owner_id = await resolve_target_user_id(current_user, target_user_id)
    now_iso = datetime.now(timezone.utc).isoformat()
    result = await db.safety_alerts.update_one(
        {"id": alert_id, "user_id": owner_id},
        {
            "$set": {
                "acknowledged": True,
                "acknowledged_at": now_iso,
                "acknowledged_by_user_id": current_user.user_id,
                "escalation_status": "acknowledged",
                "escalation_next_at": None
            }
        }
    )
    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="Safety alert not found")
    await db.safety_escalation_events.insert_one({
        "id": f"escevt_{uuid.uuid4().hex[:12]}",
        "user_id": owner_id,
        "alert_id": alert_id,
        "event_type": "acknowledge",
        "stage": None,
        "action": "acknowledged",
        "severity": None,
        "hook_results": [],
        "created_at": now_iso,
        "triggered_by_user_id": current_user.user_id
    })
    return {"message": "Alert acknowledged"}

@api_router.get("/safety/emergency-contacts", response_model=List[dict])
async def get_emergency_contacts(
    target_user_id: Optional[str] = None,
    current_user: User = Depends(get_current_user)
):
    owner_id = await resolve_target_user_id(current_user, target_user_id)
    contacts = await db.emergency_contacts.find(
        {"user_id": owner_id},
        {"_id": 0}
    ).sort([("is_primary", -1), ("created_at", 1)]).to_list(100)
    return contacts

@api_router.post("/safety/emergency-contacts", response_model=dict)
async def create_emergency_contact(
    contact: EmergencyContactCreate,
    target_user_id: Optional[str] = None,
    current_user: User = Depends(get_current_user)
):
    owner_id = await resolve_target_user_id(
        current_user,
        target_user_id,
        require_write=True if target_user_id else False
    )
    if contact.is_primary:
        await db.emergency_contacts.update_many(
            {"user_id": owner_id},
            {"$set": {"is_primary": False, "updated_at": datetime.now(timezone.utc).isoformat()}}
        )

    contact_obj = EmergencyContact(user_id=owner_id, **contact.model_dump())
    doc = contact_obj.model_dump()
    doc["created_at"] = doc["created_at"].isoformat()
    doc["updated_at"] = doc["updated_at"].isoformat()
    await db.emergency_contacts.insert_one(doc)
    doc.pop("_id", None)
    return doc

@api_router.put("/safety/emergency-contacts/{contact_id}", response_model=dict)
async def update_emergency_contact(
    contact_id: str,
    contact: EmergencyContactUpdate,
    target_user_id: Optional[str] = None,
    current_user: User = Depends(get_current_user)
):
    owner_id = await resolve_target_user_id(
        current_user,
        target_user_id,
        require_write=True if target_user_id else False
    )
    update_data = {k: v for k, v in contact.model_dump().items() if v is not None}
    if update_data.get("is_primary") is True:
        await db.emergency_contacts.update_many(
            {"user_id": owner_id},
            {"$set": {"is_primary": False, "updated_at": datetime.now(timezone.utc).isoformat()}}
        )
    update_data["updated_at"] = datetime.now(timezone.utc).isoformat()
    result = await db.emergency_contacts.update_one(
        {"id": contact_id, "user_id": owner_id},
        {"$set": update_data}
    )
    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="Emergency contact not found")
    updated = await db.emergency_contacts.find_one({"id": contact_id, "user_id": owner_id}, {"_id": 0})
    return updated

@api_router.delete("/safety/emergency-contacts/{contact_id}")
async def delete_emergency_contact(
    contact_id: str,
    target_user_id: Optional[str] = None,
    current_user: User = Depends(get_current_user)
):
    owner_id = await resolve_target_user_id(
        current_user,
        target_user_id,
        require_write=True if target_user_id else False
    )
    result = await db.emergency_contacts.delete_one({"id": contact_id, "user_id": owner_id})
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Emergency contact not found")
    return {"message": "Emergency contact deleted"}

@api_router.post("/safety/sos", response_model=dict)
async def trigger_sos(
    payload: SOSTriggerRequest,
    target_user_id: Optional[str] = None,
    current_user: User = Depends(get_current_user)
):
    owner_id = await resolve_target_user_id(
        current_user,
        target_user_id,
        require_write=True if target_user_id else False
    )
    now = datetime.now(timezone.utc)

    contacts = await db.emergency_contacts.find(
        {"user_id": owner_id},
        {"_id": 0}
    ).sort([("is_primary", -1), ("created_at", 1)]).to_list(20)
    primary = contacts[0] if contacts else None

    alert_message = payload.message.strip() if payload.message else "Emergency SOS was triggered."
    alert_doc, hook_results = await create_safety_alert_with_escalation(
        user_id=owner_id,
        event_type="sos_trigger",
        severity="critical",
        message=alert_message,
        payload={
            "message": alert_message,
            "location": {"latitude": payload.latitude, "longitude": payload.longitude},
            "triggered_by_user_id": current_user.user_id
        },
        extra_fields={
            "zone_id": None,
            "zone_name": "Emergency",
            "latitude": payload.latitude,
            "longitude": payload.longitude
        }
    )

    dial_number = clean_phone_for_dial(primary.get("phone")) if primary else None
    return {
        "message": "SOS alert sent.",
        "alert": alert_doc,
        "primary_contact": primary,
        "emergency_contacts": contacts,
        "dial_number": dial_number,
        "dial_uri": f"tel:{dial_number}" if payload.auto_call_primary and dial_number else None,
        "hook_results": hook_results
    }

@api_router.post("/safety/share-location", response_model=dict)
async def share_location_now(
    payload: LocationShareRequest,
    target_user_id: Optional[str] = None,
    current_user: User = Depends(get_current_user)
):
    owner_id = await resolve_target_user_id(
        current_user,
        target_user_id,
        require_write=True if target_user_id else False
    )
    now = datetime.now(timezone.utc)
    share_doc = {
        "id": f"locshare_{uuid.uuid4().hex[:12]}",
        "user_id": owner_id,
        "reason": (payload.reason or "manual_share").strip()[:100],
        "latitude": payload.latitude,
        "longitude": payload.longitude,
        "shared_by_user_id": current_user.user_id,
        "shared_at": now.isoformat()
    }
    await db.safety_location_shares.insert_one(share_doc)
    hook_results = await dispatch_proactive_hooks(
        patient_user_id=owner_id,
        event_type="location_share",
        severity="medium",
        payload={
            "location_share_id": share_doc["id"],
            "reason": share_doc["reason"],
            "location": {"latitude": payload.latitude, "longitude": payload.longitude}
        }
    )
    return {"message": "Location shared with caregivers.", "location_share": share_doc, "hook_results": hook_results}

@api_router.get("/safety/fall-events", response_model=List[dict])
async def get_fall_events(
    target_user_id: Optional[str] = None,
    limit: int = 20,
    current_user: User = Depends(get_current_user)
):
    owner_id = await resolve_target_user_id(current_user, target_user_id)
    safe_limit = max(1, min(limit, 100))
    events = await db.safety_fall_events.find(
        {"user_id": owner_id},
        {"_id": 0}
    ).sort("detected_at", -1).to_list(safe_limit)
    return events

@api_router.post("/safety/fall-events", response_model=dict)
async def report_fall_event(
    payload: FallEventCreate,
    target_user_id: Optional[str] = None,
    current_user: User = Depends(get_current_user)
):
    owner_id = await resolve_target_user_id(
        current_user,
        target_user_id,
        require_write=True if target_user_id else False
    )
    now = datetime.now(timezone.utc)
    detected_by = (payload.detected_by or "manual").strip().lower()
    severity = (payload.severity or "high").strip().lower()
    if detected_by not in {"manual", "device_motion", "wearable"}:
        detected_by = "manual"
    if severity not in {"medium", "high", "critical"}:
        severity = "high"

    fall_doc = {
        "id": f"fall_{uuid.uuid4().hex[:12]}",
        "user_id": owner_id,
        "detected_by": detected_by,
        "severity": severity,
        "confidence": payload.confidence,
        "notes": payload.notes,
        "latitude": payload.latitude,
        "longitude": payload.longitude,
        "reported_by_user_id": current_user.user_id,
        "detected_at": now.isoformat()
    }
    await db.safety_fall_events.insert_one(fall_doc)

    alert_doc, hook_results = await create_safety_alert_with_escalation(
        user_id=owner_id,
        event_type="fall_detected",
        severity=severity,
        message="Possible fall detected. Check on patient immediately.",
        payload={
            "fall_event_id": fall_doc["id"],
            "detected_by": detected_by,
            "confidence": payload.confidence,
            "notes": payload.notes,
            "location": {"latitude": payload.latitude, "longitude": payload.longitude}
        },
        extra_fields={
            "zone_id": None,
            "zone_name": "Emergency",
            "latitude": payload.latitude,
            "longitude": payload.longitude
        }
    )
    return {
        "message": "Fall event reported and caregivers notified.",
        "fall_event": fall_doc,
        "alert": alert_doc,
        "hook_results": hook_results
    }

# ==================== MEDICATION MANAGEMENT ====================

async def _expected_doses_for_period(
    medications: List[dict],
    start_at: datetime,
    end_at: datetime
) -> List[Tuple[str, datetime]]:
    expected = []
    day_cursor = start_at.date()
    end_day = end_at.date()
    while day_cursor <= end_day:
        day_start = datetime(day_cursor.year, day_cursor.month, day_cursor.day, tzinfo=timezone.utc)
        for med in medications:
            if not medication_due_on_day(med, day_cursor):
                continue
            times = medication_schedule_times(med)
            for time_str in times:
                t = parse_hhmm_to_datetime(time_str, day_start)
                if t and start_at <= t <= end_at:
                    expected.append((med["id"], t))
        day_cursor = day_cursor + timedelta(days=1)
    return expected

async def _build_medication_mar_for_period(
    owner_id: str,
    start_at: datetime,
    end_at: datetime
) -> dict:
    meds = await db.medications.find({"user_id": owner_id, "active": True}, {"_id": 0}).to_list(300)
    meds_map = {m["id"]: m for m in meds}
    expected_doses = await _expected_doses_for_period(meds, start_at, end_at)
    expected_doses.sort(key=lambda item: item[1])

    search_start = (start_at - timedelta(hours=8)).isoformat()
    search_end = (end_at + timedelta(hours=8)).isoformat()
    logs = await db.medication_intake_logs.find(
        {
            "user_id": owner_id,
            "confirmed_at": {"$gte": search_start, "$lte": search_end}
        },
        {"_id": 0}
    ).to_list(8000)

    logs_by_slot = {}
    for log in logs:
        med_id = log.get("medication_id")
        scheduled_for = log.get("scheduled_for")
        if not med_id or not scheduled_for:
            continue
        key = f"{med_id}|{scheduled_for}"
        logs_by_slot.setdefault(key, []).append(log)

    unslotted_taken = {}
    for log in logs:
        if log.get("status") != "taken" or log.get("scheduled_for"):
            continue
        med_id = log.get("medication_id")
        if not med_id:
            continue
        unslotted_taken.setdefault(med_id, []).append(log)
    for med_id in unslotted_taken:
        unslotted_taken[med_id].sort(key=lambda l: l.get("confirmed_at", ""))

    used_unslotted_ids = set()
    now = datetime.now(timezone.utc)
    on_time_window = timedelta(hours=2)
    late_cutoff = timedelta(hours=8)
    slots = []
    summary = {
        "expected": 0,
        "taken_on_time": 0,
        "taken_late": 0,
        "taken_total": 0,
        "due": 0,
        "upcoming": 0,
        "overdue": 0,
        "missed": 0
    }

    for med_id, scheduled_for in expected_doses:
        med = meds_map.get(med_id)
        if not med:
            continue
        summary["expected"] += 1
        scheduled_iso = scheduled_for.isoformat()
        key = f"{med_id}|{scheduled_iso}"
        slot_logs = sorted(logs_by_slot.get(key, []), key=lambda l: l.get("confirmed_at", ""), reverse=True)
        chosen = slot_logs[0] if slot_logs else None
        status = "upcoming"
        confirmed_at = None
        delay_minutes = None
        source = None
        notes = None

        if chosen:
            status = chosen.get("status", "taken")
            confirmed_at = chosen.get("confirmed_at")
            source = chosen.get("source")
            notes = chosen.get("notes")
            if status == "taken":
                confirmed_dt = parse_iso_to_utc(confirmed_at)
                if confirmed_dt:
                    delay_minutes = int(round((confirmed_dt - scheduled_for).total_seconds() / 60))
                    status = "taken_on_time" if confirmed_dt <= scheduled_for + on_time_window else "taken_late"
            elif status not in {"missed"}:
                status = "missed"
        else:
            candidate_log = None
            for log in unslotted_taken.get(med_id, []):
                if log["id"] in used_unslotted_ids:
                    continue
                confirmed_dt = parse_iso_to_utc(log.get("confirmed_at"))
                if not confirmed_dt:
                    continue
                if scheduled_for <= confirmed_dt <= scheduled_for + late_cutoff:
                    candidate_log = log
                    used_unslotted_ids.add(log["id"])
                    break

            if candidate_log:
                confirmed_at = candidate_log.get("confirmed_at")
                source = candidate_log.get("source")
                notes = candidate_log.get("notes")
                confirmed_dt = parse_iso_to_utc(confirmed_at)
                delay_minutes = int(round((confirmed_dt - scheduled_for).total_seconds() / 60)) if confirmed_dt else None
                status = "taken_on_time" if confirmed_dt and confirmed_dt <= scheduled_for + on_time_window else "taken_late"
            else:
                if now < scheduled_for:
                    status = "upcoming"
                elif now <= scheduled_for + on_time_window:
                    status = "due"
                elif now <= scheduled_for + late_cutoff:
                    status = "overdue"
                else:
                    status = "missed"

        if status == "taken_on_time":
            summary["taken_on_time"] += 1
            summary["taken_total"] += 1
        elif status == "taken_late":
            summary["taken_late"] += 1
            summary["taken_total"] += 1
        elif status == "due":
            summary["due"] += 1
        elif status == "upcoming":
            summary["upcoming"] += 1
        elif status == "overdue":
            summary["overdue"] += 1
        elif status == "missed":
            summary["missed"] += 1

        slots.append({
            "medication_id": med_id,
            "name": med.get("name"),
            "dosage": med.get("dosage"),
            "scheduled_for": scheduled_iso,
            "status": status,
            "confirmed_at": confirmed_at,
            "delay_minutes": delay_minutes,
            "source": source,
            "notes": notes
        })

    return {
        "medications": meds,
        "slots": slots,
        "summary": summary
    }

async def build_today_medication_voice_response(owner_id: str) -> str:
    now = datetime.now(timezone.utc)
    day_start = datetime(now.year, now.month, now.day, tzinfo=timezone.utc)
    day_end = datetime(now.year, now.month, now.day, 23, 59, 59, tzinfo=timezone.utc)
    mar = await _build_medication_mar_for_period(owner_id, day_start, day_end)
    meds = mar.get("medications", [])
    slots = sorted(
        mar.get("slots", []),
        key=lambda s: s.get("scheduled_for", "")
    )

    if not meds:
        return "I do not see active medications listed right now. Please ask your caregiver or doctor to add your regimen."
    if not slots:
        return "You have active medications, but none are scheduled for today."

    meds_by_id = {m.get("id"): m for m in meds}
    grouped = {}
    for slot in slots:
        med_id = slot.get("medication_id")
        if not med_id:
            continue
        med = meds_by_id.get(med_id, {})
        name = (slot.get("name") or med.get("name") or "Medication").strip()
        dosage = (slot.get("dosage") or med.get("dosage") or "").strip()
        label = f"{name} {dosage}".strip()
        scheduled_dt = parse_iso_to_utc(slot.get("scheduled_for"))
        if not scheduled_dt:
            continue
        grouped.setdefault(med_id, {"label": label, "items": []})
        grouped[med_id]["items"].append({
            "scheduled_dt": scheduled_dt,
            "status": slot.get("status", "upcoming"),
            "name": name,
            "dosage": dosage
        })

    if not grouped:
        return "I can see your medications, but I could not read today's schedule times right now."

    summaries = []
    for med_data in grouped.values():
        med_data["items"].sort(key=lambda item: item["scheduled_dt"])
        time_labels = [format_datetime_for_voice(item["scheduled_dt"]) for item in med_data["items"]]
        summaries.append(f"{med_data['label']} at {join_voice_items(time_labels)}")

    summaries = summaries[:3]
    intro = f"For today, you have {len(slots)} scheduled medication dose{'s' if len(slots) != 1 else ''}."
    detail = f"You should take {join_voice_items(summaries)}."

    pending_slots = []
    for med_data in grouped.values():
        for item in med_data["items"]:
            if item["status"] in {"due", "upcoming", "overdue"}:
                pending_slots.append(item)
    pending_slots.sort(key=lambda item: item["scheduled_dt"])

    if pending_slots:
        nxt = pending_slots[0]
        next_line = (
            f"The next one is {nxt['name']} {nxt['dosage']} at {format_datetime_for_voice(nxt['scheduled_dt'])}."
            .replace("  ", " ")
            .strip()
        )
    else:
        next_line = "All scheduled doses for today are already logged."
    return f"{intro} {detail} {next_line}"


async def handle_voice_medication_intake(owner_id: str, spoken_text: str) -> dict:
    """
    Handle a patient reporting they took their medication via voice.
    Returns dict with action and response for the voice assistant.
    """
    now_utc = datetime.now(timezone.utc)
    day_start = now_utc.replace(hour=0, minute=0, second=0, microsecond=0)
    day_end = day_start + timedelta(days=1)

    mar = await _build_medication_mar_for_period(owner_id, day_start, day_end)
    slots = mar.get("slots", [])
    medications = mar.get("medications", [])

    # Find pending doses (due, overdue, or upcoming within 2 hours)
    pending = []
    for slot in slots:
        if slot.get("status") in {"due", "overdue", "upcoming"}:
            sched_str = slot.get("scheduled_for")
            if sched_str and slot.get("status") == "upcoming":
                try:
                    sched_dt = datetime.fromisoformat(sched_str.replace("Z", "+00:00"))
                    if sched_dt.tzinfo is None:
                        sched_dt = sched_dt.replace(tzinfo=timezone.utc)
                    if (sched_dt - now_utc).total_seconds() > 7200:  # >2h in future
                        continue
                except Exception:
                    continue
            pending.append(slot)

    if not pending:
        return {
            "action": "speak",
            "response": "It looks like all your scheduled doses for today are already taken. Great job staying on track!"
        }

    # Try to match a specific medication name from the spoken text
    lowered = (spoken_text or "").lower()
    matched_slot = None

    # Build a name lookup from medications
    med_names = {}
    for med in medications:
        med_names[med.get("id", "")] = (med.get("name", ""), med.get("dosage", ""))

    for slot in pending:
        med_id = slot.get("medication_id", "")
        name, dosage = med_names.get(med_id, ("", ""))
        name_lower = name.lower()
        # Check if medication name appears in spoken text
        if name_lower and name_lower in lowered:
            matched_slot = slot
            break
        # Also check individual words of the name (e.g., "metformin" from "Metformin 500mg")
        for word in name_lower.split():
            if len(word) > 3 and word in lowered:
                matched_slot = slot
                break
        if matched_slot:
            break

    # If exactly 1 pending dose or we matched one, log it
    target_slot = matched_slot or (pending[0] if len(pending) == 1 else None)

    if not target_slot and len(pending) > 1:
        # Ask patient which medication
        names = []
        for slot in pending[:5]:
            med_id = slot.get("medication_id", "")
            name, dosage = med_names.get(med_id, ("medication", ""))
            label = f"{name} {dosage}".strip() if dosage else name
            names.append(label)
        listing = join_voice_items(names)
        return {
            "action": "speak",
            "response": f"I see you have {len(pending)} doses pending: {listing}. Which one did you take?"
        }

    # Log the intake
    med_id = target_slot["medication_id"]
    name, dosage = med_names.get(med_id, ("your medication", ""))
    label = f"{name} {dosage}".strip() if dosage else name

    scheduled_for = None
    if target_slot.get("scheduled_for"):
        try:
            scheduled_for = datetime.fromisoformat(
                target_slot["scheduled_for"].replace("Z", "+00:00")
            )
            if scheduled_for.tzinfo is None:
                scheduled_for = scheduled_for.replace(tzinfo=timezone.utc)
            scheduled_for = scheduled_for.astimezone(timezone.utc)
        except Exception:
            scheduled_for = None

    delay_minutes = None
    if scheduled_for:
        delay_minutes = int(round((now_utc - scheduled_for).total_seconds() / 60))

    log_obj = MedicationIntakeLog(
        user_id=owner_id,
        medication_id=med_id,
        status="taken",
        scheduled_for=scheduled_for,
        source="voice",
        notes="Logged via voice assistant",
        recorded_by_user_id=owner_id,
        delay_minutes=delay_minutes,
        confirmed_at=now_utc
    )
    doc = log_obj.model_dump()
    doc["confirmed_at"] = doc["confirmed_at"].isoformat()
    if doc.get("scheduled_for"):
        doc["scheduled_for"] = doc["scheduled_for"].isoformat()
    await db.medication_intake_logs.insert_one(doc)

    # Resolve missed-dose alerts
    if doc.get("scheduled_for"):
        await db.medication_missed_alerts.delete_many({
            "user_id": owner_id,
            "medication_id": med_id,
            "scheduled_for": doc["scheduled_for"]
        })
        await db.safety_alerts.update_many(
            {
                "user_id": owner_id,
                "event_type": "missed_medication_dose",
                "acknowledged": False,
                "medication_id": med_id,
                "scheduled_for": doc["scheduled_for"]
            },
            {
                "$set": {
                    "acknowledged": True,
                    "acknowledged_at": now_utc.isoformat(),
                    "acknowledged_by_user_id": owner_id,
                    "escalation_status": "resolved_after_intake",
                    "escalation_next_at": None
                }
            }
        )

    remaining = len(pending) - 1
    if remaining > 0:
        return {
            "action": "speak",
            "response": f"Got it! I've recorded that you took {label}. You still have {remaining} more dose{'s' if remaining > 1 else ''} scheduled for today."
        }
    return {
        "action": "speak",
        "response": f"Got it! I've recorded that you took {label}. That's all your medication for now. Well done!"
    }


async def handle_voice_mood_report(owner_id: str, spoken_text: str) -> dict:
    """
    Handle a patient reporting their mood via voice.
    Uses LLM to extract mood score, then creates a MoodCheckin.
    """
    # Use LLM to extract mood information
    mood_score = 3
    energy_score = 3
    notes = spoken_text

    if os.environ.get("OPENAI_API_KEY"):
        prompt = (
            "Extract mood information from the patient's statement.\n"
            "Return strict JSON: {\"mood_score\": <1-5>, \"energy_score\": <1-5>, \"summary\": \"<brief note>\"}\n"
            "Scale: 1=very bad/low, 2=bad/low, 3=neutral/okay, 4=good/high, 5=very good/high.\n"
            "If energy is not mentioned, default to 3.\n"
        )
        try:
            client = get_openai_client()
            completion = await client.chat.completions.create(
                model="gpt-4o-mini",
                temperature=0,
                max_tokens=60,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": spoken_text}
                ]
            )
            raw = (completion.choices[0].message.content or "").strip()
            parsed = json.loads(raw) if raw else {}
            mood_score = max(1, min(5, int(parsed.get("mood_score", 3))))
            energy_score = max(1, min(5, int(parsed.get("energy_score", 3))))
            notes = parsed.get("summary", spoken_text)
        except Exception as e:
            logger.warning(f"Voice mood extraction failed: {e}")

    now_utc = datetime.now(timezone.utc)
    checkin = MoodCheckin(
        user_id=owner_id,
        mood_score=mood_score,
        energy_score=energy_score,
        notes=notes,
        source="voice",
        created_by_user_id=owner_id,
        created_at=now_utc
    )
    doc = checkin.model_dump()
    doc["created_at"] = doc["created_at"].isoformat()
    await db.mood_checkins.insert_one(doc)

    mood_labels = {1: "very low", 2: "a bit down", 3: "okay", 4: "good", 5: "great"}
    mood_word = mood_labels.get(mood_score, "noted")

    return {
        "action": "speak",
        "response": f"Thank you for sharing how you feel. I've noted that you're feeling {mood_word}. Your caregiver will be able to see this update."
    }


def instruction_is_due_today(instruction: dict, target_date: date) -> bool:
    frequency = normalize_frequency(instruction.get("frequency"), fallback="daily")
    if frequency == "daily":
        return True
    if frequency == "weekly":
        target_day = target_date.strftime("%A").lower()
        configured_day = normalize_day_of_week(instruction.get("day_of_week"))
        # Weekly items without explicit weekday remain visible.
        return configured_day is None or configured_day == target_day
    return False

def normalize_instruction_item(instruction: dict) -> dict:
    text = (instruction.get("instruction_text") or "").strip()
    summary = (instruction.get("summary") or "").strip()
    if not summary and text:
        summary = text[:220] + ("..." if len(text) > 220 else "")
    return {
        "id": instruction.get("id"),
        "title": instruction.get("title"),
        "summary": summary,
        "instruction_text": text,
        "frequency": instruction.get("frequency"),
        "day_of_week": instruction.get("day_of_week"),
        "time_of_day": instruction.get("time_of_day"),
        "policy_type": instruction.get("policy_type"),
        "regimen_key": instruction.get("regimen_key"),
        "version": int(instruction.get("version", 1)),
        "signed_off_by_name": instruction.get("signed_off_by_name"),
        "source_file_url": instruction.get("source_file_url"),
        "updated_at": instruction.get("updated_at")
    }

def build_today_plan_voice_script(
    target_day: date,
    mar: dict,
    due_instructions: List[dict],
    as_needed_instructions: List[dict]
) -> str:
    readable_date = f"{target_day.strftime('%A, %B')} {target_day.day}, {target_day.year}"
    slots = sorted(mar.get("slots", []), key=lambda s: s.get("scheduled_for", ""))
    active_slots = [s for s in slots if s.get("status") in {"due", "upcoming", "overdue", "taken_on_time", "taken_late"}]
    medication_lines = []
    for slot in active_slots[:6]:
        slot_dt = parse_iso_to_utc(slot.get("scheduled_for"))
        if not slot_dt:
            continue
        medication_lines.append(
            f"{slot.get('name', 'Medication')} {slot.get('dosage', '')} at {format_datetime_for_voice(slot_dt)}".replace("  ", " ").strip()
        )

    instruction_lines = []
    for inst in due_instructions[:4]:
        inst_text = (inst.get("instruction_text") or inst.get("summary") or "").strip()
        if not inst_text:
            continue
        trimmed = inst_text[:260] + ("..." if len(inst_text) > 260 else "")
        instruction_lines.append(f"{inst.get('title', 'Instruction')}: {trimmed}")

    parts = [f"Today is {readable_date}."]
    if medication_lines:
        parts.append(
            f"You have {len(active_slots)} medication dose{'s' if len(active_slots) != 1 else ''} on today's plan: {join_voice_items(medication_lines)}."
        )
    else:
        parts.append("I do not see scheduled medication doses for today.")

    if instruction_lines:
        parts.append(f"Today's care instructions are: {join_voice_items(instruction_lines)}.")
    else:
        parts.append("There are no scheduled care procedures for today.")

    if as_needed_instructions:
        names = [inst.get("title", "As needed instruction") for inst in as_needed_instructions[:3]]
        parts.append(f"If needed, you can also use: {join_voice_items(names)}.")
    return " ".join(parts)

async def build_patient_today_plan(owner_id: str, target_day: Optional[date] = None) -> dict:
    ref_day = target_day or datetime.now(timezone.utc).date()
    day_start = datetime(ref_day.year, ref_day.month, ref_day.day, tzinfo=timezone.utc)
    day_end = datetime(ref_day.year, ref_day.month, ref_day.day, 23, 59, 59, tzinfo=timezone.utc)
    mar = await _build_medication_mar_for_period(owner_id, day_start, day_end)

    instructions_raw = await db.care_instructions.find(
        {"user_id": owner_id, "active": True},
        {"_id": 0, "search_text": 0}
    ).sort("updated_at", -1).to_list(500)
    effective = [i for i in instructions_raw if instruction_allowed_for_patient_use(i, ref_day)]

    due_today = []
    as_needed = []
    for inst in effective:
        frequency = normalize_frequency(inst.get("frequency"), fallback="daily")
        normalized = normalize_instruction_item(inst)
        if frequency == "as_needed":
            as_needed.append(normalized)
        elif instruction_is_due_today(inst, ref_day):
            due_today.append(normalized)

    due_today.sort(key=lambda item: (item.get("time_of_day") or "zzzz", item.get("title") or ""))
    as_needed.sort(key=lambda item: (item.get("title") or ""))

    active_medication_regimens = select_latest_medication_regimens([i for i in effective if i.get("policy_type") == "medication"])
    regimen_items = [normalize_instruction_item(i) for i in active_medication_regimens]

    voice_script = build_today_plan_voice_script(ref_day, mar, due_today, as_needed)
    return {
        "date": ref_day.isoformat(),
        "medication_plan": {
            "summary": mar.get("summary", {}),
            "slots": mar.get("slots", [])
        },
        "instructions": {
            "due_today": due_today,
            "as_needed": as_needed,
            "active_medication_regimens": regimen_items
        },
        "voice_script": voice_script
    }

# ==================== EXTERNAL DOCTOR BOT ====================

def extract_patient_id_from_text(text: str) -> Optional[str]:
    match = re.search(r"\buser_[a-z0-9]{12}\b", (text or "").lower())
    return match.group(0) if match else None

async def get_doctor_accessible_patients(current_user: User, limit: int = 100) -> List[dict]:
    safe_limit = max(1, min(limit, 200))
    if current_user.role == "admin":
        docs = await db.users.find(
            {"role": "patient", "account_status": "active"},
            {"_id": 0, "user_id": 1, "name": 1, "email": 1}
        ).sort("created_at", -1).to_list(safe_limit)
        return docs
    if current_user.role == "patient":
        return [{"user_id": current_user.user_id, "name": current_user.name, "email": current_user.email}]

    links = await db.care_links.find(
        {"caregiver_id": current_user.user_id, "status": "accepted"},
        {"_id": 0, "patient_id": 1}
    ).to_list(500)
    patient_ids = [l.get("patient_id") for l in links if l.get("patient_id")]
    if not patient_ids:
        return []
    docs = await db.users.find(
        {"user_id": {"$in": patient_ids}},
        {"_id": 0, "user_id": 1, "name": 1, "email": 1}
    ).to_list(min(safe_limit, len(patient_ids)))
    docs.sort(key=lambda d: (d.get("name") or "").lower())
    return docs

async def classify_doctor_bot_intent_model(text: str) -> str:
    if not os.environ.get("OPENAI_API_KEY"):
        return "unknown"
    prompt = (
        "Classify a doctor's message about patient monitoring or management.\n"
        "Allowed intents:\n"
        "- progress_summary: asking for overall status/progress\n"
        "- medications_today: asking about today's medication schedule\n"
        "- missed_doses: asking about missed or skipped doses\n"
        "- safety_alerts: asking about safety alerts, falls, geofence\n"
        "- mood_behavior: asking about mood, behavior, BPSD\n"
        "- today_instructions: asking about today's care instructions\n"
        "- compliance_check: asking about compliance/fulfillment\n"
        "- full_report: requesting a complete report\n"
        "- add_medication: doctor wants to ADD/CREATE/PRESCRIBE a new medication\n"
        "- add_care_instruction: doctor wants to ADD/CREATE a care instruction or protocol\n"
        "- deactivate_medication: doctor wants to STOP/DEACTIVATE/DISCONTINUE a medication\n"
        "- log_patient_intake: doctor wants to MARK/LOG a dose as taken for the patient\n"
        "- unknown: does not match any above\n"
        "Return strict JSON only: {\"intent\":\"...\"}."
    )
    try:
        client = get_openai_client()
        completion = await client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0,
            max_tokens=40,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": text}
            ]
        )
        raw = (completion.choices[0].message.content or "").strip()
        parsed = json.loads(raw) if raw else {}
        intent = str(parsed.get("intent", "unknown")).strip().lower()
        if intent in DOCTOR_BOT_INTENTS:
            return intent
    except Exception as exc:
        logger.warning(f"Doctor bot intent classification failed: {exc}")
    return "unknown"

def _format_slot_line(slot: dict) -> str:
    dt = parse_iso_to_utc(slot.get("scheduled_for"))
    when = format_datetime_for_voice(dt) if dt else "unknown time"
    status = str(slot.get("status") or "upcoming").replace("_", " ")
    return f"- {slot.get('name', 'Medication')} {slot.get('dosage', '')} at {when} ({status})".replace("  ", " ").strip()

def _format_safety_alert_line(alert: dict) -> str:
    dt = parse_iso_to_utc(alert.get("triggered_at"))
    when = dt.isoformat() if dt else (alert.get("triggered_at") or "unknown time")
    sev = normalize_severity(alert.get("severity"), "high")
    event_type = alert.get("event_type", "alert")
    message = alert.get("message", "")
    return f"- [{sev}] {event_type} at {when}: {message}".strip()

async def build_doctor_patient_snapshot(patient_user_id: str, current_user: User) -> dict:
    owner_id = await resolve_target_user_id(current_user, patient_user_id)
    patient = await db.users.find_one(
        {"user_id": owner_id},
        {"_id": 0, "hashed_password": 0}
    )
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")

    reminders = await db.reminders.find({"user_id": owner_id}, {"_id": 0}).to_list(600)
    completed_reminders = len([r for r in reminders if r.get("completed")])
    reminder_completion = round((completed_reminders / len(reminders)) * 100, 1) if reminders else 0.0

    adherence = await medication_adherence_summary(7, owner_id, current_user)
    missed = await medication_missed_doses(2, owner_id, current_user)
    open_alerts = await db.safety_alerts.find(
        {"user_id": owner_id, "acknowledged": False},
        {"_id": 0}
    ).sort("triggered_at", -1).to_list(50)
    today_plan = await build_patient_today_plan(owner_id)
    bpsd_analytics = await compute_bpsd_analytics(owner_id, 30)

    return {
        "patient": {
            "user_id": patient.get("user_id"),
            "name": patient.get("name"),
            "email": patient.get("email")
        },
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "reminders": {
            "total": len(reminders),
            "completed": completed_reminders,
            "completion_percent": reminder_completion
        },
        "adherence_7d": adherence,
        "missed_doses": missed,
        "open_safety_alerts": open_alerts,
        "today_plan": today_plan,
        "bpsd_analytics_30d": bpsd_analytics
    }

def compose_doctor_bot_response(intent: str, snapshot: dict, question: str) -> str:
    patient = snapshot.get("patient", {})
    patient_name = patient.get("name") or patient.get("user_id") or "patient"
    adherence = snapshot.get("adherence_7d", {})
    today_plan = snapshot.get("today_plan", {})
    mar_summary = (today_plan.get("medication_plan") or {}).get("summary", {})
    slots = (today_plan.get("medication_plan") or {}).get("slots", [])
    due_instructions = ((today_plan.get("instructions") or {}).get("due_today") or [])
    as_needed_instructions = ((today_plan.get("instructions") or {}).get("as_needed") or [])
    open_alerts = snapshot.get("open_safety_alerts", [])
    missed = snapshot.get("missed_doses", [])
    bpsd = snapshot.get("bpsd_analytics_30d", {})
    ts = snapshot.get("timestamp")

    if intent == "medications_today":
        if not slots:
            return f"{patient_name}: no medication doses are scheduled for today."
        lines = [_format_slot_line(s) for s in sorted(slots, key=lambda x: x.get("scheduled_for", ""))[:12]]
        return (
            f"{patient_name} medications for today (as of {ts}):\n"
            + "\n".join(lines)
            + f"\nSummary: expected {mar_summary.get('expected', 0)}, taken {mar_summary.get('taken_total', 0)}, "
              f"due {mar_summary.get('due', 0)}, overdue {mar_summary.get('overdue', 0)}, missed {mar_summary.get('missed', 0)}."
        )

    if intent == "missed_doses":
        if not missed:
            return f"{patient_name}: no missed/overdue doses currently detected."
        lines = []
        for dose in missed[:10]:
            lines.append(
                f"- {dose.get('name', 'Medication')} ({dose.get('dosage', '')}) "
                f"scheduled {dose.get('scheduled_for')}, overdue {dose.get('hours_overdue', 0)}h"
            )
        return f"{patient_name} missed doses:\n" + "\n".join(lines)

    if intent == "safety_alerts":
        if not open_alerts:
            return f"{patient_name}: no open safety alerts right now."
        lines = [_format_safety_alert_line(a) for a in open_alerts[:10]]
        return f"{patient_name} open safety alerts ({len(open_alerts)}):\n" + "\n".join(lines)

    if intent == "mood_behavior":
        return (
            f"{patient_name} mood/behavior (30 days): "
            f"{bpsd.get('total_observations', 0)} BPSD events, "
            f"average mood {bpsd.get('average_mood_score', 0)}, "
            f"low mood days {bpsd.get('low_mood_days', 0)}, "
            f"top symptom {bpsd.get('top_symptom') or 'none'}."
        )

    if intent == "today_instructions":
        if not due_instructions and not as_needed_instructions:
            return f"{patient_name}: no active care instructions for today."
        due_lines = [
            f"- {inst.get('title')} ({inst.get('time_of_day') or 'any time'})"
            for inst in due_instructions[:8]
        ]
        as_needed_lines = [f"- {inst.get('title')}" for inst in as_needed_instructions[:5]]
        parts = [f"{patient_name} care instructions for today:"]
        if due_lines:
            parts.append("Scheduled:")
            parts.extend(due_lines)
        if as_needed_lines:
            parts.append("As needed:")
            parts.extend(as_needed_lines)
        return "\n".join(parts)

    if intent == "compliance_check":
        expected = int(mar_summary.get("expected", 0))
        taken_total = int(mar_summary.get("taken_total", 0))
        remaining_med_doses = max(expected - taken_total, 0)
        if expected == 0:
            med_line = "No medication doses scheduled today."
        elif remaining_med_doses == 0:
            med_line = f"Medication schedule is complete today ({taken_total}/{expected})."
        else:
            med_line = f"Medication schedule is not complete: {taken_total}/{expected} taken, {remaining_med_doses} pending."
        # Instruction completion tracking does not exist yet in the data model.
        instr_line = (
            f"Active instructions due today: {len(due_instructions)}. "
            "Instruction completion tracking is not implemented yet, so fulfillment cannot be fully auto-verified."
        )
        return f"{patient_name} compliance check: {med_line} {instr_line}"

    if intent == "full_report":
        alert_count = len(open_alerts)
        return (
            f"{patient_name} full report (as of {ts}):\n"
            f"- Adherence 7d: on-time {adherence.get('adherence_percent_on_time', 0)}%, total {adherence.get('adherence_percent_total', 0)}%, "
            f"missed {adherence.get('missed_doses', 0)} doses.\n"
            f"- Today's meds: expected {mar_summary.get('expected', 0)}, taken {mar_summary.get('taken_total', 0)}, "
            f"due {mar_summary.get('due', 0)}, overdue {mar_summary.get('overdue', 0)}, missed {mar_summary.get('missed', 0)}.\n"
            f"- Safety: {alert_count} open alerts.\n"
            f"- Mood/BPSD 30d: {bpsd.get('total_observations', 0)} events, low mood days {bpsd.get('low_mood_days', 0)}, "
            f"top symptom {bpsd.get('top_symptom') or 'none'}.\n"
            f"- Care instructions: {len(due_instructions)} due today, {len(as_needed_instructions)} as needed."
        )

    if intent == "progress_summary":
        return (
            f"{patient_name} progress summary (as of {ts}): "
            f"adherence on-time {adherence.get('adherence_percent_on_time', 0)}% "
            f"(total {adherence.get('adherence_percent_total', 0)}%), "
            f"today taken {mar_summary.get('taken_total', 0)}/{mar_summary.get('expected', 0)} doses, "
            f"{len(open_alerts)} open safety alerts, "
            f"low mood days {bpsd.get('low_mood_days', 0)} in last 30 days."
        )

    return (
        f"I can report medications, missed doses, safety alerts, mood/behavior, care instructions, "
        f"and full progress for {patient_name}. Ask for one of those directly."
    )

AGENTIC_BOT_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "query_patient_data",
            "description": "Query patient snapshot data including medications, adherence, safety alerts, mood, and care instructions.",
            "parameters": {
                "type": "object",
                "properties": {
                    "data_type": {
                        "type": "string",
                        "enum": ["medications_today", "missed_doses", "safety_alerts", "mood_behavior",
                                 "today_instructions", "compliance_check", "full_report", "progress_summary"],
                        "description": "The type of patient data to query"
                    }
                },
                "required": ["data_type"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "add_medication",
            "description": "Add a new medication prescription for the patient. Use when the doctor wants to prescribe or add a medication.",
            "parameters": {
                "type": "object",
                "properties": {
                    "description": {
                        "type": "string",
                        "description": "Full description of the medication to add, e.g. 'Metformin 500mg twice daily at 8am and 8pm'"
                    }
                },
                "required": ["description"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "add_care_instruction",
            "description": "Add a care instruction or protocol for the patient.",
            "parameters": {
                "type": "object",
                "properties": {
                    "description": {
                        "type": "string",
                        "description": "Full description of the care instruction, e.g. 'Check blood pressure daily at 9am'"
                    }
                },
                "required": ["description"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "deactivate_medication",
            "description": "Stop/deactivate a medication for the patient.",
            "parameters": {
                "type": "object",
                "properties": {
                    "medication_name": {
                        "type": "string",
                        "description": "Name of the medication to deactivate"
                    }
                },
                "required": ["medication_name"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "log_patient_intake",
            "description": "Record that the patient took a specific medication dose.",
            "parameters": {
                "type": "object",
                "properties": {
                    "medication_name": {
                        "type": "string",
                        "description": "Name of the medication the patient took"
                    }
                },
                "required": ["medication_name"]
            }
        }
    }
]


async def compose_doctor_bot_response_with_llm(
    question: str, snapshot: dict,
    doctor_user: Optional[User] = None, patient_user_id: Optional[str] = None
) -> str:
    """Agentic LLM response with tool use — can both query and write patient data."""
    if not os.environ.get("OPENAI_API_KEY"):
        return compose_doctor_bot_response("progress_summary", snapshot, question)

    compact_snapshot = {
        "patient": snapshot.get("patient"),
        "timestamp": snapshot.get("timestamp"),
        "adherence_7d": snapshot.get("adherence_7d"),
        "today_medication_summary": ((snapshot.get("today_plan") or {}).get("medication_plan") or {}).get("summary", {}),
        "today_instruction_counts": {
            "due_today": len((((snapshot.get("today_plan") or {}).get("instructions") or {}).get("due_today") or [])),
            "as_needed": len((((snapshot.get("today_plan") or {}).get("instructions") or {}).get("as_needed") or [])),
        },
        "open_safety_alert_count": len(snapshot.get("open_safety_alerts") or []),
        "mood_behavior_30d": snapshot.get("bpsd_analytics_30d")
    }

    system_prompt = (
        "You are an intelligent clinical assistant for doctors managing patients with dementia.\n"
        "You can both QUERY patient data and PERFORM ACTIONS (add medications, care instructions, etc.).\n"
        "Use the available tools to fulfill the doctor's request.\n"
        "After tool calls, provide a clear summary of what was done or found.\n"
        "Keep responses concise (under 8 bullet points) with concrete numbers.\n"
        "If the doctor asks to do something you can't, explain what alternatives are available.\n"
        f"\nCurrent patient data snapshot:\n{json.dumps(compact_snapshot)}"
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question}
    ]

    # Unknown-intent fallback is READ-ONLY.  Write operations must go through
    # the explicit intent classification path (heuristic → model) which is
    # checked *before* this function is called.  Allowing the LLM fallback to
    # invoke write tools would let ambiguous prompts mutate records.
    can_write = False
    tools = [AGENTIC_BOT_TOOLS[0]]  # query_patient_data only

    try:
        client = get_openai_client()
        max_iterations = 3
        for _iteration in range(max_iterations):
            completion = await client.chat.completions.create(
                model="gpt-4o-mini",
                temperature=0.2,
                max_tokens=500,
                messages=messages,
                tools=tools
            )

            choice = completion.choices[0]
            message = choice.message

            # No tool calls — return the text response
            if not message.tool_calls:
                return (message.content or "").strip() or compose_doctor_bot_response("progress_summary", snapshot, question)

            # Process tool calls
            messages.append(message)
            for tool_call in message.tool_calls:
                fn_name = tool_call.function.name
                try:
                    args = json.loads(tool_call.function.arguments or "{}")
                except Exception:
                    args = {}

                result = ""
                if fn_name == "query_patient_data":
                    data_type = args.get("data_type", "progress_summary")
                    result = compose_doctor_bot_response(data_type, snapshot, question)

                elif fn_name == "add_medication" and can_write:
                    result = await handle_bot_add_medication(
                        doctor_user, patient_user_id, args.get("description", question)
                    )

                elif fn_name == "add_care_instruction" and can_write:
                    result = await handle_bot_add_care_instruction(
                        doctor_user, patient_user_id, args.get("description", question)
                    )

                elif fn_name == "deactivate_medication" and can_write:
                    result = await handle_bot_deactivate_medication(
                        doctor_user, patient_user_id,
                        f"stop {args.get('medication_name', '')}"
                    )

                elif fn_name == "log_patient_intake" and can_write:
                    result = await handle_bot_log_patient_intake(
                        doctor_user, patient_user_id,
                        f"mark {args.get('medication_name', '')} as taken"
                    )
                else:
                    result = f"Tool '{fn_name}' is not available in current context."

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result
                })

        # If we exhausted iterations, get final response
        final = await client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.2,
            max_tokens=300,
            messages=messages
        )
        return (final.choices[0].message.content or "").strip() or "Request processed."

    except Exception as exc:
        logger.warning(f"Agentic doctor bot LLM failed: {exc}")
    return compose_doctor_bot_response("progress_summary", snapshot, question)

async def generate_tts_audio_bytes(text: str, voice: str = "nova") -> Optional[bytes]:
    if not text or not os.environ.get("OPENAI_API_KEY"):
        return None
    try:
        client = get_openai_client()
        response = await client.audio.speech.create(
            model="tts-1",
            voice=voice,
            input=text[:4000],
            speed=0.9
        )
        return response.content
    except Exception as exc:
        logger.warning(f"TTS generation for external bot failed: {exc}")
        return None

async def transcribe_audio_bytes(audio_bytes: bytes, filename: str = "audio.ogg") -> Optional[str]:
    if not audio_bytes or not os.environ.get("OPENAI_API_KEY"):
        return None
    try:
        client = get_openai_client()
        stream = io.BytesIO(audio_bytes)
        stream.name = filename
        result = await client.audio.transcriptions.create(
            model=os.environ.get("STT_MODEL", "gpt-4o-mini-transcribe"),
            file=stream
        )
        text = getattr(result, "text", None)
        return text.strip() if text else None
    except Exception as exc:
        logger.warning(f"Audio transcription failed: {exc}")
        return None

async def download_telegram_voice_file(file_id: str) -> Tuple[Optional[bytes], Optional[str]]:
    token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
    if not token:
        return None, None
    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            file_meta_resp = await client.post(
                f"https://api.telegram.org/bot{token}/getFile",
                json={"file_id": file_id}
            )
            file_meta = file_meta_resp.json() if file_meta_resp.status_code == 200 else {}
            file_path = (file_meta.get("result") or {}).get("file_path")
            if not file_path:
                return None, None
            file_resp = await client.get(f"https://api.telegram.org/file/bot{token}/{file_path}")
            if file_resp.status_code != 200:
                return None, None
            name = file_path.split("/")[-1] if "/" in file_path else "telegram_audio.ogg"
            return file_resp.content, name
    except Exception as exc:
        logger.warning(f"Telegram voice download failed: {exc}")
        return None, None

async def download_twilio_media(media_url: str) -> Tuple[Optional[bytes], Optional[str]]:
    if not media_url:
        return None, None
    sid = os.environ.get("TWILIO_ACCOUNT_SID", "").strip()
    token = os.environ.get("TWILIO_AUTH_TOKEN", "").strip()
    auth = (sid, token) if sid and token else None
    try:
        async with httpx.AsyncClient(timeout=20.0, follow_redirects=True) as client:
            resp = await client.get(media_url, auth=auth)
            if resp.status_code != 200:
                return None, None
            content_type = resp.headers.get("content-type")
            return resp.content, content_type
    except Exception as exc:
        logger.warning(f"Twilio media download failed: {exc}")
        return None, None

async def send_telegram_text_message(chat_id: str, text: str) -> dict:
    token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
    if not token:
        return {"sent": False, "reason": "telegram_token_missing"}
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.post(
                f"https://api.telegram.org/bot{token}/sendMessage",
                json={"chat_id": chat_id, "text": text[:4000]}
            )
        return {"sent": 200 <= resp.status_code < 300, "status_code": resp.status_code}
    except Exception as exc:
        return {"sent": False, "reason": str(exc)[:240]}

async def send_telegram_voice_message(chat_id: str, text: str) -> dict:
    """Send a voice note via Telegram.

    Telegram's ``sendVoice`` expects OGG/Opus, but OpenAI TTS returns MP3.
    We first try ``sendVoice`` with the MP3 (Telegram *sometimes* accepts it).
    If that fails or is unavailable we fall back to ``sendAudio`` which is
    format-agnostic and reliably delivers the audio file.
    """
    token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
    if not token:
        return {"sent": False, "reason": "telegram_token_missing"}
    audio = await generate_tts_audio_bytes(text)
    if not audio:
        return {"sent": False, "reason": "tts_unavailable"}
    try:
        async with httpx.AsyncClient(timeout=25.0) as client:
            # Try sendAudio (format-agnostic, always works with MP3).
            resp = await client.post(
                f"https://api.telegram.org/bot{token}/sendAudio",
                data={"chat_id": chat_id, "title": "Voice summary"},
                files={"audio": ("summary.mp3", audio, "audio/mpeg")}
            )
        return {"sent": 200 <= resp.status_code < 300, "status_code": resp.status_code}
    except Exception as exc:
        return {"sent": False, "reason": str(exc)[:240]}

async def log_external_bot_message(
    channel: str,
    peer_id: str,
    doctor_user_id: Optional[str],
    patient_user_id: Optional[str],
    incoming_text: Optional[str],
    response_text: str,
    intent: Optional[str],
    status: str
):
    doc = {
        "id": f"botmsg_{uuid.uuid4().hex[:12]}",
        "channel": channel,
        "peer_id": peer_id,
        "doctor_user_id": doctor_user_id,
        "patient_user_id": patient_user_id,
        "incoming_text": incoming_text,
        "response_text": response_text,
        "intent": intent,
        "status": status,
        "created_at": datetime.now(timezone.utc).isoformat()
    }
    await db.external_bot_message_logs.insert_one(doc)

async def handle_external_link_code_message(
    channel: str,
    peer_id: str,
    peer_display_name: Optional[str],
    text: str
) -> str:
    code_match = re.search(r"(?:/link|link)\s+([A-Za-z0-9]+)", text.strip(), flags=re.IGNORECASE)
    if not code_match:
        return "To connect this chat, send: /link YOUR_CODE"
    code = code_match.group(1).strip().upper()
    now_iso = datetime.now(timezone.utc).isoformat()
    code_doc = await db.external_bot_link_codes.find_one(
        {"code": code, "status": "pending"},
        {"_id": 0}
    )
    if not code_doc:
        return "Link code not found or already used. Generate a new one from the app."
    if code_doc.get("channel") != channel:
        return f"This code is for {code_doc.get('channel')} only."
    if code_doc.get("expires_at", now_iso) < now_iso:
        await db.external_bot_link_codes.update_one({"id": code_doc["id"]}, {"$set": {"status": "expired"}})
        return "This link code has expired. Generate a new one from the app."

    doctor_doc = await db.users.find_one({"user_id": code_doc.get("doctor_user_id")}, {"_id": 0})
    if not doctor_doc:
        return "Doctor account not found for this code."
    doctor_user = User(**doctor_doc)
    if not role_can_use_external_bot(doctor_user):
        return "This account is not allowed to use external bot access."

    normalized_peer = normalize_external_peer_id(channel, peer_id)
    existing = await db.external_bot_links.find_one(
        {"channel": channel, "peer_id": normalized_peer},
        {"_id": 0}
    )
    link_doc = {
        "id": existing.get("id") if existing else f"extlink_{uuid.uuid4().hex[:12]}",
        "channel": channel,
        "peer_id": normalized_peer,
        "peer_display_name": peer_display_name,
        "doctor_user_id": doctor_user.user_id,
        "patient_user_id": code_doc.get("patient_user_id"),
        "active": True,
        "created_at": existing.get("created_at") if existing else now_iso,
        "updated_at": now_iso,
        "last_seen_at": now_iso
    }
    await db.external_bot_links.update_one(
        {"id": link_doc["id"]},
        {"$set": link_doc},
        upsert=True
    )
    await db.external_bot_link_codes.update_one(
        {"id": code_doc["id"]},
        {"$set": {"status": "used", "used_at": now_iso, "used_by_peer_id": normalized_peer}}
    )
    patient_hint = ""
    if code_doc.get("patient_user_id"):
        patient = await db.users.find_one({"user_id": code_doc["patient_user_id"]}, {"_id": 0, "name": 1})
        if patient:
            patient_hint = f" Default patient set to {patient.get('name')}."
    return f"Linked successfully to doctor account {doctor_user.name}.{patient_hint} You can now ask patient progress questions."


# ==================== BOT WRITE HANDLERS ====================

async def parse_medication_from_text(text: str) -> dict:
    """Use LLM to extract medication details from natural language."""
    if not os.environ.get("OPENAI_API_KEY"):
        return {}
    prompt = (
        "Extract medication prescription details from the doctor's message.\n"
        "Return strict JSON:\n"
        "{\n"
        '  "name": "medication name",\n'
        '  "dosage": "e.g. 500mg",\n'
        '  "frequency": "daily|twice_daily|three_times_daily|every_other_day|weekly|custom",\n'
        '  "times_per_day": 1,\n'
        '  "scheduled_times": ["HH:MM", ...],\n'
        '  "instructions": "optional special instructions",\n'
        '  "prescribing_doctor": "doctor name if mentioned"\n'
        "}\n"
        "For frequency mapping:\n"
        "- once a day / daily → daily, times_per_day=1\n"
        "- twice a day / bid → twice_daily, times_per_day=2\n"
        "- three times a day / tid → three_times_daily, times_per_day=3\n"
        "- every other day → every_other_day\n"
        "- weekly / once a week → weekly\n"
        "Convert times like '8am' to '08:00', '8pm' to '20:00'.\n"
        "If no times specified, use reasonable defaults based on frequency.\n"
    )
    try:
        client = get_openai_client()
        completion = await client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0,
            max_tokens=200,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": text}
            ]
        )
        raw = (completion.choices[0].message.content or "").strip()
        return json.loads(raw) if raw else {}
    except Exception as e:
        logger.warning(f"parse_medication_from_text failed: {e}")
        return {}


async def parse_care_instruction_from_text(text: str) -> dict:
    """Use LLM to extract care instruction details from natural language."""
    if not os.environ.get("OPENAI_API_KEY"):
        return {}
    prompt = (
        "Extract care instruction details from the doctor's message.\n"
        "Return strict JSON:\n"
        "{\n"
        '  "title": "short title for the instruction",\n'
        '  "instruction_text": "full instruction text",\n'
        '  "summary": "brief summary",\n'
        '  "frequency": "daily|weekly|as_needed",\n'
        '  "time_of_day": "morning|afternoon|evening|null",\n'
        '  "policy_type": "general|medication",\n'
        '  "day_of_week": "monday|tuesday|...|null (only for weekly)"\n'
        "}\n"
        "If the instruction is about medications, set policy_type to 'medication'.\n"
        "For general care procedures, set policy_type to 'general'.\n"
    )
    try:
        client = get_openai_client()
        completion = await client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0,
            max_tokens=300,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": text}
            ]
        )
        raw = (completion.choices[0].message.content or "").strip()
        return json.loads(raw) if raw else {}
    except Exception as e:
        logger.warning(f"parse_care_instruction_from_text failed: {e}")
        return {}


async def handle_bot_add_medication(
    doctor_user: User, patient_user_id: str, text: str
) -> str:
    """Parse and create a medication from a doctor's bot message."""
    try:
        owner_id = await resolve_target_user_id(doctor_user, patient_user_id, require_write=True)
    except HTTPException:
        return "You do not have write access to this patient's records."

    parsed = await parse_medication_from_text(text)
    name = (parsed.get("name") or "").strip()
    dosage = (parsed.get("dosage") or "").strip()
    if not name:
        return "I could not identify the medication name. Please specify, e.g.: 'Add medication Metformin 500mg twice daily at 8am and 8pm'"

    frequency = parsed.get("frequency", "daily")
    times_per_day = int(parsed.get("times_per_day", 1))
    scheduled_times = parsed.get("scheduled_times", [])
    cleaned_times = [normalize_hhmm(t) for t in scheduled_times if t]

    med_obj = Medication(
        user_id=owner_id,
        name=name,
        dosage=dosage or "as prescribed",
        frequency=frequency,
        times_per_day=times_per_day,
        scheduled_times=cleaned_times,
        prescribing_doctor=parsed.get("prescribing_doctor") or doctor_user.name,
        instructions=parsed.get("instructions")
    )
    doc = med_obj.model_dump()
    doc["created_at"] = doc["created_at"].isoformat()
    doc["updated_at"] = doc["updated_at"].isoformat()
    await db.medications.insert_one(doc)
    asyncio.create_task(notify_patient_update(
        owner_id, "medication_added",
        f"{doctor_user.name} prescribed {name}. Ask your voice assistant for details."
    ))

    patient = await db.users.find_one({"user_id": owner_id}, {"_id": 0, "name": 1})
    patient_name = (patient or {}).get("name", owner_id)
    times_str = ", ".join([format_hhmm_for_voice(t) for t in cleaned_times]) if cleaned_times else "no specific times"
    return (
        f"Medication created for {patient_name}:\n"
        f"- Name: {name}\n"
        f"- Dosage: {dosage or 'as prescribed'}\n"
        f"- Frequency: {frequency.replace('_', ' ')}\n"
        f"- Scheduled times: {times_str}\n"
        f"- Prescribed by: {med_obj.prescribing_doctor or 'N/A'}"
    )


async def handle_bot_add_care_instruction(
    doctor_user: User, patient_user_id: str, text: str
) -> str:
    """Parse and create a care instruction from a doctor's bot message."""
    try:
        owner_id = await resolve_target_user_id(doctor_user, patient_user_id, require_write=True)
    except HTTPException:
        return "You do not have write access to this patient's records."

    parsed = await parse_care_instruction_from_text(text)
    title = (parsed.get("title") or "").strip()
    instruction_text = (parsed.get("instruction_text") or "").strip()
    if not title or not instruction_text:
        return "I could not parse the instruction details. Please specify, e.g.: 'Add instruction: Check blood pressure daily at 9am'"

    policy_type = normalize_policy_type(parsed.get("policy_type", "general"))
    frequency = normalize_frequency(parsed.get("frequency", "daily"))
    signoff_required = policy_type == "medication"
    signoff_status = normalize_signoff_status(None, signoff_required)
    regimen_key = re.sub(r"\s+", "_", title.lower()) if policy_type == "medication" else None

    version_query = {"user_id": owner_id, "policy_type": policy_type}
    if regimen_key:
        version_query["regimen_key"] = regimen_key
    latest = await db.care_instructions.find_one(version_query, {"_id": 0}, sort=[("version", -1)])
    next_version = int(latest.get("version", 0) + 1) if latest else 1

    instruction_obj = CareInstruction(
        user_id=owner_id,
        title=title,
        instruction_text=instruction_text,
        summary=parsed.get("summary"),
        frequency=frequency,
        day_of_week=normalize_day_of_week(parsed.get("day_of_week")),
        time_of_day=(parsed.get("time_of_day") or "").strip() or None,
        policy_type=policy_type,
        regimen_key=regimen_key,
        version=next_version,
        status="draft" if signoff_required else "active",
        signoff_required=signoff_required,
        signoff_status=signoff_status,
        active=not signoff_required,
        uploaded_by_user_id=doctor_user.user_id,
        uploaded_by_role=doctor_user.role
    )
    doc = instruction_obj.model_dump()
    doc["created_at"] = doc["created_at"].isoformat()
    doc["updated_at"] = doc["updated_at"].isoformat()
    doc["search_text"] = build_instruction_search_text(doc)
    await db.care_instructions.insert_one(doc)
    await upsert_instruction_chunks(doc)
    asyncio.create_task(notify_patient_update(
        owner_id, "instruction_added",
        f"{doctor_user.name} added: {title}. Ask your voice assistant about it."
    ))

    patient = await db.users.find_one({"user_id": owner_id}, {"_id": 0, "name": 1})
    patient_name = (patient or {}).get("name", owner_id)
    signoff_note = " (requires clinician signoff before becoming active)" if signoff_required else ""
    return (
        f"Care instruction created for {patient_name}:\n"
        f"- Title: {title}\n"
        f"- Frequency: {frequency}\n"
        f"- Type: {policy_type}\n"
        f"- Status: {'draft' if signoff_required else 'active'}{signoff_note}"
    )


async def handle_bot_deactivate_medication(
    doctor_user: User, patient_user_id: str, text: str
) -> str:
    """Deactivate a medication by matching name from the doctor's message."""
    try:
        owner_id = await resolve_target_user_id(doctor_user, patient_user_id, require_write=True)
    except HTTPException:
        return "You do not have write access to this patient's records."

    meds = await db.medications.find(
        {"user_id": owner_id, "active": True}, {"_id": 0}
    ).to_list(100)
    if not meds:
        return "This patient has no active medications to deactivate."

    # Extract medication name from text using LLM
    parsed = await parse_medication_from_text(text)
    target_name = (parsed.get("name") or "").strip().lower()

    # Fuzzy match against active medications
    matched = None
    for med in meds:
        med_name = (med.get("name") or "").lower()
        if target_name and (target_name in med_name or med_name in target_name):
            matched = med
            break
        # Word-level match
        for word in target_name.split():
            if len(word) > 3 and word in med_name:
                matched = med
                break
        if matched:
            break

    if not matched:
        names = [f"- {m.get('name', '')} ({m.get('dosage', '')})" for m in meds[:10]]
        return f"Could not identify which medication to deactivate. Active medications:\n" + "\n".join(names)

    await db.medications.update_one(
        {"id": matched["id"], "user_id": owner_id},
        {"$set": {"active": False, "updated_at": datetime.now(timezone.utc).isoformat()}}
    )
    return f"Deactivated medication: {matched.get('name', '')} ({matched.get('dosage', '')})"


async def handle_bot_log_patient_intake(
    doctor_user: User, patient_user_id: str, text: str
) -> str:
    """Log a medication intake for a patient from a doctor/caregiver bot message."""
    try:
        owner_id = await resolve_target_user_id(doctor_user, patient_user_id, require_write=True)
    except HTTPException:
        return "You do not have write access to this patient's records."

    meds = await db.medications.find(
        {"user_id": owner_id, "active": True}, {"_id": 0}
    ).to_list(100)
    if not meds:
        return "This patient has no active medications."

    parsed = await parse_medication_from_text(text)
    target_name = (parsed.get("name") or "").strip().lower()

    matched = None
    for med in meds:
        med_name = (med.get("name") or "").lower()
        if target_name and (target_name in med_name or med_name in target_name):
            matched = med
            break
        for word in target_name.split():
            if len(word) > 3 and word in med_name:
                matched = med
                break
        if matched:
            break

    if not matched:
        if len(meds) == 1:
            matched = meds[0]
        else:
            names = [f"- {m.get('name', '')} ({m.get('dosage', '')})" for m in meds[:10]]
            return f"Could not identify which medication. Active medications:\n" + "\n".join(names)

    now_utc = datetime.now(timezone.utc)
    log_obj = MedicationIntakeLog(
        user_id=owner_id,
        medication_id=matched["id"],
        status="taken",
        source="caregiver",
        notes=f"Logged by {doctor_user.name} via bot",
        recorded_by_user_id=doctor_user.user_id,
        confirmed_at=now_utc
    )
    doc = log_obj.model_dump()
    doc["confirmed_at"] = doc["confirmed_at"].isoformat()
    await db.medication_intake_logs.insert_one(doc)

    return f"Recorded intake: {matched.get('name', '')} ({matched.get('dosage', '')}) marked as taken."


async def handle_bot_update_medication(
    doctor_user: User, patient_user_id: str, text: str
) -> str:
    """Update an existing medication from a doctor's bot message."""
    try:
        owner_id = await resolve_target_user_id(doctor_user, patient_user_id, require_write=True)
    except HTTPException:
        return "You do not have write access to this patient's records."

    meds = await db.medications.find(
        {"user_id": owner_id, "active": True}, {"_id": 0}
    ).to_list(100)
    if not meds:
        return "This patient has no active medications to update."

    parsed = await parse_medication_from_text(text)
    target_name = (parsed.get("name") or "").strip().lower()

    matched = None
    for med in meds:
        med_name = (med.get("name") or "").lower()
        if target_name and (target_name in med_name or med_name in target_name):
            matched = med
            break
        for word in target_name.split():
            if len(word) > 3 and word in med_name:
                matched = med
                break
        if matched:
            break

    if not matched:
        names = [f"- {m.get('name', '')} ({m.get('dosage', '')})" for m in meds[:10]]
        return "Could not identify which medication to update. Active medications:\n" + "\n".join(names)

    update_fields = {}
    if parsed.get("dosage"):
        update_fields["dosage"] = parsed["dosage"]
    if parsed.get("frequency"):
        update_fields["frequency"] = parsed["frequency"]
    if parsed.get("scheduled_times"):
        update_fields["scheduled_times"] = [normalize_hhmm(t) for t in parsed["scheduled_times"] if t]
    if parsed.get("times_per_day"):
        update_fields["times_per_day"] = int(parsed["times_per_day"])
    if parsed.get("instructions"):
        update_fields["instructions"] = parsed["instructions"]

    if not update_fields:
        return f"Could not determine what to change for {matched.get('name', '')}. Please specify, e.g.: 'Update metformin dosage to 1000mg'"

    update_fields["updated_at"] = datetime.now(timezone.utc).isoformat()
    await db.medications.update_one(
        {"id": matched["id"], "user_id": owner_id},
        {"$set": update_fields}
    )

    changes = ", ".join(f"{k}={v}" for k, v in update_fields.items() if k != "updated_at")
    return f"Updated {matched.get('name', '')}: {changes}"


# ==================== BOT QUERY HANDLER ====================

async def handle_external_doctor_query(
    doctor_user: User,
    question: str,
    patient_user_id: str
) -> Tuple[str, str, dict]:
    intent = classify_doctor_bot_intent_heuristic(question)
    if intent == "unknown":
        intent = await classify_doctor_bot_intent_model(question)

    # Handle write intents
    if intent in DOCTOR_BOT_WRITE_INTENTS:
        if intent == "add_medication":
            response = await handle_bot_add_medication(doctor_user, patient_user_id, question)
        elif intent == "add_care_instruction":
            response = await handle_bot_add_care_instruction(doctor_user, patient_user_id, question)
        elif intent == "deactivate_medication":
            response = await handle_bot_deactivate_medication(doctor_user, patient_user_id, question)
        elif intent == "log_patient_intake":
            response = await handle_bot_log_patient_intake(doctor_user, patient_user_id, question)
        elif intent == "update_medication":
            response = await handle_bot_update_medication(doctor_user, patient_user_id, question)
        else:
            response = "Write operation not yet supported."
        return response, intent, {}

    # Handle read intents
    snapshot = await build_doctor_patient_snapshot(patient_user_id, doctor_user)
    if intent == "unknown":
        response = await compose_doctor_bot_response_with_llm(question, snapshot, doctor_user, patient_user_id)
    else:
        response = compose_doctor_bot_response(intent, snapshot, question)
    return response, intent, snapshot

async def handle_external_doctor_message(
    channel: str,
    peer_id: str,
    peer_display_name: Optional[str],
    text: str,
    prefer_voice: bool = False
) -> dict:
    message_text = (text or "").strip()
    normalized_peer = normalize_external_peer_id(channel, peer_id)

    if re.match(r"^\s*(/link|link)\b", message_text, flags=re.IGNORECASE):
        response_text = await handle_external_link_code_message(channel, normalized_peer, peer_display_name, message_text)
        await log_external_bot_message(channel, normalized_peer, None, None, message_text, response_text, "link", "ok")
        return {"text": response_text, "voice_text": response_text if prefer_voice else None, "status": "ok", "intent": "link"}

    link = await db.external_bot_links.find_one(
        {"channel": channel, "peer_id": normalized_peer, "active": True},
        {"_id": 0}
    )
    if not link:
        response_text = "This chat is not linked yet. In the app, generate a bot link code and send: /link YOUR_CODE"
        await log_external_bot_message(channel, normalized_peer, None, None, message_text, response_text, "unlinked", "error")
        return {"text": response_text, "voice_text": None, "status": "error", "intent": "unlinked"}

    doctor_doc = await db.users.find_one({"user_id": link.get("doctor_user_id")}, {"_id": 0})
    if not doctor_doc:
        response_text = "Linked doctor account was not found. Please relink this chat."
        await log_external_bot_message(channel, normalized_peer, link.get("doctor_user_id"), None, message_text, response_text, "doctor_missing", "error")
        return {"text": response_text, "voice_text": None, "status": "error", "intent": "doctor_missing"}

    doctor_user = User(**doctor_doc)
    if not role_can_use_external_bot(doctor_user):
        response_text = "This account role is not allowed to use the external doctor bot."
        await log_external_bot_message(channel, normalized_peer, doctor_user.user_id, None, message_text, response_text, "forbidden_role", "error")
        return {"text": response_text, "voice_text": None, "status": "error", "intent": "forbidden_role"}
    if doctor_user.role == "clinician" and doctor_user.clinician_approval_status != "approved":
        response_text = "Clinician account is not approved for bot access."
        await log_external_bot_message(channel, normalized_peer, doctor_user.user_id, None, message_text, response_text, "clinician_unapproved", "error")
        return {"text": response_text, "voice_text": None, "status": "error", "intent": "clinician_unapproved"}

    if not _is_premium(doctor_doc):
        response_text = (
            "External bot access requires AlzaHelp Premium ($9.99/mo). "
            "Visit your dashboard to upgrade and unlock Telegram/WhatsApp monitoring."
        )
        await log_external_bot_message(channel, normalized_peer, doctor_user.user_id, None, message_text, response_text, "premium_required", "error")
        return {"text": response_text, "voice_text": None, "status": "error", "intent": "premium_required"}

    if re.match(r"^\s*/patient\b", message_text, flags=re.IGNORECASE):
        parts = message_text.split()
        if len(parts) < 2:
            response_text = "Usage: /patient user_xxxxxxxxxxxx"
            await log_external_bot_message(channel, normalized_peer, doctor_user.user_id, None, message_text, response_text, "patient_select", "error")
            return {"text": response_text, "voice_text": None, "status": "error", "intent": "patient_select"}
        requested_id = parts[1].strip()
        try:
            owner_id = await resolve_target_user_id(doctor_user, requested_id)
        except HTTPException:
            response_text = "You do not have access to that patient."
            await log_external_bot_message(channel, normalized_peer, doctor_user.user_id, requested_id, message_text, response_text, "patient_select", "error")
            return {"text": response_text, "voice_text": None, "status": "error", "intent": "patient_select"}

        await db.external_bot_links.update_one(
            {"id": link["id"]},
            {"$set": {"patient_user_id": owner_id, "updated_at": datetime.now(timezone.utc).isoformat()}}
        )
        patient_doc = await db.users.find_one({"user_id": owner_id}, {"_id": 0, "name": 1})
        response_text = f"Default patient set to {patient_doc.get('name') if patient_doc else owner_id}."
        await log_external_bot_message(channel, normalized_peer, doctor_user.user_id, owner_id, message_text, response_text, "patient_select", "ok")
        return {"text": response_text, "voice_text": None, "status": "ok", "intent": "patient_select"}

    explicit_patient_id = extract_patient_id_from_text(message_text)
    selected_patient_id = explicit_patient_id or link.get("patient_user_id")
    if selected_patient_id:
        try:
            selected_patient_id = await resolve_target_user_id(doctor_user, selected_patient_id)
        except HTTPException:
            selected_patient_id = None

    if not selected_patient_id:
        accessible = await get_doctor_accessible_patients(doctor_user, limit=10)
        if len(accessible) == 1:
            selected_patient_id = accessible[0]["user_id"]
            await db.external_bot_links.update_one(
                {"id": link["id"]},
                {"$set": {"patient_user_id": selected_patient_id, "updated_at": datetime.now(timezone.utc).isoformat()}}
            )
        else:
            if not accessible:
                response_text = "No linked patients found for this account."
            else:
                listing = "\n".join([f"- {p.get('name', p['user_id'])}: {p['user_id']}" for p in accessible[:10]])
                response_text = (
                    "Please select a patient first using /patient <patient_user_id>.\n"
                    f"Available patients:\n{listing}"
                )
            await log_external_bot_message(channel, normalized_peer, doctor_user.user_id, None, message_text, response_text, "patient_required", "error")
            return {"text": response_text, "voice_text": None, "status": "error", "intent": "patient_required"}

    try:
        response_text, intent, _snapshot = await handle_external_doctor_query(
            doctor_user=doctor_user,
            question=message_text,
            patient_user_id=selected_patient_id
        )
        await db.external_bot_links.update_one(
            {"id": link["id"]},
            {
                "$set": {
                    "last_seen_at": datetime.now(timezone.utc).isoformat(),
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                    "peer_display_name": peer_display_name or link.get("peer_display_name")
                }
            }
        )
        await log_external_bot_message(
            channel,
            normalized_peer,
            doctor_user.user_id,
            selected_patient_id,
            message_text,
            response_text,
            intent,
            "ok"
        )
        return {
            "text": response_text,
            "voice_text": response_text if prefer_voice else None,
            "status": "ok",
            "intent": intent,
            "patient_user_id": selected_patient_id
        }
    except HTTPException as exc:
        response_text = f"Could not complete request: {exc.detail}"
        await log_external_bot_message(
            channel,
            normalized_peer,
            doctor_user.user_id,
            selected_patient_id,
            message_text,
            response_text,
            "query_error",
            "error"
        )
        return {"text": response_text, "voice_text": None, "status": "error", "intent": "query_error"}

@api_router.get("/medications", response_model=List[dict])
async def get_medications(
    target_user_id: Optional[str] = None,
    current_user: User = Depends(get_current_user)
):
    owner_id = await resolve_target_user_id(current_user, target_user_id)
    meds = await db.medications.find({"user_id": owner_id}, {"_id": 0}).sort("created_at", -1).to_list(300)
    return meds

@api_router.post("/medications", response_model=dict)
async def create_medication(
    medication: MedicationCreate,
    target_user_id: Optional[str] = None,
    current_user: User = Depends(get_current_user)
):
    guard_demo_write(current_user)
    owner_id = await resolve_target_user_id(current_user, target_user_id, require_write=True)
    cleaned_times = [normalize_hhmm(t) for t in medication.scheduled_times if t]
    med_obj = Medication(
        user_id=owner_id,
        **{**medication.model_dump(), "scheduled_times": cleaned_times}
    )
    doc = med_obj.model_dump()
    doc["created_at"] = doc["created_at"].isoformat()
    doc["updated_at"] = doc["updated_at"].isoformat()
    await db.medications.insert_one(doc)
    if owner_id != current_user.user_id:
        asyncio.create_task(notify_patient_update(
            owner_id, "medication_added",
            f"{current_user.name} added {medication.name} ({medication.dosage or 'as prescribed'}). Open the app to review."
        ))
    if "_id" in doc:
        del doc["_id"]
    return doc

@api_router.put("/medications/{medication_id}", response_model=dict)
async def update_medication(
    medication_id: str,
    medication: MedicationUpdate,
    target_user_id: Optional[str] = None,
    current_user: User = Depends(get_current_user)
):
    guard_demo_write(current_user)
    owner_id = await resolve_target_user_id(current_user, target_user_id, require_write=True)
    update_data = {k: v for k, v in medication.model_dump().items() if v is not None}
    if "scheduled_times" in update_data:
        update_data["scheduled_times"] = [normalize_hhmm(t) for t in update_data["scheduled_times"] if t]
    update_data["updated_at"] = datetime.now(timezone.utc).isoformat()
    result = await db.medications.update_one(
        {"id": medication_id, "user_id": owner_id},
        {"$set": update_data}
    )
    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="Medication not found")
    updated = await db.medications.find_one({"id": medication_id, "user_id": owner_id}, {"_id": 0})
    return updated

@api_router.delete("/medications/{medication_id}")
async def delete_medication(
    medication_id: str,
    target_user_id: Optional[str] = None,
    current_user: User = Depends(get_current_user)
):
    guard_demo_write(current_user)
    owner_id = await resolve_target_user_id(current_user, target_user_id, require_write=True)
    result = await db.medications.delete_one({"id": medication_id, "user_id": owner_id})
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Medication not found")
    return {"message": "Medication deleted"}

@api_router.post("/medications/{medication_id}/intake", response_model=dict)
async def create_intake_log(
    medication_id: str,
    intake: MedicationIntakeCreate,
    target_user_id: Optional[str] = None,
    current_user: User = Depends(get_current_user)
):
    owner_id = await resolve_target_user_id(current_user, target_user_id, require_write=True)
    medication = await db.medications.find_one({"id": medication_id, "user_id": owner_id}, {"_id": 0})
    if not medication:
        raise HTTPException(status_code=404, detail="Medication not found")

    status = (intake.status or "taken").strip().lower()
    if status not in {"taken", "missed"}:
        raise HTTPException(status_code=400, detail="Status must be 'taken' or 'missed'")

    scheduled_for = None
    if intake.scheduled_for:
        try:
            scheduled_for = datetime.fromisoformat(intake.scheduled_for.replace("Z", "+00:00"))
            if scheduled_for.tzinfo is None:
                scheduled_for = scheduled_for.replace(tzinfo=timezone.utc)
            scheduled_for = scheduled_for.astimezone(timezone.utc)
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid scheduled_for datetime")

    source = (intake.source or "manual").strip().lower()
    if source not in {"manual", "voice", "mar", "caregiver", "auto"}:
        source = "manual"

    now_dt = datetime.now(timezone.utc)
    delay_minutes = None
    if status == "taken" and scheduled_for:
        delay_minutes = int(round((now_dt - scheduled_for).total_seconds() / 60))

    log_obj = MedicationIntakeLog(
        user_id=owner_id,
        medication_id=medication_id,
        status=status,
        scheduled_for=scheduled_for,
        source=source,
        notes=(intake.notes or None),
        recorded_by_user_id=current_user.user_id,
        delay_minutes=delay_minutes,
        confirmed_at=now_dt
    )
    doc = log_obj.model_dump()
    doc["confirmed_at"] = doc["confirmed_at"].isoformat()
    if doc.get("scheduled_for"):
        doc["scheduled_for"] = doc["scheduled_for"].isoformat()
    await db.medication_intake_logs.insert_one(doc)
    # Resolve any missed-dose markers once a dose is confirmed.
    if status == "taken" and doc.get("scheduled_for"):
        await db.medication_missed_alerts.delete_many(
            {
                "user_id": owner_id,
                "medication_id": medication_id,
                "scheduled_for": doc["scheduled_for"]
            }
        )
        await db.safety_alerts.update_many(
            {
                "user_id": owner_id,
                "event_type": "missed_medication_dose",
                "acknowledged": False,
                "medication_id": medication_id,
                "scheduled_for": doc["scheduled_for"]
            },
            {
                "$set": {
                    "acknowledged": True,
                    "acknowledged_at": datetime.now(timezone.utc).isoformat(),
                    "acknowledged_by_user_id": current_user.user_id,
                    "escalation_status": "resolved_after_intake",
                    "escalation_next_at": None
                }
            }
        )
    if "_id" in doc:
        del doc["_id"]
    return doc

@api_router.get("/medications/mar", response_model=dict)
async def get_medication_mar(
    on_date: Optional[str] = None,
    days: int = 1,
    target_user_id: Optional[str] = None,
    current_user: User = Depends(get_current_user)
):
    owner_id = await resolve_target_user_id(current_user, target_user_id)
    days = max(1, min(days, 7))
    base_day = parse_yyyy_mm_dd(on_date) if on_date else datetime.now(timezone.utc).date()
    if on_date and not base_day:
        raise HTTPException(status_code=400, detail="on_date must use YYYY-MM-DD")

    start_at = datetime(base_day.year, base_day.month, base_day.day, tzinfo=timezone.utc)
    end_day = base_day + timedelta(days=days - 1)
    end_at = datetime(end_day.year, end_day.month, end_day.day, 23, 59, 59, tzinfo=timezone.utc)
    mar = await _build_medication_mar_for_period(owner_id, start_at, end_at)
    expected = mar["summary"]["expected"]
    taken_total = mar["summary"]["taken_total"]
    on_time = mar["summary"]["taken_on_time"]
    mar["period"] = {
        "on_date": base_day.isoformat(),
        "days": days,
        "start_at": start_at.isoformat(),
        "end_at": end_at.isoformat()
    }
    mar["summary"]["adherence_percent_on_time"] = round((on_time / expected) * 100, 1) if expected > 0 else 0.0
    mar["summary"]["adherence_percent_total"] = round((taken_total / expected) * 100, 1) if expected > 0 else 0.0
    return mar

@api_router.get("/medications/adherence", response_model=dict)
async def medication_adherence_summary(
    days: int = 7,
    target_user_id: Optional[str] = None,
    current_user: User = Depends(get_current_user)
):
    owner_id = await resolve_target_user_id(current_user, target_user_id)
    days = max(1, min(days, 90))
    end_at = datetime.now(timezone.utc)
    start_at = end_at - timedelta(days=days)
    mar = await _build_medication_mar_for_period(owner_id, start_at, end_at)
    summary = mar.get("summary", {})
    expected_count = int(summary.get("expected", 0))
    taken_on_time = int(summary.get("taken_on_time", 0))
    taken_late = int(summary.get("taken_late", 0))
    taken_total = int(summary.get("taken_total", 0))
    pending_count = int(summary.get("due", 0)) + int(summary.get("upcoming", 0))
    missed_count = int(summary.get("missed", 0)) + int(summary.get("overdue", 0))
    adherence_on_time = round((taken_on_time / expected_count) * 100, 1) if expected_count > 0 else 0.0
    adherence_total = round((taken_total / expected_count) * 100, 1) if expected_count > 0 else 0.0
    return {
        "period_days": days,
        "expected_doses": expected_count,
        "taken_on_time": taken_on_time,
        "taken_late": taken_late,
        "taken_total": taken_total,
        "pending_doses": pending_count,
        "missed_doses": missed_count,
        "adherence_percent": adherence_on_time,
        "adherence_percent_on_time": adherence_on_time,
        "adherence_percent_total": adherence_total
    }

@api_router.get("/medications/missed", response_model=List[dict])
async def medication_missed_doses(
    hours_overdue: int = 2,
    target_user_id: Optional[str] = None,
    current_user: User = Depends(get_current_user)
):
    owner_id = await resolve_target_user_id(current_user, target_user_id)
    hours_overdue = max(1, min(hours_overdue, 24))
    end_at = datetime.now(timezone.utc)
    start_at = end_at - timedelta(days=2)
    overdue_list = []
    cutoff = end_at - timedelta(hours=hours_overdue)

    mar = await _build_medication_mar_for_period(owner_id, start_at, end_at)
    for slot in mar.get("slots", []):
        scheduled_for = parse_iso_to_utc(slot.get("scheduled_for"))
        if not scheduled_for or scheduled_for > cutoff:
            continue
        if slot.get("status") not in {"overdue", "missed"}:
            continue
        hours_late = round((end_at - scheduled_for).total_seconds() / 3600, 1)
        dose_item = {
            "medication_id": slot.get("medication_id"),
            "name": slot.get("name"),
            "dosage": slot.get("dosage"),
            "scheduled_for": slot.get("scheduled_for"),
            "hours_overdue": hours_late
        }
        overdue_list.append(dose_item)

        existing_alert = await db.medication_missed_alerts.find_one(
            {
                "user_id": owner_id,
                "medication_id": slot.get("medication_id"),
                "scheduled_for": slot.get("scheduled_for")
            },
            {"_id": 0}
        )
        if not existing_alert:
            alert_doc = {
                "id": f"medalert_{uuid.uuid4().hex[:12]}",
                "user_id": owner_id,
                "medication_id": slot.get("medication_id"),
                "scheduled_for": slot.get("scheduled_for"),
                "hours_overdue": hours_late,
                "created_at": datetime.now(timezone.utc).isoformat()
            }
            await db.medication_missed_alerts.insert_one(alert_doc)

            await create_safety_alert_with_escalation(
                user_id=owner_id,
                event_type="missed_medication_dose",
                severity="high",
                message=f"Missed medication dose: {slot.get('name')} ({slot.get('dosage')})",
                payload={
                    "medication_id": slot.get("medication_id"),
                    "name": slot.get("name"),
                    "dosage": slot.get("dosage"),
                    "scheduled_for": slot.get("scheduled_for"),
                    "hours_overdue": hours_late,
                    "medication_missed_alert_id": alert_doc["id"]
                },
                extra_fields={
                    "medication_id": slot.get("medication_id"),
                    "scheduled_for": slot.get("scheduled_for"),
                    "hours_overdue": hours_late
                }
            )

    overdue_list.sort(key=lambda x: x["scheduled_for"], reverse=True)
    # Keep response backward-compatible by returning list only.
    # Hook outcomes are stored in alert_notification_logs.
    return overdue_list[:30]

@api_router.get("/medications/interactions", response_model=dict)
async def medication_interaction_check(
    target_user_id: Optional[str] = None,
    current_user: User = Depends(get_current_user)
):
    owner_id = await resolve_target_user_id(current_user, target_user_id)
    meds = await db.medications.find({"user_id": owner_id, "active": True}, {"_id": 0}).to_list(300)
    names = [m["name"].strip().lower() for m in meds if m.get("name")]

    known_pairs = {
        frozenset({"donepezil", "diphenhydramine"}): "Donepezil may be less effective with strong anticholinergics like diphenhydramine.",
        frozenset({"memantine", "dextromethorphan"}): "Memantine and dextromethorphan may increase CNS side effects.",
        frozenset({"warfarin", "aspirin"}): "Warfarin with aspirin may increase bleeding risk.",
        frozenset({"metformin", "contrast dye"}): "Metformin may need temporary pause around iodinated contrast."
    }

    warnings = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            pair_key = frozenset({names[i], names[j]})
            if pair_key in known_pairs:
                warnings.append({
                    "medications": [names[i], names[j]],
                    "message": known_pairs[pair_key]
                })
    return {"warnings": warnings, "count": len(warnings)}

# ==================== MOOD & BPSD TRACKING ====================

async def compute_bpsd_analytics(owner_id: str, days: int = 30) -> dict:
    days = max(1, min(days, 180))
    end_at = datetime.now(timezone.utc)
    start_at = end_at - timedelta(days=days)
    start_iso = start_at.isoformat()

    observations = await db.bpsd_observations.find(
        {"user_id": owner_id, "observed_at": {"$gte": start_iso}},
        {"_id": 0}
    ).to_list(3000)
    mood_checkins = await db.mood_checkins.find(
        {"user_id": owner_id, "created_at": {"$gte": start_iso}},
        {"_id": 0}
    ).to_list(3000)
    med_missed = await db.medication_missed_alerts.find(
        {"user_id": owner_id, "created_at": {"$gte": start_iso}},
        {"_id": 0}
    ).to_list(3000)
    safety_alerts = await db.safety_alerts.find(
        {"user_id": owner_id, "triggered_at": {"$gte": start_iso}},
        {"_id": 0}
    ).to_list(3000)

    symptom_counts = {symptom: 0 for symptom in BPSD_SYMPTOM_TAXONOMY}
    time_of_day_counts = {slot: 0 for slot in BPSD_TIME_OF_DAY}
    weekday_counts = {day: 0 for day in ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]}
    symptom_on_low_mood = {symptom: 0 for symptom in BPSD_SYMPTOM_TAXONOMY}
    symptom_on_missed_dose = {symptom: 0 for symptom in BPSD_SYMPTOM_TAXONOMY}
    symptom_on_safety_alert = {symptom: 0 for symptom in BPSD_SYMPTOM_TAXONOMY}
    total_events = 0

    mood_by_day = {}
    for checkin in mood_checkins:
        created_dt = parse_iso_to_utc(checkin.get("created_at"))
        if not created_dt:
            continue
        dkey = created_dt.date().isoformat()
        mood_by_day.setdefault(dkey, []).append(clamp_score(checkin.get("mood_score"), 3))

    low_mood_days = set()
    for day_key, scores in mood_by_day.items():
        if scores and (sum(scores) / len(scores)) <= 2.0:
            low_mood_days.add(day_key)

    missed_dose_days = set()
    for item in med_missed:
        scheduled_dt = parse_iso_to_utc(item.get("scheduled_for")) or parse_iso_to_utc(item.get("created_at"))
        if scheduled_dt:
            missed_dose_days.add(scheduled_dt.date().isoformat())

    safety_days = set()
    for item in safety_alerts:
        triggered_dt = parse_iso_to_utc(item.get("triggered_at"))
        if triggered_dt:
            safety_days.add(triggered_dt.date().isoformat())

    for obs in observations:
        symptom = normalize_bpsd_symptom(obs.get("symptom"))
        if not symptom:
            continue
        observed_dt = parse_iso_to_utc(obs.get("observed_at")) or parse_iso_to_utc(obs.get("created_at"))
        if not observed_dt:
            continue
        day_key = observed_dt.date().isoformat()
        weekday = observed_dt.strftime("%A").lower()
        tod = normalize_bpsd_time_of_day(obs.get("time_of_day"), "evening")

        total_events += 1
        symptom_counts[symptom] = symptom_counts.get(symptom, 0) + 1
        time_of_day_counts[tod] = time_of_day_counts.get(tod, 0) + 1
        weekday_counts[weekday] = weekday_counts.get(weekday, 0) + 1
        if day_key in low_mood_days:
            symptom_on_low_mood[symptom] = symptom_on_low_mood.get(symptom, 0) + 1
        if day_key in missed_dose_days:
            symptom_on_missed_dose[symptom] = symptom_on_missed_dose.get(symptom, 0) + 1
        if day_key in safety_days:
            symptom_on_safety_alert[symptom] = symptom_on_safety_alert.get(symptom, 0) + 1

    mood_scores = [clamp_score(c.get("mood_score"), 3) for c in mood_checkins]
    average_mood = round(sum(mood_scores) / len(mood_scores), 2) if mood_scores else 0.0
    top_symptom = None
    if total_events > 0:
        top_symptom = max(symptom_counts.items(), key=lambda item: item[1])[0]
        if symptom_counts.get(top_symptom, 0) == 0:
            top_symptom = None

    dominant_tod = None
    if total_events > 0:
        dominant_tod = max(time_of_day_counts.items(), key=lambda item: item[1])[0]
        if time_of_day_counts.get(dominant_tod, 0) == 0:
            dominant_tod = None

    pattern_insights = []
    if top_symptom:
        pattern_insights.append(
            f"Most frequent symptom: {top_symptom.replace('_', ' ')} ({symptom_counts[top_symptom]} events in {days} days)."
        )
    if dominant_tod:
        pattern_insights.append(
            f"Symptoms are most common in the {dominant_tod} ({time_of_day_counts[dominant_tod]} events)."
        )
    if low_mood_days:
        pattern_insights.append(f"Low mood days detected: {len(low_mood_days)} day(s) in the last {days} days.")
    if top_symptom and symptom_on_missed_dose.get(top_symptom, 0) > 0:
        pattern_insights.append(
            f"{top_symptom.replace('_', ' ').capitalize()} co-occurred with missed-dose days {symptom_on_missed_dose[top_symptom]} time(s)."
        )

    return {
        "period_days": days,
        "total_observations": total_events,
        "average_mood_score": average_mood,
        "low_mood_days": len(low_mood_days),
        "top_symptom": top_symptom,
        "symptom_counts": symptom_counts,
        "time_of_day_counts": time_of_day_counts,
        "weekday_counts": weekday_counts,
        "correlations": {
            "symptom_on_low_mood_day": symptom_on_low_mood,
            "symptom_on_missed_dose_day": symptom_on_missed_dose,
            "symptom_on_safety_alert_day": symptom_on_safety_alert
        },
        "pattern_insights": pattern_insights[:6]
    }

@api_router.get("/bpsd/taxonomy", response_model=dict)
async def get_bpsd_taxonomy():
    return {
        "symptoms": BPSD_SYMPTOM_TAXONOMY,
        "time_of_day": BPSD_TIME_OF_DAY,
        "appetite_options": ["low", "normal", "high"]
    }

@api_router.get("/mood/checkins", response_model=List[dict])
async def get_mood_checkins(
    target_user_id: Optional[str] = None,
    days: int = 30,
    limit: int = 100,
    current_user: User = Depends(get_current_user)
):
    owner_id = await resolve_target_user_id(current_user, target_user_id)
    days = max(1, min(days, 180))
    safe_limit = max(1, min(limit, 500))
    start_iso = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
    docs = await db.mood_checkins.find(
        {"user_id": owner_id, "created_at": {"$gte": start_iso}},
        {"_id": 0}
    ).sort("created_at", -1).to_list(safe_limit)
    return docs

@api_router.post("/mood/checkins", response_model=dict)
async def create_mood_checkin(
    payload: MoodCheckinCreate,
    target_user_id: Optional[str] = None,
    current_user: User = Depends(get_current_user)
):
    owner_id = await resolve_target_user_id(
        current_user,
        target_user_id,
        require_write=True if target_user_id else False
    )
    source = (payload.source or current_user.role or "patient").strip().lower()
    if source not in {"patient", "caregiver", "clinician", "voice", "auto"}:
        source = "patient"
    mood_obj = MoodCheckin(
        user_id=owner_id,
        mood_score=clamp_score(payload.mood_score, 3),
        energy_score=clamp_score(payload.energy_score, 3),
        anxiety_score=clamp_score(payload.anxiety_score, 3),
        sleep_quality=clamp_score(payload.sleep_quality, 3),
        appetite=normalize_appetite(payload.appetite, "normal"),
        notes=(payload.notes or None),
        source=source,
        created_by_user_id=current_user.user_id
    )
    doc = mood_obj.model_dump()
    doc["created_at"] = doc["created_at"].isoformat()
    await db.mood_checkins.insert_one(doc)
    doc.pop("_id", None)
    return doc

@api_router.get("/bpsd/observations", response_model=List[dict])
async def get_bpsd_observations(
    target_user_id: Optional[str] = None,
    days: int = 30,
    limit: int = 200,
    symptom: Optional[str] = None,
    current_user: User = Depends(get_current_user)
):
    owner_id = await resolve_target_user_id(current_user, target_user_id)
    days = max(1, min(days, 180))
    safe_limit = max(1, min(limit, 1000))
    start_iso = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
    query = {"user_id": owner_id, "observed_at": {"$gte": start_iso}}
    if symptom:
        normalized_symptom = normalize_bpsd_symptom(symptom)
        if not normalized_symptom:
            raise HTTPException(status_code=400, detail="Invalid symptom")
        query["symptom"] = normalized_symptom
    docs = await db.bpsd_observations.find(query, {"_id": 0}).sort("observed_at", -1).to_list(safe_limit)
    return docs

@api_router.post("/bpsd/observations", response_model=dict)
async def create_bpsd_observation(
    payload: BPSDObservationCreate,
    target_user_id: Optional[str] = None,
    current_user: User = Depends(get_current_user)
):
    owner_id = await resolve_target_user_id(
        current_user,
        target_user_id,
        require_write=True if target_user_id else False
    )
    symptom = normalize_bpsd_symptom(payload.symptom)
    if not symptom:
        raise HTTPException(status_code=400, detail="Invalid symptom")

    observed_at = parse_iso_to_utc(payload.observed_at) if payload.observed_at else datetime.now(timezone.utc)
    obs = BPSDObservation(
        user_id=owner_id,
        symptom=symptom,
        severity=clamp_score(payload.severity, 3),
        time_of_day=normalize_bpsd_time_of_day(payload.time_of_day, "evening"),
        duration_minutes=max(1, min(int(payload.duration_minutes), 600)) if payload.duration_minutes else None,
        trigger_tags=[t.strip().lower() for t in (payload.trigger_tags or []) if t and t.strip()][:12],
        notes=(payload.notes or None),
        observed_by_role=(current_user.role or "caregiver"),
        observed_by_user_id=current_user.user_id,
        observed_at=observed_at
    )
    doc = obs.model_dump()
    doc["observed_at"] = doc["observed_at"].isoformat()
    await db.bpsd_observations.insert_one(doc)
    doc.pop("_id", None)
    return doc

@api_router.get("/bpsd/analytics", response_model=dict)
async def get_bpsd_analytics(
    target_user_id: Optional[str] = None,
    days: int = 30,
    current_user: User = Depends(get_current_user)
):
    owner_id = await resolve_target_user_id(current_user, target_user_id)
    return await compute_bpsd_analytics(owner_id, days)

# ==================== CAREGIVER PORTAL ====================

@api_router.post("/care/invites", response_model=dict)
async def create_care_invite(
    invite: CareInviteCreate,
    current_user: User = Depends(get_current_user)
):
    if current_user.role != "patient":
        raise HTTPException(status_code=403, detail="Only patients can create invite codes")

    role = invite.role if invite.role in {"caregiver", "clinician"} else "caregiver"
    permission = invite.permission if invite.permission in {"edit", "read_only"} else "edit"
    code = secrets.token_urlsafe(6)
    doc = {
        "id": f"invite_{uuid.uuid4().hex[:12]}",
        "code": code,
        "patient_id": current_user.user_id,
        "patient_name": current_user.name,
        "role": role,
        "permission": permission,
        "note": invite.note,
        "status": "pending",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "expires_at": (datetime.now(timezone.utc) + timedelta(days=max(1, invite.expires_in_days))).isoformat()
    }
    await db.care_invites.insert_one(doc)
    if "_id" in doc:
        del doc["_id"]
    return doc

@api_router.post("/care/invites/accept", response_model=dict)
async def accept_care_invite(
    payload: CareInviteAccept,
    current_user: User = Depends(get_current_user)
):
    invite = await db.care_invites.find_one({"code": payload.code, "status": "pending"}, {"_id": 0})
    if not invite:
        raise HTTPException(status_code=404, detail="Invite code not found or already used")
    if datetime.fromisoformat(invite["expires_at"]) < datetime.now(timezone.utc):
        raise HTTPException(status_code=400, detail="Invite code expired")
    if current_user.user_id == invite["patient_id"]:
        raise HTTPException(status_code=400, detail="You cannot accept your own invite")
    if current_user.role not in {"caregiver", "clinician"}:
        raise HTTPException(status_code=403, detail="Only caregiver or clinician accounts can accept invites")

    # Enforce patient limit per subscription tier
    current_count = await db.care_links.count_documents(
        {"caregiver_id": current_user.user_id, "status": "accepted"}
    )
    user_doc = await db.users.find_one({"user_id": current_user.user_id}, {"_id": 0, "subscription_tier": 1})
    tier = (user_doc or {}).get("subscription_tier", "free")
    limit = PATIENT_LIMITS.get(tier, 3)
    if current_count >= limit:
        raise HTTPException(
            status_code=403,
            detail=f"Patient limit reached ({limit}). Upgrade to add more patients."
        )

    existing = await db.care_links.find_one(
        {"patient_id": invite["patient_id"], "caregiver_id": current_user.user_id, "status": "accepted"},
        {"_id": 0}
    )
    if not existing:
        link_doc = {
            "id": f"carelink_{uuid.uuid4().hex[:12]}",
            "patient_id": invite["patient_id"],
            "caregiver_id": current_user.user_id,
            "caregiver_role": invite["role"],
            "permission": invite["permission"],
            "status": "accepted",
            "created_at": datetime.now(timezone.utc).isoformat()
        }
        await db.care_links.insert_one(link_doc)
    await db.care_invites.update_one({"id": invite["id"]}, {"$set": {"status": "accepted", "accepted_by": current_user.user_id}})
    await db.users.update_one(
        {"user_id": current_user.user_id},
        {"$addToSet": {"linked_patient_ids": invite["patient_id"]}}
    )
    return {"message": "Invite accepted", "patient_id": invite["patient_id"]}

@api_router.get("/care/links", response_model=dict)
async def get_care_links(current_user: User = Depends(get_current_user)):
    patient_links = await db.care_links.find(
        {"patient_id": current_user.user_id, "status": "accepted"},
        {"_id": 0}
    ).to_list(200)
    caregiver_links = await db.care_links.find(
        {"caregiver_id": current_user.user_id, "status": "accepted"},
        {"_id": 0}
    ).to_list(200)

    if caregiver_links:
        patient_ids = [l["patient_id"] for l in caregiver_links]
        patients = await db.users.find({"user_id": {"$in": patient_ids}}, {"_id": 0, "hashed_password": 0}).to_list(200)
        patient_map = {p["user_id"]: p for p in patients}
        for link in caregiver_links:
            link["patient"] = patient_map.get(link["patient_id"])

    if patient_links:
        caregiver_ids = [l["caregiver_id"] for l in patient_links]
        caregivers = await db.users.find({"user_id": {"$in": caregiver_ids}}, {"_id": 0, "hashed_password": 0}).to_list(200)
        caregiver_map = {c["user_id"]: c for c in caregivers}
        for link in patient_links:
            link["caregiver"] = caregiver_map.get(link["caregiver_id"])

    return {"as_patient": patient_links, "as_caregiver": caregiver_links}

@api_router.get("/external-bot/patients", response_model=List[dict])
async def get_external_bot_patients(current_user: User = Depends(get_current_user)):
    if not role_can_use_external_bot(current_user):
        raise HTTPException(status_code=403, detail="Role not allowed for external bot")
    return await get_doctor_accessible_patients(current_user, limit=120)

@api_router.post("/external-bot/link-codes", response_model=dict)
async def create_external_bot_link_code_route(
    payload: ExternalBotLinkCodeCreate,
    current_user: User = Depends(get_current_user)
):
    if not role_can_use_external_bot(current_user):
        raise HTTPException(status_code=403, detail="Role not allowed for external bot")
    require_premium(current_user)
    channel = normalize_external_bot_channel(payload.channel)
    if not channel:
        raise HTTPException(status_code=400, detail="channel must be telegram or whatsapp")

    patient_user_id = None
    if payload.patient_user_id:
        patient_user_id = await resolve_target_user_id(current_user, payload.patient_user_id)

    expires_minutes = max(5, min(int(payload.expires_in_minutes or 20), 240))
    now = datetime.now(timezone.utc)
    code = create_external_bot_link_code()
    # Retry a few times to avoid rare collisions.
    for _ in range(4):
        exists = await db.external_bot_link_codes.find_one({"code": code, "status": "pending"}, {"_id": 0, "id": 1})
        if not exists:
            break
        code = create_external_bot_link_code()

    doc = {
        "id": f"extcode_{uuid.uuid4().hex[:12]}",
        "code": code,
        "channel": channel,
        "doctor_user_id": current_user.user_id,
        "patient_user_id": patient_user_id,
        "status": "pending",
        "created_at": now.isoformat(),
        "expires_at": (now + timedelta(minutes=expires_minutes)).isoformat()
    }
    await db.external_bot_link_codes.insert_one(doc)

    if channel == "telegram":
        instructions = f"Open your Telegram bot and send: /link {code}"
    else:
        instructions = f"Send WhatsApp message: /link {code}"

    return {
        "id": doc["id"],
        "code": code,
        "channel": channel,
        "patient_user_id": patient_user_id,
        "expires_at": doc["expires_at"],
        "connect_instructions": instructions
    }

@api_router.get("/external-bot/links", response_model=List[dict])
async def list_external_bot_links(current_user: User = Depends(get_current_user)):
    if not role_can_use_external_bot(current_user):
        raise HTTPException(status_code=403, detail="Role not allowed for external bot")
    links = await db.external_bot_links.find(
        {"doctor_user_id": current_user.user_id},
        {"_id": 0}
    ).sort("updated_at", -1).to_list(200)
    if not links:
        return []
    patient_ids = [l.get("patient_user_id") for l in links if l.get("patient_user_id")]
    patient_map = {}
    if patient_ids:
        patients = await db.users.find(
            {"user_id": {"$in": patient_ids}},
            {"_id": 0, "user_id": 1, "name": 1, "email": 1}
        ).to_list(200)
        patient_map = {p["user_id"]: p for p in patients}
    for link in links:
        pid = link.get("patient_user_id")
        if pid:
            link["patient"] = patient_map.get(pid)
    return links

@api_router.put("/external-bot/links/{link_id}/patient", response_model=dict)
async def update_external_bot_link_patient(
    link_id: str,
    payload: ExternalBotLinkPatientUpdate,
    current_user: User = Depends(get_current_user)
):
    if not role_can_use_external_bot(current_user):
        raise HTTPException(status_code=403, detail="Role not allowed for external bot")
    link = await db.external_bot_links.find_one(
        {"id": link_id, "doctor_user_id": current_user.user_id, "active": True},
        {"_id": 0}
    )
    if not link:
        raise HTTPException(status_code=404, detail="External bot link not found")

    patient_user_id = None
    if payload.patient_user_id:
        patient_user_id = await resolve_target_user_id(current_user, payload.patient_user_id)
    await db.external_bot_links.update_one(
        {"id": link_id},
        {"$set": {"patient_user_id": patient_user_id, "updated_at": datetime.now(timezone.utc).isoformat()}}
    )
    updated = await db.external_bot_links.find_one({"id": link_id}, {"_id": 0})
    return updated

@api_router.delete("/external-bot/links/{link_id}", response_model=dict)
async def revoke_external_bot_link(
    link_id: str,
    current_user: User = Depends(get_current_user)
):
    if not role_can_use_external_bot(current_user):
        raise HTTPException(status_code=403, detail="Role not allowed for external bot")
    result = await db.external_bot_links.update_one(
        {"id": link_id, "doctor_user_id": current_user.user_id, "active": True},
        {"$set": {"active": False, "updated_at": datetime.now(timezone.utc).isoformat()}}
    )
    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="External bot link not found")
    return {"message": "External bot link revoked"}

@api_router.post("/external-bot/query", response_model=dict)
async def external_bot_query(
    payload: DoctorBotQueryRequest,
    current_user: User = Depends(get_current_user)
):
    if not role_can_use_external_bot(current_user):
        raise HTTPException(status_code=403, detail="Role not allowed for external bot")
    question = (payload.text or "").strip()
    if not question:
        raise HTTPException(status_code=400, detail="text is required")

    patient_user_id = payload.patient_user_id
    if patient_user_id:
        patient_user_id = await resolve_target_user_id(current_user, patient_user_id)
    else:
        patients = await get_doctor_accessible_patients(current_user, limit=5)
        if len(patients) == 1:
            patient_user_id = patients[0]["user_id"]
        else:
            raise HTTPException(
                status_code=400,
                detail="Multiple or zero patients available. Provide patient_user_id."
            )

    response_text, intent, snapshot = await handle_external_doctor_query(
        doctor_user=current_user,
        question=question,
        patient_user_id=patient_user_id
    )
    voice_base64 = None
    if payload.prefer_voice:
        audio = await generate_tts_audio_bytes(response_text)
        if audio:
            voice_base64 = base64.b64encode(audio).decode("utf-8")
    return {
        "patient_user_id": patient_user_id,
        "intent": intent,
        "response": response_text,
        "voice_base64": voice_base64,
        "snapshot": {
            "patient": snapshot.get("patient"),
            "timestamp": snapshot.get("timestamp"),
            "adherence_7d": snapshot.get("adherence_7d"),
            "open_safety_alert_count": len(snapshot.get("open_safety_alerts") or [])
        }
    }

@api_router.get("/care/patients/{patient_id}/dashboard", response_model=dict)
async def get_caregiver_patient_dashboard(
    patient_id: str,
    current_user: User = Depends(get_current_user)
):
    owner_id = await resolve_target_user_id(current_user, patient_id)
    patient = await db.users.find_one({"user_id": owner_id}, {"_id": 0, "hashed_password": 0})
    reminders = await db.reminders.find({"user_id": owner_id}, {"_id": 0}).to_list(300)
    meds = await db.medications.find({"user_id": owner_id, "active": True}, {"_id": 0}).to_list(300)
    notes = await db.daily_notes.find({"user_id": owner_id}, {"_id": 0}).sort("date", -1).to_list(7)
    alerts = await db.safety_alerts.find({"user_id": owner_id, "acknowledged": False}, {"_id": 0}).sort("triggered_at", -1).to_list(50)
    fall_events = await db.safety_fall_events.find({"user_id": owner_id}, {"_id": 0}).sort("detected_at", -1).to_list(20)
    emergency_contacts = await db.emergency_contacts.find({"user_id": owner_id}, {"_id": 0}).sort([("is_primary", -1), ("created_at", 1)]).to_list(20)
    instructions_raw = await db.care_instructions.find({"user_id": owner_id, "active": True}, {"_id": 0, "search_text": 0}).sort("updated_at", -1).to_list(100)
    ref_date = datetime.now(timezone.utc).date()
    instructions = [i for i in instructions_raw if instruction_allowed_for_patient_use(i, ref_date)]
    adherence = await medication_adherence_summary(7, owner_id, current_user)
    missed = await medication_missed_doses(2, owner_id, current_user)
    bpsd_analytics = await compute_bpsd_analytics(owner_id, 30)
    mood_checkins = await db.mood_checkins.find({"user_id": owner_id}, {"_id": 0}).sort("created_at", -1).to_list(20)
    bpsd_observations = await db.bpsd_observations.find({"user_id": owner_id}, {"_id": 0}).sort("observed_at", -1).to_list(20)

    completed_reminders = len([r for r in reminders if r.get("completed")])
    reminder_completion = round((completed_reminders / len(reminders)) * 100, 1) if reminders else 0.0

    return {
        "patient": patient,
        "summary": {
            "reminders_total": len(reminders),
            "reminders_completed": completed_reminders,
            "reminder_completion_percent": reminder_completion,
            "medications_active": len(meds),
            "unacknowledged_safety_alerts": len(alerts),
            "emergency_contacts_count": len(emergency_contacts),
            "care_instructions_active": len(instructions),
            "fall_events_recent_count": len(fall_events),
            "adherence_percent_last_7_days": adherence.get("adherence_percent", 0.0),
            "adherence_total_percent_last_7_days": adherence.get("adherence_percent_total", 0.0),
            "bpsd_events_last_30_days": bpsd_analytics.get("total_observations", 0),
            "low_mood_days_last_30_days": bpsd_analytics.get("low_mood_days", 0),
            "top_bpsd_symptom": bpsd_analytics.get("top_symptom")
        },
        "daily_notes_recent": notes,
        "medication_missed_recent": missed,
        "safety_alerts_open": alerts,
        "fall_events_recent": fall_events,
        "emergency_contacts": emergency_contacts,
        "care_instructions_recent": instructions,
        "mood_checkins_recent": mood_checkins,
        "bpsd_observations_recent": bpsd_observations,
        "bpsd_analytics_30d": bpsd_analytics
    }

@api_router.post("/care/patients/{patient_id}/reminders", response_model=dict)
async def caregiver_create_reminder(
    patient_id: str,
    reminder: CaregiverReminderCreate,
    current_user: User = Depends(get_current_user)
):
    owner_id = await resolve_target_user_id(current_user, patient_id, require_write=True)
    reminder_obj = Reminder(user_id=owner_id, **reminder.model_dump())
    doc = reminder_obj.model_dump()
    doc["created_at"] = doc["created_at"].isoformat()
    await db.reminders.insert_one(doc)
    if "_id" in doc:
        del doc["_id"]
    return doc

@api_router.post("/care/patients/{patient_id}/family", response_model=dict)
async def caregiver_create_family_member(
    patient_id: str,
    member: FamilyMemberCreate,
    current_user: User = Depends(get_current_user)
):
    owner_id = await resolve_target_user_id(current_user, patient_id, require_write=True)
    member_obj = FamilyMember(user_id=owner_id, **member.model_dump())
    doc = member_obj.model_dump()
    doc["created_at"] = doc["created_at"].isoformat()
    doc["updated_at"] = doc["updated_at"].isoformat()
    doc["search_text"] = f"{member.name} {member.relationship} {member.relationship_label} {member.notes or ''}".lower()
    await db.family_members.insert_one(doc)
    doc.pop("_id", None)
    doc.pop("search_text", None)
    return doc

@api_router.put("/care/patients/{patient_id}/family/{member_id}", response_model=dict)
async def caregiver_update_family_member(
    patient_id: str,
    member_id: str,
    member: FamilyMemberUpdate,
    current_user: User = Depends(get_current_user)
):
    owner_id = await resolve_target_user_id(current_user, patient_id, require_write=True)
    update_data = {k: v for k, v in member.model_dump().items() if v is not None}
    update_data["updated_at"] = datetime.now(timezone.utc).isoformat()
    await db.family_members.update_one(
        {"id": member_id, "user_id": owner_id},
        {"$set": update_data}
    )
    updated = await db.family_members.find_one({"id": member_id, "user_id": owner_id}, {"_id": 0})
    if not updated:
        raise HTTPException(status_code=404, detail="Family member not found")
    return updated

@api_router.post("/care/patients/{patient_id}/share-readonly", response_model=dict)
async def create_readonly_share_link(
    patient_id: str,
    payload: ShareLinkCreate,
    current_user: User = Depends(get_current_user)
):
    owner_id = await resolve_target_user_id(current_user, patient_id)
    # allow patient or any accepted caregiver to create a read-only share
    token = secrets.token_urlsafe(18)
    share_doc = {
        "id": f"share_{uuid.uuid4().hex[:12]}",
        "token": token,
        "patient_id": owner_id,
        "created_by": current_user.user_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "expires_at": (datetime.now(timezone.utc) + timedelta(days=max(1, payload.expires_in_days))).isoformat(),
        "active": True
    }
    await db.care_share_links.insert_one(share_doc)
    return {"share_token": token, "share_path": f"/api/care/share/{token}"}

@api_router.get("/care/share/{token}", response_model=dict)
async def read_shared_patient_dashboard(token: str):
    share = await db.care_share_links.find_one({"token": token, "active": True}, {"_id": 0})
    if not share:
        raise HTTPException(status_code=404, detail="Share link not found")
    if datetime.fromisoformat(share["expires_at"]) < datetime.now(timezone.utc):
        raise HTTPException(status_code=410, detail="Share link expired")

    patient = await db.users.find_one({"user_id": share["patient_id"]}, {"_id": 0, "hashed_password": 0})
    reminders = await db.reminders.find({"user_id": share["patient_id"]}, {"_id": 0}).to_list(300)
    meds = await db.medications.find({"user_id": share["patient_id"], "active": True}, {"_id": 0}).to_list(300)
    notes = await db.daily_notes.find({"user_id": share["patient_id"]}, {"_id": 0}).sort("date", -1).to_list(7)
    alerts = await db.safety_alerts.find({"user_id": share["patient_id"], "acknowledged": False}, {"_id": 0}).sort("triggered_at", -1).to_list(50)
    fall_events = await db.safety_fall_events.find({"user_id": share["patient_id"]}, {"_id": 0}).sort("detected_at", -1).to_list(20)
    instructions_raw = await db.care_instructions.find({"user_id": share["patient_id"], "active": True}, {"_id": 0, "search_text": 0}).sort("updated_at", -1).to_list(100)
    ref_date = datetime.now(timezone.utc).date()
    instructions = [i for i in instructions_raw if instruction_allowed_for_patient_use(i, ref_date)]
    bpsd_analytics = await compute_bpsd_analytics(share["patient_id"], 30)
    mood_checkins = await db.mood_checkins.find({"user_id": share["patient_id"]}, {"_id": 0}).sort("created_at", -1).to_list(10)
    bpsd_observations = await db.bpsd_observations.find({"user_id": share["patient_id"]}, {"_id": 0}).sort("observed_at", -1).to_list(10)
    return {
        "patient": patient,
        "summary": {
            "reminders_total": len(reminders),
            "medications_active": len(meds),
            "open_safety_alerts": len(alerts),
            "fall_events_recent_count": len(fall_events),
            "care_instructions_active": len(instructions),
            "bpsd_events_last_30_days": bpsd_analytics.get("total_observations", 0),
            "low_mood_days_last_30_days": bpsd_analytics.get("low_mood_days", 0),
            "top_bpsd_symptom": bpsd_analytics.get("top_symptom")
        },
        "daily_notes_recent": notes,
        "safety_alerts_open": alerts,
        "fall_events_recent": fall_events,
        "care_instructions_recent": instructions,
        "mood_checkins_recent": mood_checkins,
        "bpsd_observations_recent": bpsd_observations,
        "bpsd_analytics_30d": bpsd_analytics
    }

@api_router.get("/care/instructions", response_model=List[dict])
async def get_care_instructions(
    target_user_id: Optional[str] = None,
    only_active: bool = True,
    frequency: Optional[str] = None,
    policy_type: Optional[str] = None,
    effective_on: Optional[str] = None,
    include_drafts: bool = False,
    current_user: User = Depends(get_current_user)
):
    owner_id = await resolve_target_user_id(current_user, target_user_id)
    query = {"user_id": owner_id}
    if only_active:
        query["active"] = True
    if frequency:
        normalized_frequency = normalize_frequency(frequency, fallback="")
        if normalized_frequency:
            query["frequency"] = normalized_frequency
    if policy_type:
        query["policy_type"] = normalize_policy_type(policy_type, fallback="")
    instructions = await db.care_instructions.find(
        query,
        {"_id": 0, "search_text": 0}
    ).sort("updated_at", -1).to_list(300)

    ref_date = parse_yyyy_mm_dd(effective_on) if effective_on else datetime.now(timezone.utc).date()
    if effective_on and ref_date is None:
        raise HTTPException(status_code=400, detail="effective_on must use YYYY-MM-DD")

    if not include_drafts:
        instructions = [d for d in instructions if instruction_allowed_for_patient_use(d, ref_date)]
    return instructions

@api_router.get("/care/instructions/active-regimen", response_model=List[dict])
async def get_active_medication_regimen(
    target_user_id: Optional[str] = None,
    on_date: Optional[str] = None,
    current_user: User = Depends(get_current_user)
):
    owner_id = await resolve_target_user_id(current_user, target_user_id)
    ref_date = parse_yyyy_mm_dd(on_date) if on_date else datetime.now(timezone.utc).date()
    if on_date and ref_date is None:
        raise HTTPException(status_code=400, detail="on_date must use YYYY-MM-DD")
    docs = await db.care_instructions.find(
        {"user_id": owner_id, "policy_type": "medication", "active": True},
        {"_id": 0, "search_text": 0}
    ).to_list(300)
    docs = [d for d in docs if instruction_allowed_for_patient_use(d, ref_date)]
    latest = select_latest_medication_regimens(docs)
    latest.sort(key=lambda x: (x.get("regimen_key") or x.get("title", ""), -int(x.get("version", 1))))
    return latest

@api_router.get("/care/today-plan", response_model=dict)
async def get_today_care_plan(
    target_user_id: Optional[str] = None,
    on_date: Optional[str] = None,
    current_user: User = Depends(get_current_user)
):
    owner_id = await resolve_target_user_id(current_user, target_user_id)
    ref_date = parse_yyyy_mm_dd(on_date) if on_date else datetime.now(timezone.utc).date()
    if on_date and ref_date is None:
        raise HTTPException(status_code=400, detail="on_date must use YYYY-MM-DD")
    return await build_patient_today_plan(owner_id, ref_date)

@api_router.post("/care/instructions", response_model=dict)
async def create_care_instruction(
    payload: CareInstructionCreate,
    target_user_id: Optional[str] = None,
    current_user: User = Depends(get_current_user)
):
    owner_id = await resolve_target_user_id(
        current_user,
        target_user_id,
        require_write=True if target_user_id else False
    )
    normalized_policy = normalize_policy_type(payload.policy_type)
    regimen_key = (payload.regimen_key or "").strip().lower() or (
        re.sub(r"\s+", "_", payload.title.strip().lower()) if normalized_policy == "medication" else None
    )
    effective_start = normalize_yyyy_mm_dd(payload.effective_start_date)
    effective_end = normalize_yyyy_mm_dd(payload.effective_end_date)
    if payload.effective_start_date and not effective_start:
        raise HTTPException(status_code=400, detail="effective_start_date must use YYYY-MM-DD")
    if payload.effective_end_date and not effective_end:
        raise HTTPException(status_code=400, detail="effective_end_date must use YYYY-MM-DD")
    if effective_start and effective_end and effective_end < effective_start:
        raise HTTPException(status_code=400, detail="effective_end_date cannot be before effective_start_date")

    signoff_required = payload.signoff_required if payload.signoff_required is not None else (normalized_policy == "medication")
    signoff_status = normalize_signoff_status(None, signoff_required)

    version_query = {"user_id": owner_id, "policy_type": normalized_policy}
    if regimen_key:
        version_query["regimen_key"] = regimen_key
    latest = await db.care_instructions.find_one(version_query, {"_id": 0}, sort=[("version", -1), ("updated_at", -1)])
    next_version = int(latest.get("version", 0) + 1) if latest else 1

    instruction_obj = CareInstruction(
        user_id=owner_id,
        title=payload.title.strip(),
        instruction_text=payload.instruction_text.strip(),
        summary=payload.summary.strip() if payload.summary else None,
        frequency=normalize_frequency(payload.frequency),
        day_of_week=normalize_day_of_week(payload.day_of_week),
        time_of_day=payload.time_of_day.strip() if payload.time_of_day else None,
        tags=[t.strip().lower() for t in (payload.tags or []) if t and t.strip()],
        policy_type=normalized_policy,
        regimen_key=regimen_key,
        version=next_version,
        status="draft" if signoff_required else "active",
        effective_start_date=effective_start,
        effective_end_date=effective_end,
        signoff_required=signoff_required,
        signoff_status=signoff_status,
        supersedes_instruction_id=latest.get("id") if latest else None,
        active=bool(payload.active) if not signoff_required else False,
        uploaded_by_user_id=current_user.user_id,
        uploaded_by_role=current_user.role
    )
    doc = instruction_obj.model_dump()
    doc["created_at"] = doc["created_at"].isoformat()
    doc["updated_at"] = doc["updated_at"].isoformat()
    doc["search_text"] = build_instruction_search_text(doc)
    await db.care_instructions.insert_one(doc)
    await upsert_instruction_chunks(doc)
    if owner_id != current_user.user_id:
        asyncio.create_task(notify_patient_update(
            owner_id, "instruction_added",
            f"{current_user.name} added a care instruction: {payload.title.strip()}. Your voice assistant has the details."
        ))

    if doc["policy_type"] == "medication" and doc.get("status") == "active" and doc.get("regimen_key"):
        # Keep only one active medication regimen version at a time for a regimen_key.
        await db.care_instructions.update_many(
            {
                "user_id": owner_id,
                "policy_type": "medication",
                "regimen_key": doc["regimen_key"],
                "id": {"$ne": doc["id"]},
                "status": "active",
                "active": True
            },
            {
                "$set": {
                    "status": "archived",
                    "active": False,
                    "updated_at": datetime.now(timezone.utc).isoformat()
                }
            }
        )

    doc.pop("_id", None)
    doc.pop("search_text", None)
    return doc

@api_router.post("/care/instructions/upload", response_model=dict)
async def create_care_instruction_with_upload(
    file: UploadFile = File(...),
    title: str = Form(...),
    instruction_text: Optional[str] = Form(None),
    summary: Optional[str] = Form(None),
    frequency: str = Form("daily"),
    day_of_week: Optional[str] = Form(None),
    time_of_day: Optional[str] = Form(None),
    tags_csv: Optional[str] = Form(None),
    policy_type: str = Form("general"),
    regimen_key: Optional[str] = Form(None),
    effective_start_date: Optional[str] = Form(None),
    effective_end_date: Optional[str] = Form(None),
    signoff_required: Optional[bool] = Form(None),
    target_user_id: Optional[str] = Form(None),
    current_user: User = Depends(get_current_user)
):
    owner_id = await resolve_target_user_id(
        current_user,
        target_user_id,
        require_write=True if target_user_id else False
    )

    content = await file.read()
    ext = Path(file.filename).suffix if file.filename else ".bin"
    filename = f"{owner_id}_{uuid.uuid4().hex[:8]}{ext}"
    content_type = file.content_type or "application/octet-stream"

    await fs_bucket.upload_from_stream(
        filename,
        io.BytesIO(content),
        metadata={
            "user_id": owner_id,
            "content_type": content_type,
            "original_filename": file.filename,
            "uploaded_at": datetime.now(timezone.utc).isoformat()
        }
    )
    file_url = f"/api/files/{filename}"

    extracted_text = extract_instruction_text_from_upload(content, content_type, file.filename)
    combined_text = "\n\n".join(
        [p.strip() for p in [instruction_text or "", extracted_text] if p and p.strip()]
    ).strip()
    if not combined_text:
        raise HTTPException(
            status_code=400,
            detail="Could not extract text from this file. Provide instruction_text or upload txt/md/csv/json/pdf/docx."
        )

    normalized_policy = normalize_policy_type(policy_type)
    normalized_regimen_key = (regimen_key or "").strip().lower() or (
        re.sub(r"\s+", "_", title.strip().lower()) if normalized_policy == "medication" else None
    )
    effective_start = normalize_yyyy_mm_dd(effective_start_date)
    effective_end = normalize_yyyy_mm_dd(effective_end_date)
    if effective_start_date and not effective_start:
        raise HTTPException(status_code=400, detail="effective_start_date must use YYYY-MM-DD")
    if effective_end_date and not effective_end:
        raise HTTPException(status_code=400, detail="effective_end_date must use YYYY-MM-DD")
    if effective_start and effective_end and effective_end < effective_start:
        raise HTTPException(status_code=400, detail="effective_end_date cannot be before effective_start_date")

    requires_signoff = signoff_required if signoff_required is not None else (normalized_policy == "medication")
    signoff_state = normalize_signoff_status(None, requires_signoff)

    version_query = {"user_id": owner_id, "policy_type": normalized_policy}
    if normalized_regimen_key:
        version_query["regimen_key"] = normalized_regimen_key
    latest = await db.care_instructions.find_one(version_query, {"_id": 0}, sort=[("version", -1), ("updated_at", -1)])
    next_version = int(latest.get("version", 0) + 1) if latest else 1

    tags = [t.strip().lower() for t in (tags_csv or "").split(",") if t.strip()]
    instruction_obj = CareInstruction(
        user_id=owner_id,
        title=title.strip(),
        instruction_text=combined_text,
        summary=summary.strip() if summary else None,
        frequency=normalize_frequency(frequency),
        day_of_week=normalize_day_of_week(day_of_week),
        time_of_day=time_of_day.strip() if time_of_day else None,
        tags=tags,
        policy_type=normalized_policy,
        regimen_key=normalized_regimen_key,
        version=next_version,
        status="draft" if requires_signoff else "active",
        effective_start_date=effective_start,
        effective_end_date=effective_end,
        signoff_required=requires_signoff,
        signoff_status=signoff_state,
        supersedes_instruction_id=latest.get("id") if latest else None,
        active=False if requires_signoff else True,
        source_type="file",
        source_filename=file.filename,
        source_file_url=file_url,
        uploaded_by_user_id=current_user.user_id,
        uploaded_by_role=current_user.role
    )
    doc = instruction_obj.model_dump()
    doc["created_at"] = doc["created_at"].isoformat()
    doc["updated_at"] = doc["updated_at"].isoformat()
    doc["search_text"] = build_instruction_search_text(doc)
    await db.care_instructions.insert_one(doc)
    await upsert_instruction_chunks(doc)
    if owner_id != current_user.user_id:
        asyncio.create_task(notify_patient_update(
            owner_id, "instruction_added",
            f"{current_user.name} added a care instruction: {title.strip()}. Your voice assistant has the details."
        ))
    doc.pop("_id", None)
    doc.pop("search_text", None)
    return doc

@api_router.put("/care/instructions/{instruction_id}", response_model=dict)
async def update_care_instruction(
    instruction_id: str,
    payload: CareInstructionUpdate,
    create_new_version: bool = False,
    target_user_id: Optional[str] = None,
    current_user: User = Depends(get_current_user)
):
    owner_id = await resolve_target_user_id(
        current_user,
        target_user_id,
        require_write=True if target_user_id else False
    )
    existing = await db.care_instructions.find_one({"id": instruction_id, "user_id": owner_id}, {"_id": 0})
    if not existing:
        raise HTTPException(status_code=404, detail="Instruction not found")

    update_data = {k: v for k, v in payload.model_dump().items() if v is not None}
    if "frequency" in update_data:
        update_data["frequency"] = normalize_frequency(update_data["frequency"], fallback=existing.get("frequency", "daily"))
    if "day_of_week" in update_data:
        update_data["day_of_week"] = normalize_day_of_week(update_data["day_of_week"])
    if "policy_type" in update_data:
        update_data["policy_type"] = normalize_policy_type(update_data["policy_type"], fallback=existing.get("policy_type", "general"))
    if "effective_start_date" in update_data:
        normalized = normalize_yyyy_mm_dd(update_data["effective_start_date"])
        if update_data["effective_start_date"] and not normalized:
            raise HTTPException(status_code=400, detail="effective_start_date must use YYYY-MM-DD")
        update_data["effective_start_date"] = normalized
    if "effective_end_date" in update_data:
        normalized = normalize_yyyy_mm_dd(update_data["effective_end_date"])
        if update_data["effective_end_date"] and not normalized:
            raise HTTPException(status_code=400, detail="effective_end_date must use YYYY-MM-DD")
        update_data["effective_end_date"] = normalized
    if "title" in update_data and isinstance(update_data["title"], str):
        update_data["title"] = update_data["title"].strip()
    if "summary" in update_data and isinstance(update_data["summary"], str):
        update_data["summary"] = update_data["summary"].strip()
    if "instruction_text" in update_data and isinstance(update_data["instruction_text"], str):
        update_data["instruction_text"] = update_data["instruction_text"].strip()
    if "time_of_day" in update_data and isinstance(update_data["time_of_day"], str):
        update_data["time_of_day"] = update_data["time_of_day"].strip()
    if "regimen_key" in update_data and isinstance(update_data["regimen_key"], str):
        update_data["regimen_key"] = update_data["regimen_key"].strip().lower() or None
    if "tags" in update_data and isinstance(update_data["tags"], list):
        update_data["tags"] = [t.strip().lower() for t in update_data["tags"] if isinstance(t, str) and t.strip()]
    if "status" in update_data and update_data["status"] not in {"draft", "active", "archived", "rejected"}:
        raise HTTPException(status_code=400, detail="Invalid status")

    merged = {**existing, **update_data}
    start_d = parse_yyyy_mm_dd(merged.get("effective_start_date"))
    end_d = parse_yyyy_mm_dd(merged.get("effective_end_date"))
    if start_d and end_d and end_d < start_d:
        raise HTTPException(status_code=400, detail="effective_end_date cannot be before effective_start_date")

    now_iso = datetime.now(timezone.utc).isoformat()
    if create_new_version:
        new_doc = {**existing, **update_data}
        new_doc["id"] = f"instruction_{uuid.uuid4().hex[:12]}"
        new_doc["version"] = int(existing.get("version", 1)) + 1
        new_doc["supersedes_instruction_id"] = existing["id"]
        new_doc["created_at"] = now_iso
        new_doc["updated_at"] = now_iso
        signoff_required = bool(new_doc.get("signoff_required", new_doc.get("policy_type") == "medication"))
        new_doc["signoff_required"] = signoff_required
        new_doc["signoff_status"] = normalize_signoff_status(None, signoff_required)
        new_doc["status"] = "draft" if signoff_required else "active"
        new_doc["active"] = False if signoff_required else bool(new_doc.get("active", True))
        new_doc["signed_off_by_name"] = None
        new_doc["signed_off_by_user_id"] = None
        new_doc["signed_off_at"] = None
        new_doc["signed_off_notes"] = None
        new_doc["search_text"] = build_instruction_search_text(new_doc)
        await db.care_instructions.insert_one(new_doc)
        await upsert_instruction_chunks(new_doc)
        new_doc.pop("_id", None)
        new_doc.pop("search_text", None)
        return new_doc

    content_fields = {
        "title", "instruction_text", "summary", "frequency", "day_of_week",
        "time_of_day", "tags", "effective_start_date", "effective_end_date"
    }
    if any(field in update_data for field in content_fields) and bool(merged.get("signoff_required", False)):
        update_data["signoff_status"] = "pending"
        update_data["status"] = "draft"
        update_data["active"] = False
        update_data["signed_off_by_name"] = None
        update_data["signed_off_by_user_id"] = None
        update_data["signed_off_at"] = None
        update_data["signed_off_notes"] = None

    update_data["search_text"] = build_instruction_search_text(merged)
    update_data["updated_at"] = now_iso
    await db.care_instructions.update_one(
        {"id": instruction_id, "user_id": owner_id},
        {"$set": update_data}
    )
    updated = await db.care_instructions.find_one({"id": instruction_id, "user_id": owner_id}, {"_id": 0})
    if updated:
        await upsert_instruction_chunks(updated)
        updated.pop("search_text", None)
    return updated

@api_router.post("/care/instructions/{instruction_id}/signoff", response_model=dict)
async def signoff_care_instruction(
    instruction_id: str,
    payload: CareInstructionSignoffRequest,
    target_user_id: Optional[str] = None,
    current_user: User = Depends(get_current_user)
):
    if current_user.role not in {"clinician", "admin"}:
        raise HTTPException(status_code=403, detail="Only approved clinician or admin can sign off instructions")

    instruction = await db.care_instructions.find_one({"id": instruction_id}, {"_id": 0})
    if not instruction:
        raise HTTPException(status_code=404, detail="Instruction not found")

    owner_id = instruction["user_id"]
    if target_user_id and target_user_id != owner_id:
        raise HTTPException(status_code=400, detail="target_user_id does not match instruction owner")
    await resolve_target_user_id(
        current_user,
        target_user_id or owner_id,
        require_write=True if (target_user_id or owner_id) != current_user.user_id else False
    )

    effective_start = normalize_yyyy_mm_dd(payload.effective_start_date) if payload.effective_start_date else instruction.get("effective_start_date")
    effective_end = normalize_yyyy_mm_dd(payload.effective_end_date) if payload.effective_end_date else instruction.get("effective_end_date")
    if payload.effective_start_date and not effective_start:
        raise HTTPException(status_code=400, detail="effective_start_date must use YYYY-MM-DD")
    if payload.effective_end_date and not effective_end:
        raise HTTPException(status_code=400, detail="effective_end_date must use YYYY-MM-DD")
    if effective_start and effective_end and effective_end < effective_start:
        raise HTTPException(status_code=400, detail="effective_end_date cannot be before effective_start_date")

    now_iso = datetime.now(timezone.utc).isoformat()
    if payload.approved:
        update_data = {
            "signoff_status": "signed_off",
            "status": "active",
            "active": True,
            "effective_start_date": effective_start,
            "effective_end_date": effective_end,
            "signed_off_by_name": (payload.signed_by_name or current_user.name).strip(),
            "signed_off_by_user_id": current_user.user_id,
            "signed_off_at": now_iso,
            "signed_off_notes": payload.notes,
            "updated_at": now_iso
        }
        await db.care_instructions.update_one({"id": instruction_id, "user_id": owner_id}, {"$set": update_data})

        # Ensure a single active medication regimen per regimen_key.
        regimen_key = instruction.get("regimen_key")
        if instruction.get("policy_type") == "medication" and regimen_key:
            others = await db.care_instructions.find(
                {
                    "user_id": owner_id,
                    "policy_type": "medication",
                    "regimen_key": regimen_key,
                    "id": {"$ne": instruction_id},
                    "status": "active",
                    "active": True
                },
                {"_id": 0, "id": 1, "effective_end_date": 1}
            ).to_list(200)
            for other in others:
                patch = {
                    "status": "archived",
                    "active": False,
                    "updated_at": now_iso
                }
                if not other.get("effective_end_date"):
                    patch["effective_end_date"] = datetime.now(timezone.utc).date().isoformat()
                await db.care_instructions.update_one(
                    {"id": other["id"], "user_id": owner_id},
                    {"$set": patch}
                )
    else:
        update_data = {
            "signoff_status": "rejected",
            "status": "rejected",
            "active": False,
            "signed_off_by_name": (payload.signed_by_name or current_user.name).strip(),
            "signed_off_by_user_id": current_user.user_id,
            "signed_off_at": now_iso,
            "signed_off_notes": payload.notes,
            "updated_at": now_iso
        }
        await db.care_instructions.update_one({"id": instruction_id, "user_id": owner_id}, {"$set": update_data})

    updated = await db.care_instructions.find_one({"id": instruction_id, "user_id": owner_id}, {"_id": 0, "search_text": 0})
    return updated

@api_router.delete("/care/instructions/{instruction_id}")
async def delete_care_instruction(
    instruction_id: str,
    target_user_id: Optional[str] = None,
    current_user: User = Depends(get_current_user)
):
    owner_id = await resolve_target_user_id(
        current_user,
        target_user_id,
        require_write=True if target_user_id else False
    )
    result = await db.care_instructions.delete_one({"id": instruction_id, "user_id": owner_id})
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Instruction not found")
    await db.care_instruction_chunks.delete_many({"instruction_id": instruction_id, "user_id": owner_id})
    return {"message": "Instruction deleted"}

# ==================== DAILY NOTES ====================

class DailyNote(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str = Field(default_factory=lambda: f"note_{uuid.uuid4().hex[:12]}")
    user_id: str
    date: str  # YYYY-MM-DD format
    note: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class DailyNoteCreate(BaseModel):
    date: str
    note: str

@api_router.get("/daily-notes", response_model=List[dict])
async def get_daily_notes(current_user: User = Depends(get_current_user)):
    """Get all daily notes for the current user (last 30 days)"""
    notes = await db.daily_notes.find(
        {"user_id": current_user.user_id},
        {"_id": 0}
    ).sort("date", -1).to_list(30)
    return notes

@api_router.post("/daily-notes", response_model=dict)
async def create_daily_note(
    note_data: DailyNoteCreate,
    current_user: User = Depends(get_current_user)
):
    """Create or update a daily note"""
    # Check if note already exists for this date
    existing = await db.daily_notes.find_one(
        {"user_id": current_user.user_id, "date": note_data.date},
        {"_id": 0}
    )
    
    if existing:
        # Append to existing note
        updated_note = existing['note'] + '\n\n' + note_data.note
        await db.daily_notes.update_one(
            {"id": existing['id']},
            {"$set": {"note": updated_note, "updated_at": datetime.now(timezone.utc).isoformat()}}
        )
        existing['note'] = updated_note
        return existing
    else:
        # Create new note
        note_obj = DailyNote(
            user_id=current_user.user_id,
            date=note_data.date,
            note=note_data.note
        )
        doc = note_obj.model_dump()
        doc['created_at'] = doc['created_at'].isoformat()
        
        await db.daily_notes.insert_one(doc)
        
        if '_id' in doc:
            del doc['_id']
        return doc

@api_router.get("/daily-notes/{date}", response_model=dict)
async def get_daily_note(
    date: str,
    current_user: User = Depends(get_current_user)
):
    """Get daily note for a specific date"""
    note = await db.daily_notes.find_one(
        {"user_id": current_user.user_id, "date": date},
        {"_id": 0}
    )
    if not note:
        raise HTTPException(status_code=404, detail="Note not found")
    return note

# ==================== DAILY DIGEST & REPORTS ====================

@api_router.get("/care/patients/{patient_id}/daily-digest")
async def daily_digest(patient_id: str, current_user: User = Depends(get_current_user)):
    """Generate an AI-powered daily summary for a caregiver."""
    await resolve_target_user_id(current_user, patient_id)

    today = datetime.now(timezone.utc).date().isoformat()
    patient = await db.users.find_one({"user_id": patient_id}, {"_id": 0, "name": 1})
    patient_name = patient.get("name", "Patient") if patient else "Patient"

    meds = await db.medications.find({"user_id": patient_id, "active": True}, {"_id": 0}).to_list(50)
    intakes = await db.medication_intake_logs.find({
        "user_id": patient_id,
        "taken_at": {"$regex": f"^{today}"}
    }, {"_id": 0}).to_list(200)
    mood = await db.mood_checkins.find({
        "user_id": patient_id,
        "created_at": {"$regex": f"^{today}"}
    }, {"_id": 0}).to_list(10)
    alerts = await db.safety_alerts.find({
        "user_id": patient_id,
        "created_at": {"$regex": f"^{today}"}
    }, {"_id": 0}).to_list(50)

    total_doses = sum(len(m.get("scheduled_times", [])) for m in meds)
    taken_doses = len(intakes)
    adherence = f"{taken_doses}/{total_doses}" if total_doses > 0 else "No medications scheduled"

    mood_summary = "No check-in today"
    if mood:
        latest = mood[-1]
        mood_summary = f"Mood: {latest.get('mood_score', '?')}/3, Energy: {latest.get('energy_level', '?')}/3"

    alert_summary = f"{len(alerts)} safety alert(s)" if alerts else "No safety alerts"

    prompt = f"""Summarize this patient's day in 3 concise, caring sentences for their caregiver. Patient name: {patient_name}.

Medication adherence: {adherence}
{mood_summary}
{alert_summary}

Be warm but factual. If adherence is low, mention it gently."""

    try:
        ai_client = get_openai_client()
        resp = await ai_client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "system", "content": "You are a healthcare assistant writing brief daily summaries for family caregivers."}, {"role": "user", "content": prompt}],
            max_tokens=200,
            temperature=0.7
        )
        summary = resp.choices[0].message.content.strip()
    except Exception:
        summary = f"{patient_name}'s day: {adherence} medications taken. {mood_summary}. {alert_summary}."

    return {
        "patient_name": patient_name,
        "date": today,
        "summary": summary,
        "adherence": adherence,
        "mood": mood_summary,
        "alerts": alert_summary
    }


@api_router.get("/care/patients/{patient_id}/report")
async def care_report_pdf(
    patient_id: str,
    days: int = 30,
    current_user: User = Depends(get_current_user)
):
    """Generate a PDF care report for a patient."""
    if not _HAS_REPORTLAB:
        raise HTTPException(status_code=503, detail="PDF generation not available")

    await resolve_target_user_id(current_user, patient_id)

    patient = await db.users.find_one({"user_id": patient_id}, {"_id": 0, "name": 1, "email": 1})
    patient_name = patient.get("name", "Patient") if patient else "Patient"

    cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()

    meds = await db.medications.find({"user_id": patient_id, "active": True}, {"_id": 0}).to_list(50)
    intakes = await db.medication_intake_logs.find({"user_id": patient_id, "taken_at": {"$gte": cutoff}}, {"_id": 0}).to_list(5000)
    moods = await db.mood_checkins.find({"user_id": patient_id, "created_at": {"$gte": cutoff}}, {"_id": 0}).to_list(500)
    alerts = await db.safety_alerts.find({"user_id": patient_id, "created_at": {"$gte": cutoff}}, {"_id": 0}).to_list(500)

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, topMargin=0.75*inch, bottomMargin=0.75*inch)
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle('ReportTitle', parent=styles['Title'], fontSize=18, spaceAfter=20)
    heading_style = ParagraphStyle('SectionHead', parent=styles['Heading2'], fontSize=14, spaceAfter=10, textColor=colors.HexColor('#7c3aed'))

    story = []
    story.append(Paragraph(f"AlzaHelp Care Report &mdash; {patient_name}", title_style))
    story.append(Paragraph(f"Report period: Last {days} days | Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d')}", styles['Normal']))
    story.append(Spacer(1, 20))

    # Medications
    story.append(Paragraph("Medications", heading_style))
    if meds:
        med_data = [["Medication", "Dosage", "Schedule"]]
        for m in meds:
            med_data.append([m.get("name", ""), m.get("dosage", ""), ", ".join(m.get("scheduled_times", []))])
        tbl = Table(med_data, colWidths=[2.5*inch, 1.5*inch, 2*inch])
        tbl.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#7c3aed')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f5f3ff')]),
        ]))
        story.append(tbl)
    else:
        story.append(Paragraph("No active medications.", styles['Normal']))
    story.append(Spacer(1, 15))

    # Adherence
    total_doses = len(intakes)
    story.append(Paragraph("Medication Adherence", heading_style))
    story.append(Paragraph(f"Total recorded intakes in period: {total_doses}", styles['Normal']))
    story.append(Spacer(1, 15))

    # Mood
    story.append(Paragraph("Mood &amp; Wellbeing", heading_style))
    if moods:
        avg_mood = sum(m.get("mood_score", 0) for m in moods) / len(moods)
        avg_energy = sum(m.get("energy_level", 0) for m in moods) / len(moods)
        story.append(Paragraph(f"Check-ins: {len(moods)} | Avg Mood: {avg_mood:.1f}/3 | Avg Energy: {avg_energy:.1f}/3", styles['Normal']))
    else:
        story.append(Paragraph("No mood check-ins recorded.", styles['Normal']))
    story.append(Spacer(1, 15))

    # Safety
    story.append(Paragraph("Safety Events", heading_style))
    story.append(Paragraph(f"Alerts in period: {len(alerts)}", styles['Normal']))
    if alerts:
        for a in alerts[:10]:
            story.append(Paragraph(f"&bull; {a.get('type', 'alert')} &mdash; {a.get('created_at', '')[:10]}: {a.get('message', '')[:100]}", styles['Normal']))
    story.append(Spacer(1, 20))

    story.append(Paragraph("Generated by AlzaHelp &mdash; alzahelp.app", styles['Normal']))

    doc.build(story)
    buffer.seek(0)

    safe_name = patient_name.replace(' ', '_')
    return StreamingResponse(
        buffer,
        media_type="application/pdf",
        headers={"Content-Disposition": f"attachment; filename=alzahelp_report_{safe_name}_{datetime.now(timezone.utc).strftime('%Y%m%d')}.pdf"}
    )


# ==================== REFERRAL SHARING ====================

@api_router.post("/referral/generate")
async def generate_referral(current_user: User = Depends(get_current_user)):
    """Generate a unique referral code for the current user."""
    guard_demo_write(current_user)
    user_doc = await db.users.find_one({"user_id": current_user.user_id}, {"_id": 0, "referral_code": 1})
    existing_code = user_doc.get("referral_code") if user_doc else None

    if existing_code:
        count = await db.users.count_documents({"referred_by": existing_code})
        return {"code": existing_code, "referral_count": count}

    code = secrets.token_urlsafe(6)
    await db.users.update_one(
        {"user_id": current_user.user_id},
        {"$set": {"referral_code": code}}
    )
    return {"code": code, "referral_count": 0}


@api_router.get("/referral/stats")
async def referral_stats(current_user: User = Depends(get_current_user)):
    """Get referral statistics for the current user."""
    user_doc = await db.users.find_one({"user_id": current_user.user_id}, {"_id": 0, "referral_code": 1})
    code = user_doc.get("referral_code") if user_doc else None
    if not code:
        return {"code": None, "referral_count": 0}
    count = await db.users.count_documents({"referred_by": code})
    return {"code": code, "referral_count": count}


# ==================== RAG CHAT ====================

def keyword_match_score(query: str, text: str) -> float:
    """Calculate keyword match score between query and text"""
    if not text:
        return 0.0
    
    query_words = set(re.findall(r'\w+', query.lower()))
    text_words = set(re.findall(r'\w+', text.lower()))
    
    # Common words to filter out
    stop_words = {'is', 'my', 'the', 'a', 'an', 'who', 'what', 'when', 'where', 'how', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can', 'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from', 'or', 'and', 'but', 'not', 'so', 'if', 'then', 'than', 'too', 'very', 'just', 'about', 'me', 'tell', 'us', 'them', 'their', 'i'}
    
    query_words = query_words - stop_words
    
    if not query_words:
        return 0.0
    
    # Check for exact matches and partial matches
    matches = 0
    for q_word in query_words:
        if q_word in text_words:
            matches += 2  # Exact match
        elif any(q_word in t_word or t_word in q_word for t_word in text_words):
            matches += 1  # Partial match
    
    return matches / (len(query_words) * 2)  # Normalize to 0-1

def _instruction_sort_key(inst: dict) -> Tuple[int, str]:
    return int(inst.get("version", 1)), inst.get("updated_at", "")

async def retrieve_instruction_semantic_context(
    user_id: str,
    query: str,
    top_k: int = 6
) -> dict:
    """Retrieve active instruction chunks using embedding + lexical hybrid scoring."""
    reference_date = datetime.now(timezone.utc).date()
    query_lower = query.lower()
    medication_query = any(
        token in query_lower for token in
        ["medication", "medicine", "pill", "dosage", "dose", "take", "regimen", "procedure"]
    )

    all_instructions = await db.care_instructions.find(
        {"user_id": user_id, "active": True},
        {"_id": 0}
    ).to_list(500)
    effective = [i for i in all_instructions if instruction_allowed_for_patient_use(i, reference_date)]

    # Keep only latest active version for each medication regimen key.
    non_med = []
    med_by_key = {}
    for inst in effective:
        if inst.get("policy_type") != "medication":
            non_med.append(inst)
            continue
        key = inst.get("regimen_key") or re.sub(r"\s+", "_", inst.get("title", "").lower())
        prev = med_by_key.get(key)
        if not prev or _instruction_sort_key(inst) > _instruction_sort_key(prev):
            med_by_key[key] = inst

    effective = non_med + list(med_by_key.values())
    if not effective:
        return {"instructions": [], "citations": [], "context_block": ""}

    instruction_map = {i["id"]: i for i in effective}
    instruction_ids = list(instruction_map.keys())
    chunks = await db.care_instruction_chunks.find(
        {"user_id": user_id, "instruction_id": {"$in": instruction_ids}},
        {"_id": 0}
    ).to_list(5000)

    # Fallback if chunk collection is not ready yet.
    if not chunks:
        for inst in effective:
            chunks.append({
                "instruction_id": inst["id"],
                "chunk_index": 0,
                "chunk_text": (inst.get("instruction_text") or "")[:900],
                "snippet": (inst.get("instruction_text") or "")[:260],
                "embedding": None
            })

    query_embedding = await get_query_embedding(query)
    scored_chunks = []
    for chunk in chunks:
        chunk_text = chunk.get("chunk_text") or ""
        lexical = keyword_match_score(query, chunk_text)
        semantic = cosine_similarity(query_embedding, chunk.get("embedding")) if query_embedding else 0.0
        score = lexical if query_embedding is None else (semantic * 0.78 + lexical * 0.22)

        inst = instruction_map.get(chunk.get("instruction_id"))
        if not inst:
            continue
        if medication_query and inst.get("policy_type") == "medication":
            score += 0.08
        if "today" in query_lower and inst.get("frequency") == "daily":
            score += 0.05
        if "week" in query_lower and inst.get("frequency") == "weekly":
            score += 0.05
        if score <= 0.01:
            continue
        scored_chunks.append((score, chunk, inst))

    if not scored_chunks:
        # Ensure at least one instruction appears in context.
        fallback_inst = sorted(effective, key=_instruction_sort_key, reverse=True)[: min(top_k, len(effective))]
        citations = []
        context_lines = []
        for idx, inst in enumerate(fallback_inst, start=1):
            c_id = f"C{idx}"
            snippet = (inst.get("summary") or inst.get("instruction_text") or "")[:260]
            citations.append({
                "id": c_id,
                "instruction_id": inst["id"],
                "title": inst.get("title"),
                "version": int(inst.get("version", 1)),
                "policy_type": inst.get("policy_type", "general"),
                "snippet": snippet
            })
            context_lines.append(
                f"[{c_id}] {inst.get('title', 'Instruction')} (v{inst.get('version', 1)}): {snippet}"
            )
        return {
            "instructions": fallback_inst,
            "citations": citations,
            "context_block": "\n".join(context_lines)
        }

    scored_chunks.sort(key=lambda x: x[0], reverse=True)
    top = scored_chunks[:top_k]

    citations = []
    ordered_instructions = []
    seen_instruction_ids = set()
    context_lines = []
    for idx, (_, chunk, inst) in enumerate(top, start=1):
        c_id = f"C{idx}"
        snippet = re.sub(r"\s+", " ", chunk.get("snippet") or chunk.get("chunk_text") or "").strip()[:260]
        citations.append({
            "id": c_id,
            "instruction_id": inst["id"],
            "title": inst.get("title"),
            "version": int(inst.get("version", 1)),
            "policy_type": inst.get("policy_type", "general"),
            "snippet": snippet
        })
        schedule = ", ".join([
            v for v in [inst.get("frequency"), inst.get("day_of_week"), inst.get("time_of_day")] if v
        ])
        context_lines.append(
            f"[{c_id}] {inst.get('title', 'Instruction')} (v{inst.get('version', 1)}, {schedule or 'as needed'}) - {snippet}"
        )
        if inst["id"] not in seen_instruction_ids:
            ordered_instructions.append(inst)
            seen_instruction_ids.add(inst["id"])

    return {
        "instructions": ordered_instructions,
        "citations": citations,
        "context_block": "\n".join(context_lines)
    }

async def search_similar_content(user_id: str, query: str, top_k: int = 10) -> dict:
    """Search for similar memories, family members, and care instructions using hybrid semantic+lexical scoring."""
    query_embedding = await get_query_embedding(query)

    memories = await db.memories.find(
        {"user_id": user_id},
        {"_id": 0, "search_text": 0}
    ).to_list(500)

    family = await db.family_members.find(
        {"user_id": user_id},
        {"_id": 0, "search_text": 0}
    ).to_list(100)

    instruction_context = await retrieve_instruction_semantic_context(user_id, query, top_k=min(top_k, 8))

    # Hybrid score memories (semantic 78% + lexical 22%)
    memory_scores = []
    for mem in memories:
        search_text = _build_memory_search_text(mem)
        lexical = keyword_match_score(query, search_text)
        mem_emb = mem.get("embedding")
        semantic = cosine_similarity(query_embedding, mem_emb) if query_embedding and mem_emb else 0.0
        score = (semantic * 0.78 + lexical * 0.22) if (query_embedding and mem_emb) else lexical
        if score > 0.05:
            mem_clean = {k: v for k, v in mem.items() if k != "embedding"}
            memory_scores.append((score, mem_clean))

    # Hybrid score family
    family_scores = []
    for fam in family:
        search_text = _build_family_search_text(fam)
        lexical = keyword_match_score(query, search_text)
        fam_emb = fam.get("embedding")
        semantic = cosine_similarity(query_embedding, fam_emb) if query_embedding and fam_emb else 0.0
        score = (semantic * 0.78 + lexical * 0.22) if (query_embedding and fam_emb) else lexical
        if score > 0.05:
            fam_clean = {k: v for k, v in fam.items() if k != "embedding"}
            family_scores.append((score, fam_clean))

    memory_scores.sort(key=lambda x: x[0], reverse=True)
    family_scores.sort(key=lambda x: x[0], reverse=True)

    top_memories = [m[1] for m in memory_scores[:top_k]]
    top_family = [f[1] for f in family_scores[:top_k]]
    top_instructions = instruction_context.get("instructions", [])[:top_k]

    return {
        "memories": top_memories,
        "family": top_family,
        "instructions": top_instructions,
        "instruction_context": instruction_context.get("context_block", ""),
        "citations": instruction_context.get("citations", [])
    }

@api_router.post("/admin/backfill-embeddings")
async def backfill_embeddings(current_user: User = Depends(get_current_user)):
    """Backfill embeddings for memories and family members that don't have them."""
    if current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Admin only")
    memories = await db.memories.find({"embedding": None}, {"_id": 0}).to_list(2000)
    mem_count = 0
    for i in range(0, len(memories), 20):
        batch = memories[i:i+20]
        texts = [_build_memory_search_text(m) for m in batch]
        embeddings = await generate_embeddings(texts)
        if embeddings:
            for doc, emb in zip(batch, embeddings):
                if emb:
                    await db.memories.update_one({"id": doc["id"]}, {"$set": {"embedding": emb}})
                    mem_count += 1
    family = await db.family_members.find({"embedding": None}, {"_id": 0}).to_list(2000)
    fam_count = 0
    for i in range(0, len(family), 20):
        batch = family[i:i+20]
        texts = [_build_family_search_text(f) for f in batch]
        embeddings = await generate_embeddings(texts)
        if embeddings:
            for doc, emb in zip(batch, embeddings):
                if emb:
                    await db.family_members.update_one({"id": doc["id"]}, {"$set": {"embedding": emb}})
                    fam_count += 1
    return {"memories_updated": mem_count, "family_updated": fam_count}

@api_router.post("/chat")
@rate_limit("60/minute")
async def chat_with_assistant(
    request: Request,
    chat_request: ChatRequest,
    current_user: User = Depends(get_current_user)
):
    """Chat with AI assistant using RAG"""
    client = get_openai_client()
    
    # Search for relevant context
    context = await search_similar_content(current_user.user_id, chat_request.message)
    
    # Build context string
    context_parts = []
    
    if context['family']:
        context_parts.append("FAMILY MEMBERS:")
        for fam in context['family']:
            info = f"- {fam['name']} ({fam['relationship_label']})"
            if fam.get('birthday'):
                info += f", Birthday: {fam['birthday']}"
            if fam.get('phone'):
                info += f", Phone: {fam['phone']}"
            if fam.get('address'):
                info += f", Address: {fam['address']}"
            if fam.get('notes'):
                info += f", Notes: {fam['notes']}"
            context_parts.append(info)
    
    if context['memories']:
        context_parts.append("\nMEMORIES:")
        for mem in context['memories']:
            people_str = ", ".join(mem.get('people', [])) if mem.get('people') else "unknown"
            info = f"- {mem['title']} ({mem['date']}): {mem['description']}"
            if mem.get('location'):
                info += f" Location: {mem['location']}"
            info += f" People: {people_str}"
            context_parts.append(info)

    if context.get('instruction_context'):
        context_parts.append("\nCARE INSTRUCTIONS:")
        context_parts.append(context.get("instruction_context", ""))
    
    context_str = "\n".join(context_parts) if context_parts else "No specific memories, family, or care instructions found."
    
    # System message for the assistant
    system_message = f"""You are a warm, caring assistant helping someone with memory challenges remember their loved ones and precious memories.

The user's name is {current_user.name}. Be patient, kind, and speak in simple, clear language.

Here is information about their family, memories, and care instructions that may help answer their question:

{context_str}

Important guidelines:
1. Always be warm and reassuring
2. Use simple, clear language
3. If you find relevant information, share it naturally and warmly
4. If you don't have specific information, gently say so and offer to help with what you do know
5. Never make up information - only use what's provided
6. Address the user by name occasionally to make it personal
7. If they ask about family, describe relationships clearly (e.g., "Maria is your wife")
8. If the question is about medication or routines, prioritize CARE INSTRUCTIONS exactly as written
9. When you use care instruction evidence, reference citations like [C1], [C2] in your answer"""

    # Save user message
    user_msg = ChatMessage(
        user_id=current_user.user_id,
        session_id=chat_request.session_id,
        role="user",
        content=chat_request.message
    )
    await db.chat_messages.insert_one(user_msg.model_dump())
    
    # Get chat history for context
    history = await db.chat_messages.find(
        {"user_id": current_user.user_id, "session_id": chat_request.session_id},
        {"_id": 0}
    ).sort("timestamp", 1).to_list(20)
    
    # Build messages array for OpenAI
    messages = [{"role": "system", "content": system_message}]
    for msg in history[-10:]:  # Last 10 messages for context
        messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({"role": "user", "content": chat_request.message})
    
    # Call OpenAI directly
    citations = context.get("citations", [])
    try:
        completion = await client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            max_tokens=500,
            temperature=0.7
        )
        response = completion.choices[0].message.content
    except Exception as e:
        logger.error(f"OpenAI chat error: {e}")
        response = "I'm sorry, I'm having trouble thinking right now. Could you try again?"

    citation_block = build_citation_snippet_block(citations)
    if citation_block:
        if "Citations:" not in response:
            response = f"{response}\n\n{citation_block}"
    
    # Save assistant response
    assistant_msg = ChatMessage(
        user_id=current_user.user_id,
        session_id=chat_request.session_id,
        role="assistant",
        content=response
    )
    await db.chat_messages.insert_one(assistant_msg.model_dump())
    
    return {"response": response, "citations": citations}

@api_router.get("/chat/history/{session_id}")
async def get_chat_history(
    session_id: str,
    current_user: User = Depends(get_current_user)
):
    """Get chat history for a session"""
    messages = await db.chat_messages.find(
        {"user_id": current_user.user_id, "session_id": session_id},
        {"_id": 0}
    ).sort("timestamp", 1).to_list(100)
    return messages

# ==================== VOICE ASSISTANT ====================

class VoiceCommand(BaseModel):
    text: str
    session_id: str

class TTSRequest(BaseModel):
    text: str
    voice: str = "nova"  # Warm, friendly voice good for elderly

@api_router.post("/tts")
async def text_to_speech(
    request: TTSRequest,
    current_user: User = Depends(get_current_user)
):
    """Convert text to speech using OpenAI TTS"""
    client = get_openai_client()
    
    try:
        # Generate speech using OpenAI TTS API directly
        response = await client.audio.speech.create(
            model="tts-1",
            voice=request.voice,
            input=request.text[:4000],  # Limit to 4000 chars
            speed=0.9  # Slightly slower for elderly users
        )
        
        # Get audio content and encode to base64
        audio_content = response.content
        audio_base64 = base64.b64encode(audio_content).decode('utf-8')
        
        return {"audio": audio_base64, "format": "mp3"}
    except Exception as e:
        logger.error(f"TTS error: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate speech")

@api_router.post("/voice/transcribe")
async def voice_transcribe(
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user)
):
    """Transcribe audio file using Whisper (iOS Safari fallback for Web Speech API)."""
    audio_bytes = await file.read()
    if len(audio_bytes) > 10 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Audio file too large (max 10MB)")
    text = await transcribe_audio_bytes(audio_bytes, filename=file.filename or "audio.webm")
    if not text:
        raise HTTPException(status_code=422, detail="Could not transcribe audio")
    return {"text": text}

@api_router.post("/voice-command")
@rate_limit("60/minute")
async def process_voice_command(
    request: Request,
    command: VoiceCommand,
    current_user: User = Depends(get_current_user)
):
    """Process voice command and return action + response"""
    text = command.text.lower().strip()
    
    # Define navigation commands
    nav_commands = {
        "family": ["family", "loved ones", "relatives", "my family", "show family", "go to family"],
        "timeline": ["memories", "memory", "timeline", "my memories", "show memories", "remember"],
        "quiz": ["face quiz", "faces quiz", "who is this", "recognize faces"],
        "cards": ["card game", "matching game", "match game", "memory match", "flip cards", "card match"],
        "sudoku": ["sudoku", "number puzzle", "number game", "numbers game"],
        "week": ["week", "my week", "yesterday", "today activities", "what did i do", "recent"],
        "assistant": ["ask", "help", "assistant", "chat", "talk"],
        "reminders": ["reminders", "today", "tasks", "schedule"],
        "mood": ["mood", "behavior", "agitation", "anxiety", "how i feel", "sleep quality"],
        "medications": ["medication tracker", "my medications", "pills", "medicine"],
        "caregiver": ["caregiver", "care portal", "clinician"],
        "navigation": ["gps", "navigation", "navigate", "route", "directions", "path", "where do i go"],
        "home": ["home", "main", "start", "beginning"]
    }

    emergency_terms = [
        "sos", "emergency", "help me", "i fell", "fall down", "i am lost", "i'm lost", "lost"
    ]
    if any(term in text for term in emergency_terms):
        return {
            "action": "navigate",
            "target": "navigation",
            "response": "Opening Safety now. Press the SOS button and I will help notify your caregiver right away."
        }

    chess_query = "chess" in text and any(term in text for term in ["play", "game", "quiz", "practice"])
    medication_taken_report = is_medication_taken_report(text)
    medication_schedule_query = is_medication_schedule_question(text)
    instruction_today_query = is_today_instruction_question(text)
    special_intent = "none"
    if not chess_query and not medication_taken_report and not medication_schedule_query and not instruction_today_query:
        special_intent = await classify_special_voice_intent(command.text)

    if chess_query or special_intent == "unsupported_chess":
        return {
            "action": "speak",
            "response": "We do not have chess yet. We can play Faces Quiz, Memory Match, or Sudoku. Which one would you like?"
        }

    if medication_schedule_query or special_intent == "medication_schedule":
        try:
            response = await build_today_medication_voice_response(current_user.user_id)
        except Exception as e:
            logger.error(f"Medication voice schedule error: {e}")
            response = "I could not read your medication schedule right now. Please open your medication tracker or ask your caregiver."
        return {
            "action": "speak",
            "response": response
        }

    if instruction_today_query or special_intent == "today_instructions":
        try:
            plan = await build_patient_today_plan(current_user.user_id)
            response = plan.get("voice_script") or "I do not have a care plan to read right now."
        except Exception as e:
            logger.error(f"Today instruction voice error: {e}")
            response = "I could not read today's care instructions right now. Please open your medication tracker or ask your caregiver."
        return {
            "action": "speak",
            "response": response
        }

    # Medication taken report (voice intake logging)
    if medication_taken_report or special_intent == "medication_taken":
        try:
            return await handle_voice_medication_intake(current_user.user_id, command.text)
        except Exception as e:
            logger.error(f"Voice medication intake error: {e}")
            return {
                "action": "speak",
                "response": "I had trouble recording your medication. Please try again or use the medication tracker to mark it as taken."
            }

    # Mood report (voice mood logging)
    if special_intent == "mood_report":
        try:
            return await handle_voice_mood_report(current_user.user_id, command.text)
        except Exception as e:
            logger.error(f"Voice mood report error: {e}")
            return {
                "action": "speak",
                "response": "I had trouble recording your mood. Please try again later or use the mood tracker."
            }

    # Check for generic game/play/quiz intent first
    game_words = ["game", "play", "quiz", "practice", "exercise", "brain"]
    specific_game_words = ["face", "sudoku", "card", "match", "flip", "number"]
    if any(w in text for w in game_words) and not any(w in text for w in specific_game_words):
        return {
            "action": None,
            "response": "We have three brain games: Faces Quiz to recognize family, Memory Match to find pairs, and Sudoku for number puzzles. Which one would you like to play?"
        }

    # Check for navigation commands
    for nav_target, keywords in nav_commands.items():
        for keyword in keywords:
            if keyword in text:
                # Generate friendly response
                responses = {
                    "family": "Taking you to see your family now.",
                    "timeline": "Let's look at your memories together.",
                    "quiz": "Great! Let's practice recognizing faces!",
                    "cards": "Let's play the memory matching game!",
                    "sudoku": "Let's exercise your brain with Sudoku!",
                    "week": "Let me show you what you did this week.",
                    "assistant": "I'm here to help. What would you like to know?",
                    "reminders": "Here are your reminders for today.",
                    "mood": "Opening mood and behavior tracking.",
                    "medications": "Opening your medication tracker.",
                    "caregiver": "Opening the caregiver portal.",
                    "navigation": "Let's open your route guidance so you can see where to go.",
                    "home": "Taking you back home."
                }
                return {
                    "action": "navigate",
                    "target": nav_target,
                    "response": responses.get(nav_target, f"Going to {nav_target}.")
                }
    
    # Check for creation commands
    if any(word in text for word in ["add", "create", "new", "remember this"]):
        if "memory" in text or "remember" in text:
            return {
                "action": "create_memory",
                "response": "I'll help you add a new memory. What would you like to remember?"
            }
        elif "reminder" in text:
            return {
                "action": "create_reminder", 
                "response": "I'll help you set a reminder. What should I remind you about?"
            }
        elif "family" in text or "person" in text:
            return {
                "action": "create_family",
                "response": "I'll help you add a family member. Who would you like to add?"
            }
    
    # For questions, use the RAG chat system
    context = await search_similar_content(current_user.user_id, text)
    medication_query = any(
        token in text for token in ["medication", "medicine", "pill", "dose", "dosage", "regimen", "take"]
    )
    
    context_parts = []
    if context['family']:
        context_parts.append("FAMILY MEMBERS:")
        for fam in context['family']:
            info = f"- {fam['name']} ({fam['relationship_label']})"
            if fam.get('notes'):
                info += f": {fam['notes']}"
            context_parts.append(info)
    
    if context['memories']:
        context_parts.append("\nMEMORIES:")
        for mem in context['memories']:
            info = f"- {mem['title']} ({mem['date']}): {mem['description']}"
            context_parts.append(info)

    if context.get('instruction_context'):
        context_parts.append("\nCARE INSTRUCTIONS:")
        context_parts.append(context.get("instruction_context", ""))
    if medication_query and not context.get("instructions"):
        return {
            "action": "speak",
            "response": "I do not have an active signed medication protocol for today. Please contact your caregiver or clinician before taking medicine."
        }
    
    context_str = "\n".join(context_parts) if context_parts else "No specific information found."
    
    system_message = f"""You are a warm, caring voice assistant helping {current_user.name} who has memory challenges.
Keep responses SHORT (2-3 sentences max) and speak naturally as if talking to them.

User's information:
{context_str}

Guidelines:
- Be warm, patient, and reassuring
- Use simple, clear language
- If you find information, share it naturally
- If you don't know, say so kindly
- Address them by name occasionally
- For medication and routine questions, follow CARE INSTRUCTIONS exactly and do not invent steps
- Never read draft, rejected, or unsigned medication instructions"""

    # Call OpenAI directly
    try:
        client = get_openai_client()
        completion = await client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": command.text}
            ],
            max_tokens=150,
            temperature=0.7
        )
        response = completion.choices[0].message.content
    except Exception as e:
        logger.error(f"OpenAI voice command error: {e}")
        response = "I'm sorry, I'm having trouble understanding right now. Could you try again?"
    
    return {
        "action": "speak",
        "response": response
    }

# ==================== EXTERNAL BOT WEBHOOKS ====================

@api_router.post("/webhooks/telegram/bot", response_model=dict)
@rate_limit("30/minute")
async def telegram_bot_webhook(request: Request):
    expected_secret = os.environ.get("TELEGRAM_BOT_WEBHOOK_SECRET", "").strip()
    if expected_secret:
        provided_secret = request.headers.get("X-Telegram-Bot-Api-Secret-Token", "")
        if provided_secret != expected_secret:
            raise HTTPException(status_code=401, detail="Invalid Telegram webhook secret")

    update = await request.json()
    message = update.get("message") or update.get("edited_message")
    if not message:
        return {"ok": True, "ignored": "no_message"}

    chat = message.get("chat") or {}
    chat_id = str(chat.get("id") or "")
    if not chat_id:
        return {"ok": True, "ignored": "no_chat_id"}

    display_name = " ".join(
        [p for p in [chat.get("first_name"), chat.get("last_name")] if p]
    ) or chat.get("username") or "Telegram User"

    incoming_text = (message.get("text") or message.get("caption") or "").strip()
    prefer_voice = False

    if not incoming_text:
        voice_obj = message.get("voice") or message.get("audio")
        file_id = (voice_obj or {}).get("file_id") if isinstance(voice_obj, dict) else None
        if file_id:
            audio_bytes, filename = await download_telegram_voice_file(file_id)
            transcript = await transcribe_audio_bytes(audio_bytes, filename or "telegram_audio.ogg") if audio_bytes else None
            if transcript:
                incoming_text = transcript
                prefer_voice = True

    if not incoming_text:
        response_text = "I could not parse that message. Please send text or a clear voice note."
        await send_telegram_text_message(chat_id, response_text)
        await log_external_bot_message("telegram", chat_id, None, None, None, response_text, "parse_error", "error")
        return {"ok": True, "status": "parse_error"}

    result = await handle_external_doctor_message(
        channel="telegram",
        peer_id=chat_id,
        peer_display_name=display_name,
        text=incoming_text,
        prefer_voice=prefer_voice
    )
    text_send = await send_telegram_text_message(chat_id, result.get("text", ""))
    voice_send = None
    if prefer_voice and result.get("voice_text"):
        voice_send = await send_telegram_voice_message(chat_id, result["voice_text"])
    return {
        "ok": True,
        "status": result.get("status"),
        "intent": result.get("intent"),
        "text_send": text_send,
        "voice_send": voice_send
    }

@api_router.post("/webhooks/whatsapp/bot")
@rate_limit("30/minute")
async def whatsapp_bot_webhook(request: Request):
    form = await request.form()
    form_data = {k: str(v) for k, v in form.items()}
    provided_signature = request.headers.get("X-Twilio-Signature", "")
    twilio_auth_token = os.environ.get("TWILIO_AUTH_TOKEN", "").strip()
    # Use a configurable public URL for signature validation.  Behind a
    # reverse proxy the request.url seen by the app often differs from the
    # URL Twilio actually signed (the public-facing one).
    twilio_public_url = os.environ.get("TWILIO_WEBHOOK_PUBLIC_URL", "").strip()
    sig_url = twilio_public_url or str(request.url)
    if not verify_twilio_signature(twilio_auth_token, sig_url, form_data, provided_signature):
        raise HTTPException(status_code=401, detail="Invalid Twilio signature")

    from_peer = normalize_external_peer_id("whatsapp", form_data.get("From", ""))
    display_name = form_data.get("ProfileName", "WhatsApp User")
    incoming_text = (form_data.get("Body") or "").strip()

    if not incoming_text:
        try:
            num_media = int(form_data.get("NumMedia", "0") or 0)
        except Exception:
            num_media = 0
        if num_media > 0:
            media_url = form_data.get("MediaUrl0", "")
            media_type = (form_data.get("MediaContentType0") or "").lower()
            if media_url and ("audio" in media_type or "ogg" in media_type):
                audio_bytes, _ctype = await download_twilio_media(media_url)
                transcript = await transcribe_audio_bytes(audio_bytes, "whatsapp_audio.ogg") if audio_bytes else None
                if transcript:
                    incoming_text = transcript

    if not incoming_text:
        reply_text = "I could not parse that message. Please send text or a clear voice note."
        await log_external_bot_message("whatsapp", from_peer, None, None, None, reply_text, "parse_error", "error")
        xml = f'<?xml version="1.0" encoding="UTF-8"?><Response><Message>{xml_escape(reply_text)}</Message></Response>'
        return Response(content=xml, media_type="application/xml")

    result = await handle_external_doctor_message(
        channel="whatsapp",
        peer_id=from_peer,
        peer_display_name=display_name,
        text=incoming_text,
        prefer_voice=False
    )
    reply_text = result.get("text", "Request processed.")
    xml = f'<?xml version="1.0" encoding="UTF-8"?><Response><Message>{xml_escape(reply_text)}</Message></Response>'
    return Response(content=xml, media_type="application/xml")

# ==================== PUSH NOTIFICATIONS + MEDICATION SCHEDULER ====================

VAPID_PRIVATE_KEY = os.environ.get("VAPID_PRIVATE_KEY", "")
VAPID_CLAIMS = {"sub": f"mailto:{os.environ.get('VAPID_CONTACT_EMAIL', 'admin@alzahelp.com')}"}
TWILIO_SID = os.environ.get("TWILIO_ACCOUNT_SID", "")
TWILIO_FROM = os.environ.get("TWILIO_FROM_NUMBER", "")

async def send_push_to_user(user_id: str, title: str, body: str, url: str = "/dashboard"):
    """Send web push notification to all subscriptions for a user."""
    if not VAPID_PRIVATE_KEY:
        return
    try:
        from pywebpush import webpush, WebPushException
    except ImportError:
        return
    subs = await db.push_subscriptions.find({"user_id": user_id}).to_list(20)
    payload = json.dumps({"title": title, "body": body, "url": url})
    for sub in subs:
        try:
            webpush(
                subscription_info={"endpoint": sub["endpoint"], "keys": sub["keys"]},
                data=payload,
                vapid_private_key=VAPID_PRIVATE_KEY,
                vapid_claims=VAPID_CLAIMS
            )
        except Exception as e:
            # Clean up stale subscriptions (gone/not found)
            if hasattr(e, 'response') and getattr(e.response, 'status_code', 0) in (404, 410):
                await db.push_subscriptions.delete_one({"id": sub.get("id")})
            else:
                logger.error("Push send failed: %s", e)

async def notify_patient_update(patient_user_id: str, update_type: str, detail: str):
    """Send push notification to a patient when their care data is updated."""
    titles = {
        "medication_added": "New Medication Added",
        "medication_updated": "Medication Updated",
        "instruction_added": "New Care Instruction",
    }
    try:
        await send_push_to_user(patient_user_id, titles.get(update_type, "Care Update"), detail, "/dashboard")
    except Exception:
        pass


async def send_sms_fallback(phone: str, message: str):
    """Send SMS via Twilio as fallback notification."""
    twilio_token = os.environ.get("TWILIO_AUTH_TOKEN", "").strip()
    if not all([TWILIO_SID, twilio_token, TWILIO_FROM, phone]):
        return
    try:
        from twilio.rest import Client as TwilioClient
        tc = TwilioClient(TWILIO_SID, twilio_token)
        tc.messages.create(body=message, from_=TWILIO_FROM, to=phone)
    except Exception as e:
        logger.error("SMS send failed: %s", e)

@api_router.post("/push/subscribe")
async def subscribe_push(request: Request, current_user: User = Depends(get_current_user)):
    body = await request.json()
    sub_doc = {
        "id": f"push_{uuid.uuid4().hex[:12]}",
        "user_id": current_user.user_id,
        "endpoint": body["endpoint"],
        "keys": body.get("keys", {}),
        "created_at": datetime.now(timezone.utc).isoformat()
    }
    await db.push_subscriptions.update_one(
        {"user_id": current_user.user_id, "endpoint": body["endpoint"]},
        {"$set": sub_doc},
        upsert=True
    )
    return {"status": "subscribed"}

@api_router.delete("/push/unsubscribe")
async def unsubscribe_push(request: Request, current_user: User = Depends(get_current_user)):
    body = await request.json()
    await db.push_subscriptions.delete_one({"user_id": current_user.user_id, "endpoint": body.get("endpoint", "")})
    return {"status": "unsubscribed"}

async def check_medication_reminders():
    """Check for medications due in ~15 minutes and send push notifications."""
    try:
        now = datetime.now(timezone.utc)
        target_time = (now + timedelta(minutes=15)).strftime("%H:%M")
        medications = await db.medications.find({"active": True}).to_list(1000)
        for med in medications:
            scheduled_times = med.get("scheduled_times", [])
            if target_time in scheduled_times:
                user_id = med["user_id"]
                await send_push_to_user(
                    user_id,
                    f"Medication Reminder: {med['name']}",
                    f"Take {med.get('dosage', '')} {med['name']} in 15 minutes",
                    "/dashboard"
                )
        # Check for overdue doses (30min+) and send SMS fallback
        overdue_time = (now - timedelta(minutes=30)).strftime("%H:%M")
        for med in medications:
            if overdue_time in med.get("scheduled_times", []):
                user_id = med["user_id"]
                # Check if dose was already taken today
                today_str = now.strftime("%Y-%m-%d")
                intake = await db.medication_intake_logs.find_one({
                    "medication_id": med["id"],
                    "taken_at": {"$regex": f"^{today_str}"}
                })
                if not intake:
                    user_doc = await db.users.find_one({"user_id": user_id})
                    if user_doc and _is_premium(user_doc):
                        contacts = await db.safety_emergency_contacts.find({"user_id": user_id, "receive_sms": True}).to_list(5)
                        msg = f"AlzaHelp: {user_doc.get('name', 'Patient')} has a missed dose of {med['name']} ({med.get('dosage', '')})"
                        for contact in contacts:
                            if contact.get("phone"):
                                await send_sms_fallback(contact["phone"], msg)
    except Exception as e:
        logger.error("Medication reminder check failed: %s", e)

async def ensure_indexes():
    """Create MongoDB indexes for query performance."""
    # Users
    await db.users.create_index("user_id", unique=True)
    await db.users.create_index("email", unique=True)
    await db.users.create_index("role")

    # Medications
    await db.medications.create_index([("user_id", 1), ("active", 1)])
    await db.medications.create_index("id", unique=True)

    # Care instructions
    await db.care_instructions.create_index([("user_id", 1), ("active", 1), ("policy_type", 1)])
    await db.care_instructions.create_index("id", unique=True)
    await db.care_instructions.create_index([("user_id", 1), ("regimen_key", 1), ("version", -1)])

    # Care links
    await db.care_links.create_index([("caregiver_id", 1), ("status", 1)])
    await db.care_links.create_index([("patient_id", 1), ("status", 1)])
    await db.care_links.create_index("id", unique=True)

    # Care invites
    await db.care_invites.create_index("code", unique=True)
    await db.care_invites.create_index([("status", 1), ("expires_at", 1)])

    # Families
    await db.families.create_index([("user_id", 1)])
    await db.families.create_index("id", unique=True)

    # Memories
    await db.memories.create_index([("user_id", 1), ("date", -1)])
    await db.memories.create_index("id", unique=True)

    # Reminders
    await db.reminders.create_index([("user_id", 1), ("active", 1)])
    await db.reminders.create_index("id", unique=True)

    # Safety zones
    await db.safety_zones.create_index([("user_id", 1), ("active", 1)])

    # Chat history
    await db.chat_history.create_index([("user_id", 1), ("created_at", -1)])

    # Push subscriptions
    await db.push_subscriptions.create_index("user_id")

    # Audit logs
    await db.audit_logs.create_index([("user_id", 1), ("created_at", -1)])

    # Sessions - TTL index to auto-expire
    await db.user_sessions.create_index("user_id")
    await db.user_sessions.create_index("created_at", expireAfterSeconds=86400 * 30)

    # External bot links
    await db.external_bot_links.create_index([("doctor_user_id", 1), ("status", 1)])
    await db.external_bot_links.create_index("link_code", unique=True, sparse=True)

    # Medication intake logs
    await db.medication_intake_logs.create_index([("user_id", 1), ("medication_id", 1), ("taken_at", -1)])

    # BPSD observations
    await db.bpsd_observations.create_index([("patient_user_id", 1), ("observed_at", -1)])

    # Mood check-ins
    await db.mood_checkins.create_index([("user_id", 1), ("created_at", -1)])

    # Instruction chunks (for RAG)
    await db.instruction_chunks.create_index([("user_id", 1), ("instruction_id", 1)])

    logger.info("MongoDB indexes ensured")


@app.on_event("startup")
async def setup_db_indexes():
    await ensure_indexes()


# Start scheduler on app startup
@app.on_event("startup")
async def start_medication_scheduler():
    try:
        from apscheduler.schedulers.asyncio import AsyncIOScheduler
        scheduler = AsyncIOScheduler()
        scheduler.add_job(check_medication_reminders, 'interval', minutes=1)
        scheduler.start()
        logger.info("Medication reminder scheduler started")
    except ImportError:
        logger.warning("APScheduler not installed — medication reminders disabled")

# ==================== LEGACY ROUTES ====================

@api_router.get("/")
async def root():
    return {"message": "MemoryKeeper API"}

# Health check endpoint (outside /api prefix for load balancers)
@app.get("/health")
async def health_check():
    try:
        await db.command("ping")
        return {"status": "ok", "db": "connected"}
    except Exception:
        from fastapi.responses import JSONResponse
        return JSONResponse(status_code=503, content={"status": "degraded", "db": "disconnected"})

# Include the router in the main app
app.include_router(api_router)

_cors_origins = os.environ.get('CORS_ORIGINS', '').strip()
if not _cors_origins:
    _cors_origins = "http://localhost:3000"
app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=[o.strip() for o in _cors_origins.split(',') if o.strip()],
    allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
)

@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["Permissions-Policy"] = "camera=(), microphone=(self), geolocation=(self)"
    response.headers["Content-Security-Policy"] = "default-src 'self'; script-src 'self' 'unsafe-inline' https://us-assets.i.posthog.com https://assets.emergent.sh; style-src 'self' 'unsafe-inline'; img-src 'self' data: blob:; connect-src 'self' https://us.i.posthog.com; font-src 'self'; frame-ancestors 'none'"
    return response

@app.on_event("startup")
async def startup_seed():
    await _seed_demo_account()

@app.on_event("shutdown")
async def shutdown_db_client():
    client.close()
