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
import uuid
from datetime import datetime, timezone, timedelta, date
import httpx
import json
import re
import io
import base64
import math
import secrets
import zipfile
import xml.etree.ElementTree as ET
from openai import AsyncOpenAI
from passlib.context import CryptContext
from jose import JWTError, jwt

try:
    from pypdf import PdfReader
except Exception:  # pragma: no cover - optional dependency
    PdfReader = None

try:
    from docx import Document as DocxDocument
except Exception:  # pragma: no cover - optional dependency
    DocxDocument = None

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# Auth Configuration
SECRET_KEY = os.environ.get("JWT_SECRET_KEY", "fallback_secret_key_change_in_production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24 * 30  # 30 days

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
app = FastAPI()

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

# ==================== HELPERS ====================

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def parse_admin_bootstrap_emails() -> set:
    raw = os.environ.get("ADMIN_BOOTSTRAP_EMAILS", "")
    return {e.strip().lower() for e in raw.split(",") if e and e.strip()}

def is_bootstrap_admin_email(email: str) -> bool:
    return email.strip().lower() in parse_admin_bootstrap_emails()

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
        if intent in {"medication_schedule", "today_instructions", "unsupported_chess", "none"}:
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
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class UserCreate(BaseModel):
    email: str
    password: str
    name: str
    role: str = "patient"
    license_number: Optional[str] = None
    medical_organization: Optional[str] = None
    jurisdiction: Optional[str] = None

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
async def register(user_data: UserCreate):
    """Register a new user"""
    # Check if user exists
    existing_user = await db.users.find_one({"email": user_data.email})
    if existing_user:
        raise HTTPException(status_code=400, detail="Email already registered")
    
    requested_role = user_data.role if user_data.role in {"patient", "caregiver", "clinician", "admin"} else "patient"

    # Create user
    user_id = f"user_{uuid.uuid4().hex[:12]}"
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
        "created_at": datetime.now(timezone.utc),
        "updated_at": datetime.now(timezone.utc).isoformat()
    }
    
    await db.users.insert_one(new_user)
    if requested_role == "clinician":
        return {
            "message": "Clinician registration submitted. Waiting for admin approval.",
            "user_id": user_id,
            "requires_approval": True
        }
    return {"message": "User registered successfully", "user_id": user_id, "requires_approval": False}

@api_router.post("/auth/login")
async def login(response: Response, form_data: UserLogin):
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
    
    # Set cookie
    response.set_cookie(
        key="access_token",
        value=access_token,
        httponly=True,
        secure=True,
        samesite="none",
        path="/",
        max_age=ACCESS_TOKEN_EXPIRE_MINUTES * 60
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
    return {"message": "Logged out"}

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
    owner_user_id = await resolve_target_user_id(
        current_user,
        target_user_id,
        require_write=True if target_user_id else False
    )

    # Read file content
    content = await file.read()
    
    # Generate unique filename
    ext = Path(file.filename).suffix if file.filename else ".jpg"
    filename = f"{owner_user_id}_{uuid.uuid4().hex[:8]}{ext}"
    
    # Determine content type
    content_type = file.content_type or 'application/octet-stream'
    
    # Store in GridFS
    file_id = await fs_bucket.upload_from_stream(
        filename,
        io.BytesIO(content),
        metadata={
            "user_id": owner_user_id,
            "content_type": content_type,
            "original_filename": file.filename,
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
    owner_user_id = await resolve_target_user_id(
        current_user,
        target_user_id,
        require_write=True if target_user_id else False
    )

    urls = []
    for file in files:
        content = await file.read()
        ext = Path(file.filename).suffix if file.filename else ".jpg"
        filename = f"{owner_user_id}_{uuid.uuid4().hex[:8]}{ext}"
        content_type = file.content_type or 'application/octet-stream'
        
        await fs_bucket.upload_from_stream(
            filename,
            io.BytesIO(content),
            metadata={
                "user_id": owner_user_id,
                "content_type": content_type,
                "original_filename": file.filename,
                "uploaded_at": datetime.now(timezone.utc).isoformat()
            }
        )
        
        urls.append(f"/api/files/{filename}")
    
    return {"urls": urls}

@api_router.get("/files/{filename}")
async def get_file(filename: str):
    """Retrieve a file from MongoDB GridFS"""
    try:
        # Find file in GridFS
        grid_out = await fs_bucket.open_download_stream_by_name(filename)
        
        # Read content
        content = await grid_out.read()
        
        # Get content type from metadata
        content_type = grid_out.metadata.get('content_type', 'application/octet-stream') if grid_out.metadata else 'application/octet-stream'
        
        return StreamingResponse(
            io.BytesIO(content),
            media_type=content_type,
            headers={
                "Content-Disposition": f"inline; filename={filename}",
                "Cache-Control": "public, max-age=31536000"  # Cache for 1 year
            }
        )
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
    return updated

@api_router.delete("/family/{member_id}")
async def delete_family_member(
    member_id: str,
    current_user: User = Depends(get_current_user)
):
    """Delete a family member"""
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
    return updated

@api_router.delete("/memories/{memory_id}")
async def delete_memory(
    memory_id: str,
    current_user: User = Depends(get_current_user)
):
    """Delete a memory"""
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
    """Search for similar memories, family members, and care instructions."""
    
    # Get all memories
    memories = await db.memories.find(
        {"user_id": user_id},
        {"_id": 0, "search_text": 0}
    ).to_list(500)
    
    # Get all family members
    family = await db.family_members.find(
        {"user_id": user_id},
        {"_id": 0, "search_text": 0}
    ).to_list(100)

    instruction_context = await retrieve_instruction_semantic_context(user_id, query, top_k=min(top_k, 8))
    
    # Calculate scores for memories
    memory_scores = []
    for mem in memories:
        # Build search text from memory fields
        search_text = f"{mem.get('title', '')} {mem.get('date', '')} {mem.get('location', '')} {mem.get('description', '')} {' '.join(mem.get('people', []))}"
        score = keyword_match_score(query, search_text)
        if score > 0:
            memory_scores.append((score, mem))
    
    # Calculate scores for family
    family_scores = []
    for fam in family:
        # Build search text from family fields
        search_text = f"{fam.get('name', '')} {fam.get('relationship', '')} {fam.get('relationship_label', '')} {fam.get('notes', '')} {fam.get('category', '')}"
        score = keyword_match_score(query, search_text)
        if score > 0:
            family_scores.append((score, fam))

    # Sort and get top results
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

@api_router.post("/chat")
async def chat_with_assistant(
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

@api_router.post("/voice-command")
async def process_voice_command(
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
    medication_schedule_query = is_medication_schedule_question(text)
    instruction_today_query = is_today_instruction_question(text)
    special_intent = "none"
    if not chess_query and not medication_schedule_query and not instruction_today_query:
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

# ==================== LEGACY ROUTES ====================

@api_router.get("/")
async def root():
    return {"message": "MemoryKeeper API"}

# Include the router in the main app
app.include_router(api_router)

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=os.environ.get('CORS_ORIGINS', '*').split(','),
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("shutdown")
async def shutdown_db_client():
    client.close()
