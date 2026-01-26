from fastapi import FastAPI, APIRouter, HTTPException, Depends, UploadFile, File, Form, Request, Response
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import os
import logging
from pathlib import Path
from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional, Any
import uuid
from datetime import datetime, timezone, timedelta
import httpx
import aiofiles
import json
from openai import AsyncOpenAI
import numpy as np

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# MongoDB connection
mongo_url = os.environ['MONGO_URL']
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ['DB_NAME']]

# OpenAI client for embeddings
openai_client = AsyncOpenAI(api_key=os.environ.get('EMERGENT_LLM_KEY', ''))

# Create uploads directory if not exists
UPLOAD_DIR = ROOT_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

# Create the main app without a prefix
app = FastAPI()

# Mount static files for uploads
app.mount("/uploads", StaticFiles(directory=str(UPLOAD_DIR)), name="uploads")

# Create a router with the /api prefix
api_router = APIRouter(prefix="/api")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ==================== MODELS ====================

class User(BaseModel):
    model_config = ConfigDict(extra="ignore")
    user_id: str
    email: str
    name: str
    picture: Optional[str] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class UserSession(BaseModel):
    model_config = ConfigDict(extra="ignore")
    user_id: str
    session_token: str
    expires_at: datetime
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

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
    """Get current user from session token in cookie or Authorization header"""
    # Try cookie first
    session_token = request.cookies.get("session_token")
    
    # Fallback to Authorization header
    if not session_token:
        auth_header = request.headers.get("Authorization")
        if auth_header and auth_header.startswith("Bearer "):
            session_token = auth_header.split(" ")[1]
    
    if not session_token:
        raise HTTPException(status_code=401, detail="Not authenticated")
    
    # Find session
    session_doc = await db.user_sessions.find_one(
        {"session_token": session_token},
        {"_id": 0}
    )
    
    if not session_doc:
        raise HTTPException(status_code=401, detail="Invalid session")
    
    # Check expiry
    expires_at = session_doc["expires_at"]
    if isinstance(expires_at, str):
        expires_at = datetime.fromisoformat(expires_at)
    if expires_at.tzinfo is None:
        expires_at = expires_at.replace(tzinfo=timezone.utc)
    if expires_at < datetime.now(timezone.utc):
        raise HTTPException(status_code=401, detail="Session expired")
    
    # Get user
    user_doc = await db.users.find_one(
        {"user_id": session_doc["user_id"]},
        {"_id": 0}
    )
    
    if not user_doc:
        raise HTTPException(status_code=401, detail="User not found")
    
    return User(**user_doc)

# ==================== AUTH ROUTES ====================

@api_router.post("/auth/session")
async def create_session(request: Request, response: Response):
    """Exchange session_id for session_token"""
    body = await request.json()
    session_id = body.get("session_id")
    
    if not session_id:
        raise HTTPException(status_code=400, detail="session_id required")
    
    # Call Emergent auth to get user data
    async with httpx.AsyncClient() as client_http:
        auth_response = await client_http.get(
            "https://demobackend.emergentagent.com/auth/v1/env/oauth/session-data",
            headers={"X-Session-ID": session_id}
        )
    
    if auth_response.status_code != 200:
        raise HTTPException(status_code=401, detail="Invalid session_id")
    
    user_data = auth_response.json()
    
    # Check if user exists
    existing_user = await db.users.find_one(
        {"email": user_data["email"]},
        {"_id": 0}
    )
    
    if existing_user:
        user_id = existing_user["user_id"]
        # Update user info
        await db.users.update_one(
            {"user_id": user_id},
            {"$set": {
                "name": user_data["name"],
                "picture": user_data.get("picture")
            }}
        )
    else:
        # Create new user
        user_id = f"user_{uuid.uuid4().hex[:12]}"
        new_user = {
            "user_id": user_id,
            "email": user_data["email"],
            "name": user_data["name"],
            "picture": user_data.get("picture"),
            "created_at": datetime.now(timezone.utc).isoformat()
        }
        await db.users.insert_one(new_user)
    
    # Create session
    session_token = user_data.get("session_token", f"sess_{uuid.uuid4().hex}")
    expires_at = datetime.now(timezone.utc) + timedelta(days=7)
    
    session_doc = {
        "user_id": user_id,
        "session_token": session_token,
        "expires_at": expires_at.isoformat(),
        "created_at": datetime.now(timezone.utc).isoformat()
    }
    
    # Remove old sessions for this user
    await db.user_sessions.delete_many({"user_id": user_id})
    await db.user_sessions.insert_one(session_doc)
    
    # Set cookie
    response.set_cookie(
        key="session_token",
        value=session_token,
        httponly=True,
        secure=True,
        samesite="none",
        path="/",
        max_age=7 * 24 * 60 * 60  # 7 days
    )
    
    # Get user data to return
    user_doc = await db.users.find_one({"user_id": user_id}, {"_id": 0})
    
    return {"user": user_doc, "session_token": session_token}

@api_router.get("/auth/me")
async def get_me(current_user: User = Depends(get_current_user)):
    """Get current user info"""
    return current_user.model_dump()

@api_router.post("/auth/logout")
async def logout(request: Request, response: Response):
    """Logout user"""
    session_token = request.cookies.get("session_token")
    
    if session_token:
        await db.user_sessions.delete_many({"session_token": session_token})
    
    response.delete_cookie(key="session_token", path="/")
    return {"message": "Logged out"}

# ==================== FILE UPLOAD ====================

@api_router.post("/upload")
async def upload_file(
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user)
):
    """Upload a file (photo or voice note)"""
    # Generate unique filename
    ext = Path(file.filename).suffix if file.filename else ".jpg"
    filename = f"{current_user.user_id}_{uuid.uuid4().hex[:8]}{ext}"
    filepath = UPLOAD_DIR / filename
    
    # Save file
    async with aiofiles.open(filepath, 'wb') as f:
        content = await file.read()
        await f.write(content)
    
    # Return URL
    return {"url": f"/uploads/{filename}", "filename": filename}

@api_router.post("/upload/multiple")
async def upload_multiple_files(
    files: List[UploadFile] = File(...),
    current_user: User = Depends(get_current_user)
):
    """Upload multiple files"""
    urls = []
    for file in files:
        ext = Path(file.filename).suffix if file.filename else ".jpg"
        filename = f"{current_user.user_id}_{uuid.uuid4().hex[:8]}{ext}"
        filepath = UPLOAD_DIR / filename
        
        async with aiofiles.open(filepath, 'wb') as f:
            content = await file.read()
            await f.write(content)
        
        urls.append(f"/uploads/{filename}")
    
    return {"urls": urls}

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
    
    # Generate embedding for RAG
    text_for_embedding = f"{member.name} {member.relationship} {member.relationship_label} {member.notes or ''}"
    try:
        embedding_response = await openai_client.embeddings.create(
            model="text-embedding-3-small",
            input=text_for_embedding
        )
        doc['embedding'] = embedding_response.data[0].embedding
    except Exception as e:
        logger.error(f"Error creating embedding: {e}")
    
    await db.family_members.insert_one(doc)
    
    # Return without embedding
    del doc['embedding'] if 'embedding' in doc else None
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
    
    # Update embedding if name or notes changed
    if 'name' in update_data or 'notes' in update_data or 'relationship' in update_data:
        existing = await db.family_members.find_one(
            {"id": member_id, "user_id": current_user.user_id},
            {"_id": 0}
        )
        if existing:
            name = update_data.get('name', existing.get('name', ''))
            relationship = update_data.get('relationship', existing.get('relationship', ''))
            notes = update_data.get('notes', existing.get('notes', ''))
            text_for_embedding = f"{name} {relationship} {notes}"
            try:
                embedding_response = await openai_client.embeddings.create(
                    model="text-embedding-3-small",
                    input=text_for_embedding
                )
                update_data['embedding'] = embedding_response.data[0].embedding
            except Exception as e:
                logger.error(f"Error creating embedding: {e}")
    
    result = await db.family_members.update_one(
        {"id": member_id, "user_id": current_user.user_id},
        {"$set": update_data}
    )
    
    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="Family member not found")
    
    updated = await db.family_members.find_one(
        {"id": member_id},
        {"_id": 0, "embedding": 0}
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
    
    # Generate embedding for RAG
    people_str = ", ".join(memory.people) if memory.people else ""
    text_for_embedding = f"{memory.title} {memory.date} {memory.location or ''} {memory.description} {people_str}"
    try:
        embedding_response = await openai_client.embeddings.create(
            model="text-embedding-3-small",
            input=text_for_embedding
        )
        doc['embedding'] = embedding_response.data[0].embedding
    except Exception as e:
        logger.error(f"Error creating embedding: {e}")
    
    await db.memories.insert_one(doc)
    
    # Return without embedding
    if 'embedding' in doc:
        del doc['embedding']
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
    
    # Update embedding
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
        text_for_embedding = f"{title} {date} {location} {description} {people_str}"
        try:
            embedding_response = await openai_client.embeddings.create(
                model="text-embedding-3-small",
                input=text_for_embedding
            )
            update_data['embedding'] = embedding_response.data[0].embedding
        except Exception as e:
            logger.error(f"Error creating embedding: {e}")
    
    result = await db.memories.update_one(
        {"id": memory_id, "user_id": current_user.user_id},
        {"$set": update_data}
    )
    
    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="Memory not found")
    
    updated = await db.memories.find_one(
        {"id": memory_id},
        {"_id": 0, "embedding": 0}
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

# ==================== RAG CHAT ====================

def cosine_similarity(a: List[float], b: List[float]) -> float:
    """Calculate cosine similarity between two vectors"""
    a_np = np.array(a)
    b_np = np.array(b)
    return float(np.dot(a_np, b_np) / (np.linalg.norm(a_np) * np.linalg.norm(b_np)))

async def search_similar_content(user_id: str, query: str, top_k: int = 5) -> dict:
    """Search for similar memories and family members using embeddings"""
    # Generate query embedding
    try:
        embedding_response = await openai_client.embeddings.create(
            model="text-embedding-3-small",
            input=query
        )
        query_embedding = embedding_response.data[0].embedding
    except Exception as e:
        logger.error(f"Error creating query embedding: {e}")
        return {"memories": [], "family": []}
    
    # Get all memories with embeddings
    memories = await db.memories.find(
        {"user_id": user_id, "embedding": {"$exists": True}},
        {"_id": 0}
    ).to_list(500)
    
    # Get all family members with embeddings
    family = await db.family_members.find(
        {"user_id": user_id, "embedding": {"$exists": True}},
        {"_id": 0}
    ).to_list(100)
    
    # Calculate similarities for memories
    memory_scores = []
    for mem in memories:
        if mem.get('embedding'):
            score = cosine_similarity(query_embedding, mem['embedding'])
            mem_copy = {k: v for k, v in mem.items() if k != 'embedding'}
            memory_scores.append((score, mem_copy))
    
    # Calculate similarities for family
    family_scores = []
    for fam in family:
        if fam.get('embedding'):
            score = cosine_similarity(query_embedding, fam['embedding'])
            fam_copy = {k: v for k, v in fam.items() if k != 'embedding'}
            family_scores.append((score, fam_copy))
    
    # Sort and get top results
    memory_scores.sort(key=lambda x: x[0], reverse=True)
    family_scores.sort(key=lambda x: x[0], reverse=True)
    
    top_memories = [m[1] for m in memory_scores[:top_k] if m[0] > 0.3]
    top_family = [f[1] for f in family_scores[:top_k] if f[0] > 0.3]
    
    return {"memories": top_memories, "family": top_family}

@api_router.post("/chat")
async def chat_with_assistant(
    chat_request: ChatRequest,
    current_user: User = Depends(get_current_user)
):
    """Chat with AI assistant using RAG"""
    from emergentintegrations.llm.chat import LlmChat, UserMessage
    
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
    
    context_str = "\n".join(context_parts) if context_parts else "No specific memories or family information found."
    
    # System message for the assistant
    system_message = f"""You are a warm, caring assistant helping someone with memory challenges remember their loved ones and precious memories.

The user's name is {current_user.name}. Be patient, kind, and speak in simple, clear language.

Here is information about their family and memories that may help answer their question:

{context_str}

Important guidelines:
1. Always be warm and reassuring
2. Use simple, clear language
3. If you find relevant information, share it naturally and warmly
4. If you don't have specific information, gently say so and offer to help with what you do know
5. Never make up information - only use what's provided
6. Address the user by name occasionally to make it personal
7. If they ask about family, describe relationships clearly (e.g., "Maria is your wife")"""

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
    
    # Initialize LLM chat
    chat = LlmChat(
        api_key=os.environ.get('EMERGENT_LLM_KEY', ''),
        session_id=f"{current_user.user_id}_{chat_request.session_id}",
        system_message=system_message
    ).with_model("openai", "gpt-4.1")
    
    # Send message
    user_message = UserMessage(text=chat_request.message)
    response = await chat.send_message(user_message)
    
    # Save assistant response
    assistant_msg = ChatMessage(
        user_id=current_user.user_id,
        session_id=chat_request.session_id,
        role="assistant",
        content=response
    )
    await db.chat_messages.insert_one(assistant_msg.model_dump())
    
    return {"response": response}

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
