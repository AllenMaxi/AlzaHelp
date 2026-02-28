# AlzaHelp Security & Virality Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Harden all security vulnerabilities and add viral growth features (demo mode, PWA install, i18n, SEO, daily digest, PDF reports, onboarding, referral sharing).

**Architecture:** Security fixes go in `backend/server.py` (middleware, auth, upload/serve endpoints) and `docker-compose.yml`. Viral features add new React components, i18n infrastructure, and backend endpoints. All changes build on existing patterns (shadcn/ui, Tailwind, FastAPI router, GridFS, JWT cookies).

**Tech Stack:** FastAPI, React 19, react-i18next, Tailwind/shadcn, reportlab (PDF), MongoDB/GridFS, OpenAI

---

### Task 1: File Upload Validation

**Files:**
- Modify: `backend/server.py:2311-2382` (upload endpoints)

**Step 1: Add upload constants and validation helper after line 188**

Add after the `DOCTOR_BOT_WRITE_INTENTS` block (line 188):

```python
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
    # Check size
    if len(content) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(status_code=413, detail=f"File too large. Maximum size is {MAX_FILE_SIZE_BYTES // (1024*1024)}MB.")
    # Check MIME type
    content_type = file.content_type or "application/octet-stream"
    if content_type not in ALLOWED_MIME_TYPES:
        raise HTTPException(status_code=415, detail=f"File type '{content_type}' not allowed.")
    # Check extension
    ext = Path(file.filename).suffix.lower() if file.filename else ""
    if ext and ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=415, detail=f"File extension '{ext}' not allowed.")
    # Sanitize filename — strip path components
    return ext or ".bin"
```

**Step 2: Update single upload endpoint (line 2311-2347)**

Replace the body of `upload_file` to call `validate_upload` after reading content:

```python
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
    content = await file.read()
    ext = validate_upload(file, content)
    filename = f"{owner_user_id}_{uuid.uuid4().hex[:8]}{ext}"
    content_type = file.content_type or 'application/octet-stream'

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
    return {"url": f"/api/files/{filename}", "filename": filename, "file_id": str(file_id)}
```

**Step 3: Update multiple upload endpoint (line 2349-2382)**

Same pattern — add `validate_upload` call inside the loop:

```python
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
```

**Step 4: Commit**

```bash
git add backend/server.py
git commit -m "security: add file upload validation (type, size, extension whitelist)"
```

---

### Task 2: File Serving Auth

**Files:**
- Modify: `backend/server.py:2384-2407` (get_file endpoint)

**Step 1: Add auth + ownership check to file serving**

Replace the entire `get_file` function:

```python
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
            # Require authentication
            current_user = await get_current_user(request)

            # Check ownership: user owns the file, or is a linked caregiver/clinician, or is admin
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
```

**Step 2: Commit**

```bash
git add backend/server.py
git commit -m "security: add auth + ownership check to file serving endpoint"
```

---

### Task 3: MongoDB Auth & Docker Hardening

**Files:**
- Modify: `docker-compose.yml`

**Step 1: Update docker-compose.yml**

Replace entire file:

```yaml
services:
  mongo:
    image: mongo:7
    restart: unless-stopped
    volumes:
      - mongo_data:/data/db
    environment:
      MONGO_INITDB_ROOT_USERNAME: ${MONGO_USER:-alzahelp}
      MONGO_INITDB_ROOT_PASSWORD: ${MONGO_PASSWORD:-changeme_in_production}
    # No external port exposure — only accessible via Docker network

  backend:
    build: ./backend
    restart: unless-stopped
    env_file: .env
    environment:
      - MONGO_URL=mongodb://${MONGO_USER:-alzahelp}:${MONGO_PASSWORD:-changeme_in_production}@mongo:27017
      - DB_NAME=memorykeeper
    depends_on:
      - mongo
    ports:
      - "8000:8000"

  frontend:
    build: ./frontend
    restart: unless-stopped
    depends_on:
      - backend
    ports:
      - "80:80"

volumes:
  mongo_data:
```

**Step 2: Commit**

```bash
git add docker-compose.yml
git commit -m "security: add MongoDB auth, remove exposed port"
```

---

### Task 4: Password Strength, Token Revocation, Cookie SameSite, CSP Header

**Files:**
- Modify: `backend/server.py` (multiple locations)

**Step 1: Add password validation function after `get_password_hash` (line 196)**

```python
def validate_password_strength(password: str):
    """Enforce minimum password requirements."""
    if len(password) < 8:
        raise HTTPException(status_code=400, detail="Password must be at least 8 characters long.")
    if not re.search(r'[A-Z]', password):
        raise HTTPException(status_code=400, detail="Password must contain at least one uppercase letter.")
    if not re.search(r'\d', password):
        raise HTTPException(status_code=400, detail="Password must contain at least one number.")
```

**Step 2: Call it in register endpoint (after line 1790, before hashing)**

Insert before `hashed_password = get_password_hash(user_data.password)`:

```python
    validate_password_strength(user_data.password)
```

**Step 3: Add token revocation — update `get_current_user` (line 1401-1444)**

After `payload = jwt.decode(...)` (line 1415), add `iat` check:

```python
        token_iat = payload.get("iat")
```

After fetching `user_doc` (line 1426), add password_changed_at check:

```python
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
```

**Step 4: Include `iat` in token creation (line 198-206)**

In `create_access_token`, add `iat` to the payload. After `to_encode.update({"exp": expire})`:

```python
    to_encode["iat"] = datetime.now(timezone.utc)
```

**Step 5: Add `password_changed_at` to registration (line 1812-1829)**

In the `new_user` dict, add:

```python
        "password_changed_at": datetime.now(timezone.utc).isoformat(),
```

**Step 6: Change all `samesite="none"` to configurable value**

Add near the top constants (after line 77):

```python
COOKIE_SAMESITE = os.environ.get("COOKIE_SAMESITE", "lax")
COOKIE_SECURE = os.environ.get("COOKIE_SECURE", "true").lower() == "true"
```

Then replace all `samesite="none"` with `samesite=COOKIE_SAMESITE` and all `secure=True` (on cookies) with `secure=COOKIE_SECURE`. There are 4 occurrences: lines 1876-1883, 1886-1893, 1977-1980, and any refresh token reissue.

**Step 7: Add CSP header to security middleware (line 7674-7682)**

Add after the Permissions-Policy line:

```python
    response.headers["Content-Security-Policy"] = "default-src 'self'; script-src 'self' 'unsafe-inline' https://us-assets.i.posthog.com https://assets.emergent.sh; style-src 'self' 'unsafe-inline'; img-src 'self' data: blob:; connect-src 'self' https://us.i.posthog.com; font-src 'self'; frame-ancestors 'none'"
```

**Step 8: Add CSP to nginx.conf**

Add after the Referrer-Policy line:

```nginx
    add_header Content-Security-Policy "default-src 'self'; script-src 'self' 'unsafe-inline' https://us-assets.i.posthog.com https://assets.emergent.sh; style-src 'self' 'unsafe-inline'; img-src 'self' data: blob:; connect-src 'self' https://us.i.posthog.com; font-src 'self'; frame-ancestors 'none'" always;
```

**Step 9: Commit**

```bash
git add backend/server.py frontend/nginx.conf
git commit -m "security: password strength, token revocation, samesite=lax, CSP header"
```

---

### Task 5: Demo Mode Backend

**Files:**
- Modify: `backend/server.py` (add demo endpoint + demo guard)

**Step 1: Add demo account seeder and endpoint**

Add a new section after the auth routes (after line 1982). This creates a demo account with pre-populated data:

```python
# ==================== DEMO MODE ====================

DEMO_USER_ID = "demo_patient_001"
DEMO_USER_EMAIL = "demo@alzahelp.app"

async def _seed_demo_account():
    """Create or refresh the demo account with sample data."""
    existing = await db.users.find_one({"user_id": DEMO_USER_ID})
    if existing:
        return  # Already seeded

    # Create demo user
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

    # Seed family members
    family = [
        {"id": f"fam_demo_01", "user_id": DEMO_USER_ID, "name": "Carlos Garcia", "relationship": "Son", "phone": "+1-555-0101", "notes": "Visits every Sunday"},
        {"id": f"fam_demo_02", "user_id": DEMO_USER_ID, "name": "Sofia Garcia", "relationship": "Daughter", "phone": "+1-555-0102", "notes": "Lives nearby, helps with groceries"},
        {"id": f"fam_demo_03", "user_id": DEMO_USER_ID, "name": "Pedro Garcia", "relationship": "Husband", "phone": "+1-555-0100", "notes": "Primary caregiver"},
    ]
    for fm in family:
        fm["created_at"] = datetime.now(timezone.utc).isoformat()
        await db.family_members.insert_one(fm)

    # Seed medications
    meds = [
        {"id": f"med_demo_01", "user_id": DEMO_USER_ID, "name": "Donepezil", "dosage": "10mg", "frequency": "daily", "scheduled_times": ["08:00"], "active": True, "notes": "Take with breakfast"},
        {"id": f"med_demo_02", "user_id": DEMO_USER_ID, "name": "Memantine", "dosage": "20mg", "frequency": "daily", "scheduled_times": ["09:00"], "active": True, "notes": "For cognitive symptoms"},
        {"id": f"med_demo_03", "user_id": DEMO_USER_ID, "name": "Vitamin D", "dosage": "2000 IU", "frequency": "daily", "scheduled_times": ["08:00"], "active": True, "notes": "With food"},
        {"id": f"med_demo_04", "user_id": DEMO_USER_ID, "name": "Melatonin", "dosage": "3mg", "frequency": "daily", "scheduled_times": ["21:00"], "active": True, "notes": "Before bed for sleep"},
    ]
    for med in meds:
        med["created_at"] = datetime.now(timezone.utc).isoformat()
        await db.medications.insert_one(med)

    # Seed memories
    memories = [
        {"id": f"mem_demo_01", "user_id": DEMO_USER_ID, "title": "Wedding Day", "date": "1985-06-15", "year": 1985, "location": "Santiago, Chile", "description": "Pedro and I got married at the beautiful church on Avenida Providencia. Sofia was the flower girl and Carlos carried the rings.", "people": ["Pedro Garcia", "Sofia Garcia", "Carlos Garcia"], "photos": [], "category": "milestone"},
        {"id": f"mem_demo_02", "user_id": DEMO_USER_ID, "title": "Carlos's Graduation", "date": "2010-12-20", "year": 2010, "location": "University of Chile", "description": "Carlos graduated with honors in engineering. The whole family was there. We celebrated at that restaurant by the park.", "people": ["Carlos Garcia", "Pedro Garcia"], "photos": [], "category": "milestone"},
        {"id": f"mem_demo_03", "user_id": DEMO_USER_ID, "title": "Summer at the Beach", "date": "2019-01-10", "year": 2019, "location": "Viña del Mar", "description": "We spent two weeks at the coast. The grandchildren loved building sandcastles. Sofia made her famous empanadas.", "people": ["Sofia Garcia", "Pedro Garcia"], "photos": [], "category": "vacation"},
        {"id": f"mem_demo_04", "user_id": DEMO_USER_ID, "title": "Morning Walk Routine", "date": "2025-11-01", "year": 2025, "location": "Neighborhood Park", "description": "Pedro and I walk every morning around the park. We feed the pigeons near the fountain. It helps me feel calm.", "people": ["Pedro Garcia"], "photos": [], "category": "routine"},
        {"id": f"mem_demo_05", "user_id": DEMO_USER_ID, "title": "Cooking with Sofia", "date": "2025-12-25", "year": 2025, "location": "Home", "description": "Sofia came over for Christmas and we made pastel de choclo together. She said it tasted just like abuela's recipe.", "people": ["Sofia Garcia"], "photos": [], "category": "family"},
    ]
    for mem in memories:
        mem["created_at"] = datetime.now(timezone.utc).isoformat()
        mem["updated_at"] = datetime.now(timezone.utc).isoformat()
        mem["search_text"] = f"{mem['title']} {mem['date']} {mem.get('location', '')} {mem['description']} {', '.join(mem['people'])}".lower()
        await db.memories.insert_one(mem)

    # Seed mood checkins (last 7 days)
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

    # Seed reminders
    reminders = [
        {"id": "rem_demo_01", "user_id": DEMO_USER_ID, "title": "Morning walk with Pedro", "time": "09:00", "active": True},
        {"id": "rem_demo_02", "user_id": DEMO_USER_ID, "title": "Call Sofia", "time": "15:00", "active": True},
        {"id": "rem_demo_03", "user_id": DEMO_USER_ID, "title": "Water the plants", "time": "10:00", "active": True},
    ]
    for rem in reminders:
        rem["created_at"] = datetime.now(timezone.utc).isoformat()
        await db.reminders.insert_one(rem)


def is_demo_user(current_user: User) -> bool:
    """Check if the current user is the demo account."""
    return getattr(current_user, 'user_id', '') == DEMO_USER_ID


def guard_demo_write(current_user: User):
    """Block write operations for demo users."""
    if is_demo_user(current_user):
        raise HTTPException(status_code=403, detail="Demo mode is read-only. Sign up to save your own data!")


@api_router.post("/auth/demo")
@rate_limit("20/minute")
async def start_demo(request: Request, response: Response):
    """Start a demo session with pre-populated data."""
    await _seed_demo_account()

    access_token_expires = timedelta(hours=2)  # Short-lived demo session
    access_token = create_access_token(
        data={
            "sub": DEMO_USER_EMAIL,
            "user_id": DEMO_USER_ID,
            "role": "patient"
        },
        expires_delta=access_token_expires
    )

    response.set_cookie(
        key="access_token",
        value=access_token,
        httponly=True,
        secure=COOKIE_SECURE,
        samesite=COOKIE_SAMESITE,
        path="/",
        max_age=2 * 60 * 60  # 2 hours
    )

    return {"message": "Demo started", "user": {"user_id": DEMO_USER_ID, "name": "Maria Garcia", "role": "patient", "is_demo": True}}
```

**Step 2: Add `guard_demo_write` calls to all write endpoints**

Add `guard_demo_write(current_user)` as the first line in all POST/PUT/DELETE endpoints that accept `current_user: User = Depends(get_current_user)`. Key endpoints:

- `create_memory` (line ~2513)
- `update_memory` (line ~2542)
- `delete_memory` (line ~2582)
- `create_family_member` (line ~2420)
- `update_family_member` (line ~2448)
- `delete_family_member` (line ~2487)
- `create_reminder` (line ~2608)
- `delete_reminder` (line ~2652)
- `upload_file` (line ~2311)
- `upload_multiple_files` (line ~2349)
- `create_medication` (line ~5132)
- `update_medication` (line ~5157)
- `delete_medication` (line ~5178)
- `delete_account` (line ~1910)

Pattern for each:

```python
async def create_memory(memory: MemoryCreate, current_user: User = Depends(get_current_user)):
    guard_demo_write(current_user)
    # ... rest of function
```

**Step 3: Seed demo on startup**

Add to the app startup (around line 7660, before `app.include_router`):

```python
@app.on_event("startup")
async def startup_seed():
    await _seed_demo_account()
```

**Step 4: Commit**

```bash
git add backend/server.py
git commit -m "feat: add demo mode with pre-populated patient data"
```

---

### Task 6: Demo Mode Frontend

**Files:**
- Modify: `frontend/src/pages/LandingPage.jsx` (add Try Demo button)
- Modify: `frontend/src/services/api.js` (add demo API call)
- Modify: `frontend/src/context/AuthContext.jsx` (handle demo user)
- Modify: `frontend/src/pages/DashboardPage.jsx` (show demo banner)

**Step 1: Add `startDemo` to api.js**

Add to the auth section of api.js:

```javascript
export const authApi = {
  // ... existing methods ...
  startDemo: () => fetchWithAuth('/api/auth/demo', { method: 'POST' }),
};
```

**Step 2: Add demo support to AuthContext**

Add `isDemo` state and `startDemo` function. The demo flag can be derived from the user object having `is_demo: true`.

**Step 3: Add "Try Demo" button to LandingPage hero**

After the existing "Get Started" and "Learn More" buttons (around line 77-93), add:

```jsx
<Button
  variant="outline"
  size="lg"
  className="text-lg px-8 py-6 border-violet-300 text-violet-700 hover:bg-violet-50"
  onClick={async () => {
    try {
      const res = await authApi.startDemo();
      navigate('/dashboard');
    } catch (err) {
      console.error('Demo failed:', err);
    }
  }}
>
  Try Demo — No Signup
</Button>
```

**Step 4: Add demo banner to DashboardPage**

At the top of the dashboard, show a dismissible banner when user `is_demo`:

```jsx
{user?.is_demo && (
  <div className="bg-amber-50 dark:bg-amber-900/20 border border-amber-200 dark:border-amber-800 rounded-lg p-3 mx-4 mt-4 flex items-center justify-between">
    <p className="text-sm text-amber-800 dark:text-amber-200">
      You're exploring AlzaHelp in demo mode. <strong>Sign up free</strong> to save your own data.
    </p>
    <Button size="sm" onClick={() => navigate('/login')}>Sign Up</Button>
  </div>
)}
```

**Step 5: Commit**

```bash
git add frontend/src/pages/LandingPage.jsx frontend/src/services/api.js frontend/src/context/AuthContext.jsx frontend/src/pages/DashboardPage.jsx
git commit -m "feat: add demo mode frontend (try without signup)"
```

---

### Task 7: PWA Install Prompt

**Files:**
- Create: `frontend/src/components/InstallPrompt.jsx`
- Modify: `frontend/src/pages/DashboardPage.jsx` (include InstallPrompt)

**Step 1: Create InstallPrompt component**

```jsx
import React, { useState, useEffect } from 'react';
import { Button } from '@/components/ui/button';
import { X, Download } from 'lucide-react';

export const InstallPrompt = () => {
  const [deferredPrompt, setDeferredPrompt] = useState(null);
  const [showBanner, setShowBanner] = useState(false);

  useEffect(() => {
    const dismissed = localStorage.getItem('alzahelp_install_dismissed');
    if (dismissed) {
      const dismissedAt = parseInt(dismissed, 10);
      if (Date.now() - dismissedAt < 7 * 24 * 60 * 60 * 1000) return; // 7 days
    }

    const handler = (e) => {
      e.preventDefault();
      setDeferredPrompt(e);
      setShowBanner(true);
    };
    window.addEventListener('beforeinstallprompt', handler);
    return () => window.removeEventListener('beforeinstallprompt', handler);
  }, []);

  const handleInstall = async () => {
    if (!deferredPrompt) return;
    deferredPrompt.prompt();
    const { outcome } = await deferredPrompt.userChoice;
    setDeferredPrompt(null);
    setShowBanner(false);
    if (outcome === 'accepted') {
      localStorage.setItem('alzahelp_installed', 'true');
    }
  };

  const handleDismiss = () => {
    setShowBanner(false);
    localStorage.setItem('alzahelp_install_dismissed', Date.now().toString());
  };

  if (!showBanner) return null;

  return (
    <div className="fixed bottom-0 left-0 right-0 z-50 p-4 bg-violet-600 text-white shadow-lg">
      <div className="container mx-auto flex items-center justify-between gap-4">
        <div className="flex items-center gap-3">
          <Download className="h-5 w-5 shrink-0" />
          <p className="text-sm font-medium">Install AlzaHelp for quick access and offline support</p>
        </div>
        <div className="flex items-center gap-2">
          <Button size="sm" variant="secondary" onClick={handleInstall}>
            Install
          </Button>
          <button onClick={handleDismiss} className="p-1 hover:bg-violet-500 rounded">
            <X className="h-4 w-4" />
          </button>
        </div>
      </div>
    </div>
  );
};
```

**Step 2: Add to DashboardPage**

Import and render at the bottom of the dashboard layout:

```jsx
import { InstallPrompt } from '@/components/InstallPrompt';
// ... in the return:
<InstallPrompt />
```

**Step 3: Commit**

```bash
git add frontend/src/components/InstallPrompt.jsx frontend/src/pages/DashboardPage.jsx
git commit -m "feat: add PWA install prompt banner"
```

---

### Task 8: i18n Setup (English + Spanish)

**Files:**
- Create: `frontend/src/i18n/index.js` (i18next config)
- Create: `frontend/src/i18n/en.json` (English translations)
- Create: `frontend/src/i18n/es.json` (Spanish translations)
- Modify: `frontend/src/index.js` (import i18n)
- Modify: `frontend/src/components/layout/Header.jsx` (language switcher)
- Modify: `frontend/src/pages/LandingPage.jsx` (use translations)
- Modify: `frontend/src/pages/DashboardPage.jsx` (use translations)
- Modify: `frontend/src/components/OnboardingWizard.jsx` (use translations)
- Modify: `frontend/package.json` (add dependencies)

**Step 1: Install i18n dependencies**

```bash
cd frontend && yarn add react-i18next i18next i18next-browser-languagedetector
```

**Step 2: Create i18n config**

Create `frontend/src/i18n/index.js`:

```javascript
import i18n from 'i18next';
import { initReactI18next } from 'react-i18next';
import LanguageDetector from 'i18next-browser-languagedetector';
import en from './en.json';
import es from './es.json';

i18n
  .use(LanguageDetector)
  .use(initReactI18next)
  .init({
    resources: { en: { translation: en }, es: { translation: es } },
    fallbackLng: 'en',
    interpolation: { escapeValue: false },
    detection: {
      order: ['localStorage', 'navigator'],
      caches: ['localStorage'],
    },
  });

export default i18n;
```

**Step 3: Create English translations `frontend/src/i18n/en.json`**

```json
{
  "landing": {
    "badge": "Caring technology for memory challenges",
    "title": "Remember what",
    "titleHighlight": "matters most",
    "subtitle": "AlzaHelp helps people with Alzheimer's and memory challenges stay connected to their loved ones, medications, and daily routines through voice AI and smart reminders.",
    "getStarted": "Get Started",
    "tryDemo": "Try Demo — No Signup",
    "learnMore": "Learn More",
    "features": {
      "voice": { "title": "Voice Companion", "desc": "Say \"Hey Memory\" to get medication reminders, hear your daily schedule, or simply have a conversation." },
      "medication": { "title": "Medication Tracking", "desc": "Never miss a dose with smart reminders, voice-powered intake logging, and real-time caregiver alerts." },
      "family": { "title": "Family Connections", "desc": "Keep photos, stories, and contact info for loved ones always at your fingertips. Ask the AI who anyone is." },
      "safety": { "title": "Safety Net", "desc": "Geofencing, SOS alerts, and fall detection keep caregivers informed and patients safe." }
    },
    "roles": {
      "patients": { "title": "For Patients", "desc": "Talk to your AI companion, play memory games, and stay connected to your daily routine and loved ones." },
      "caregivers": { "title": "For Doctors & Caregivers", "desc": "Manage medications and care instructions from the portal or via Telegram/WhatsApp chatbot." },
      "ai": { "title": "AI-Powered", "desc": "Semantic search understands context. Ask about memories naturally and get intelligent, caring responses." }
    },
    "pricing": {
      "title": "Simple, transparent pricing",
      "free": { "name": "Free", "price": "$0", "period": "forever", "features": ["All patient features", "Voice AI companion", "Medication tracking", "Memory games", "Up to 3 linked caregivers"] },
      "premium": { "name": "Premium", "price": "$9.99", "period": "/month", "features": ["Everything in Free", "Telegram/WhatsApp bot", "SMS medication alerts", "Remote monitoring", "Up to 20 linked caregivers", "Priority support"] }
    },
    "cta": "Start preserving memories today",
    "ctaButton": "Create Free Account"
  },
  "nav": {
    "home": "Home",
    "memories": "My Memories",
    "logout": "Logout",
    "profile": "Profile"
  },
  "dashboard": {
    "welcome": "Welcome back",
    "demoBanner": "You're exploring AlzaHelp in demo mode.",
    "demoSignup": "Sign up free to save your own data.",
    "tabs": {
      "companion": "AI Companion",
      "family": "Family",
      "timeline": "Timeline",
      "games": "Games",
      "reminders": "Reminders",
      "medications": "Medications",
      "mood": "Mood & Behavior",
      "navigation": "Navigation",
      "assistant": "Chat",
      "caregiver": "Caregiver Portal",
      "admin": "Admin"
    }
  },
  "onboarding": {
    "title": "Welcome to AlzaHelp!",
    "subtitle": "Let's set up a couple of things. This takes about 1 minute.",
    "step1": "Welcome",
    "step2": "Add a Photo",
    "step3": "Add Medication",
    "step4": "Add Family",
    "step5": "Install App",
    "skip": "Skip",
    "skipAll": "Skip Setup",
    "next": "Next",
    "finish": "Finish Setup",
    "photoPrompt": "Upload a photo of yourself or your family",
    "medName": "Medication Name",
    "medDosage": "Dosage",
    "familyName": "Family Member Name",
    "familyRelation": "Relationship",
    "installTitle": "Install AlzaHelp",
    "installDesc": "Add AlzaHelp to your home screen for quick access",
    "installButton": "Install App"
  },
  "common": {
    "save": "Save",
    "cancel": "Cancel",
    "delete": "Delete",
    "edit": "Edit",
    "add": "Add",
    "search": "Search",
    "loading": "Loading...",
    "error": "Something went wrong",
    "success": "Success",
    "confirm": "Confirm",
    "back": "Back",
    "signUp": "Sign Up",
    "login": "Log In"
  },
  "language": {
    "en": "English",
    "es": "Español"
  }
}
```

**Step 4: Create Spanish translations `frontend/src/i18n/es.json`**

```json
{
  "landing": {
    "badge": "Tecnología de cuidado para desafíos de memoria",
    "title": "Recuerda lo que",
    "titleHighlight": "más importa",
    "subtitle": "AlzaHelp ayuda a personas con Alzheimer y desafíos de memoria a mantenerse conectados con sus seres queridos, medicamentos y rutinas diarias mediante IA de voz y recordatorios inteligentes.",
    "getStarted": "Comenzar",
    "tryDemo": "Probar Demo — Sin Registro",
    "learnMore": "Saber Más",
    "features": {
      "voice": { "title": "Compañero de Voz", "desc": "Di \"Hey Memory\" para recibir recordatorios de medicamentos, escuchar tu horario diario o simplemente conversar." },
      "medication": { "title": "Control de Medicamentos", "desc": "Nunca pierdas una dosis con recordatorios inteligentes, registro por voz y alertas en tiempo real para cuidadores." },
      "family": { "title": "Conexiones Familiares", "desc": "Mantén fotos, historias e información de contacto de tus seres queridos siempre a tu alcance." },
      "safety": { "title": "Red de Seguridad", "desc": "Geovallas, alertas SOS y detección de caídas mantienen informados a los cuidadores y seguros a los pacientes." }
    },
    "roles": {
      "patients": { "title": "Para Pacientes", "desc": "Habla con tu compañero IA, juega juegos de memoria y mantente conectado con tu rutina diaria y seres queridos." },
      "caregivers": { "title": "Para Doctores y Cuidadores", "desc": "Gestiona medicamentos e instrucciones de cuidado desde el portal o por chatbot de Telegram/WhatsApp." },
      "ai": { "title": "Potenciado por IA", "desc": "La búsqueda semántica entiende el contexto. Pregunta sobre recuerdos naturalmente y obtén respuestas inteligentes." }
    },
    "pricing": {
      "title": "Precios simples y transparentes",
      "free": { "name": "Gratis", "price": "$0", "period": "para siempre", "features": ["Todas las funciones del paciente", "Compañero de voz IA", "Control de medicamentos", "Juegos de memoria", "Hasta 3 cuidadores vinculados"] },
      "premium": { "name": "Premium", "price": "$9.99", "period": "/mes", "features": ["Todo lo de Gratis", "Bot de Telegram/WhatsApp", "Alertas SMS de medicamentos", "Monitoreo remoto", "Hasta 20 cuidadores vinculados", "Soporte prioritario"] }
    },
    "cta": "Comienza a preservar recuerdos hoy",
    "ctaButton": "Crear Cuenta Gratis"
  },
  "nav": {
    "home": "Inicio",
    "memories": "Mis Recuerdos",
    "logout": "Cerrar Sesión",
    "profile": "Perfil"
  },
  "dashboard": {
    "welcome": "Bienvenido de nuevo",
    "demoBanner": "Estás explorando AlzaHelp en modo demo.",
    "demoSignup": "Regístrate gratis para guardar tus propios datos.",
    "tabs": {
      "companion": "Compañero IA",
      "family": "Familia",
      "timeline": "Línea de Tiempo",
      "games": "Juegos",
      "reminders": "Recordatorios",
      "medications": "Medicamentos",
      "mood": "Ánimo y Conducta",
      "navigation": "Navegación",
      "assistant": "Chat",
      "caregiver": "Portal de Cuidador",
      "admin": "Administración"
    }
  },
  "onboarding": {
    "title": "¡Bienvenido a AlzaHelp!",
    "subtitle": "Configuremos un par de cosas. Esto toma aproximadamente 1 minuto.",
    "step1": "Bienvenida",
    "step2": "Añadir Foto",
    "step3": "Añadir Medicamento",
    "step4": "Añadir Familiar",
    "step5": "Instalar App",
    "skip": "Omitir",
    "skipAll": "Omitir Configuración",
    "next": "Siguiente",
    "finish": "Finalizar",
    "photoPrompt": "Sube una foto tuya o de tu familia",
    "medName": "Nombre del Medicamento",
    "medDosage": "Dosis",
    "familyName": "Nombre del Familiar",
    "familyRelation": "Relación",
    "installTitle": "Instalar AlzaHelp",
    "installDesc": "Agrega AlzaHelp a tu pantalla de inicio para acceso rápido",
    "installButton": "Instalar App"
  },
  "common": {
    "save": "Guardar",
    "cancel": "Cancelar",
    "delete": "Eliminar",
    "edit": "Editar",
    "add": "Añadir",
    "search": "Buscar",
    "loading": "Cargando...",
    "error": "Algo salió mal",
    "success": "Éxito",
    "confirm": "Confirmar",
    "back": "Volver",
    "signUp": "Registrarse",
    "login": "Iniciar Sesión"
  },
  "language": {
    "en": "English",
    "es": "Español"
  }
}
```

**Step 5: Import i18n in index.js**

Add `import '@/i18n';` as the first import in `frontend/src/index.js`.

**Step 6: Add language switcher to Header.jsx**

Add a globe button dropdown next to the dark mode toggle:

```jsx
import { useTranslation } from 'react-i18next';
import { Globe } from 'lucide-react';

// Inside Header component:
const { i18n } = useTranslation();

// In the header actions area, add:
<DropdownMenu>
  <DropdownMenuTrigger asChild>
    <Button variant="ghost" size="icon">
      <Globe className="h-5 w-5" />
    </Button>
  </DropdownMenuTrigger>
  <DropdownMenuContent align="end">
    <DropdownMenuItem onClick={() => i18n.changeLanguage('en')}>
      English {i18n.language === 'en' && '✓'}
    </DropdownMenuItem>
    <DropdownMenuItem onClick={() => i18n.changeLanguage('es')}>
      Español {i18n.language === 'es' && '✓'}
    </DropdownMenuItem>
  </DropdownMenuContent>
</DropdownMenu>
```

**Step 7: Update LandingPage to use translations**

Replace all hardcoded strings with `t('landing.xxx')` calls. Import `useTranslation`:

```jsx
import { useTranslation } from 'react-i18next';
// Inside component:
const { t } = useTranslation();
// Replace: "Remember what" → {t('landing.title')}
// Replace: "matters most" → {t('landing.titleHighlight')}
// etc.
```

**Step 8: Update DashboardPage tab labels to use translations**

Use `t('dashboard.tabs.companion')` etc. for tab labels.

**Step 9: Commit**

```bash
git add frontend/src/i18n/ frontend/src/index.js frontend/src/components/layout/Header.jsx frontend/src/pages/LandingPage.jsx frontend/src/pages/DashboardPage.jsx frontend/src/components/OnboardingWizard.jsx frontend/package.json frontend/yarn.lock
git commit -m "feat: add i18n with English + Spanish translations"
```

---

### Task 9: Enhanced SEO

**Files:**
- Modify: `frontend/public/index.html` (meta tags, JSON-LD, canonical)
- Create: `frontend/public/robots.txt`
- Create: `frontend/public/sitemap.xml`

**Step 1: Add to index.html `<head>`**

Add after existing meta tags:

```html
<meta property="og:url" content="https://alzahelp.app" />
<meta property="og:image" content="https://alzahelp.app/og-image.png" />
<meta property="og:image:width" content="1200" />
<meta property="og:image:height" content="630" />
<meta name="twitter:title" content="AlzaHelp - Remember What Matters Most" />
<meta name="twitter:description" content="AI-powered memory assistance for Alzheimer's care." />
<meta name="twitter:image" content="https://alzahelp.app/og-image.png" />
<link rel="canonical" href="https://alzahelp.app" />
<script type="application/ld+json">
{
  "@context": "https://schema.org",
  "@type": "SoftwareApplication",
  "name": "AlzaHelp",
  "applicationCategory": "HealthApplication",
  "operatingSystem": "Web",
  "description": "AI-powered memory assistance for Alzheimer's care. Voice companion, medication tracking, family connections, and safety monitoring.",
  "offers": [
    { "@type": "Offer", "price": "0", "priceCurrency": "USD", "description": "Free tier" },
    { "@type": "Offer", "price": "9.99", "priceCurrency": "USD", "description": "Premium monthly" }
  ],
  "featureList": "Voice AI Companion, Medication Tracking, Family Connections, Safety Monitoring, Memory Games"
}
</script>
```

**Step 2: Create robots.txt**

```
User-agent: *
Allow: /
Sitemap: https://alzahelp.app/sitemap.xml

Disallow: /dashboard
Disallow: /api/
```

**Step 3: Create sitemap.xml**

```xml
<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaschem.org/schemas/sitemap/0.9">
  <url><loc>https://alzahelp.app/</loc><changefreq>weekly</changefreq><priority>1.0</priority></url>
  <url><loc>https://alzahelp.app/login</loc><changefreq>monthly</changefreq><priority>0.5</priority></url>
  <url><loc>https://alzahelp.app/privacy</loc><changefreq>monthly</changefreq><priority>0.3</priority></url>
  <url><loc>https://alzahelp.app/terms</loc><changefreq>monthly</changefreq><priority>0.3</priority></url>
</urlset>
```

**Step 4: Remove Emergent badge from index.html**

Remove the `<a id="emergent-badge" ...>` block (lines 55-101) — this is a third-party watermark that shouldn't be in production.

**Step 5: Commit**

```bash
git add frontend/public/index.html frontend/public/robots.txt frontend/public/sitemap.xml
git commit -m "feat: add SEO enhancements (JSON-LD, OG tags, sitemap, robots.txt)"
```

---

### Task 10: Daily Caregiver Digest

**Files:**
- Modify: `backend/server.py` (add digest endpoint + scheduled task)

**Step 1: Add daily digest endpoint**

Add after the care instructions section:

```python
@api_router.get("/care/patients/{patient_id}/daily-digest")
async def daily_digest(patient_id: str, current_user: User = Depends(get_current_user)):
    """Generate an AI-powered daily summary for a caregiver."""
    await resolve_target_user_id(current_user, patient_id)

    today = datetime.now(timezone.utc).date().isoformat()
    patient = await db.users.find_one({"user_id": patient_id}, {"_id": 0, "name": 1})
    patient_name = patient.get("name", "Patient") if patient else "Patient"

    # Gather today's data
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

    # Calculate adherence
    total_doses = sum(len(m.get("scheduled_times", [])) for m in meds)
    taken_doses = len(intakes)
    adherence = f"{taken_doses}/{total_doses}" if total_doses > 0 else "No medications scheduled"

    mood_summary = "No check-in today"
    if mood:
        latest = mood[-1]
        mood_summary = f"Mood: {latest.get('mood_score', '?')}/3, Energy: {latest.get('energy_level', '?')}/3"

    alert_summary = f"{len(alerts)} safety alert(s)" if alerts else "No safety alerts"

    # Generate AI summary
    prompt = f"""Summarize this patient's day in 3 concise, caring sentences for their caregiver. Patient name: {patient_name}.

Medication adherence: {adherence}
{mood_summary}
{alert_summary}

Be warm but factual. If adherence is low, mention it gently."""

    try:
        client = get_openai_client()
        resp = await client.chat.completions.create(
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
```

**Step 2: Commit**

```bash
git add backend/server.py
git commit -m "feat: add daily caregiver digest endpoint with AI summary"
```

---

### Task 11: PDF Care Reports

**Files:**
- Modify: `backend/server.py` (add PDF generation endpoint)
- Modify: `backend/requirements.txt` (add reportlab)

**Step 1: Add reportlab to requirements.txt**

```
reportlab>=4.0.0
```

**Step 2: Add PDF import near top of server.py (after existing imports)**

```python
try:
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    from reportlab.lib import colors
    _HAS_REPORTLAB = True
except ImportError:
    _HAS_REPORTLAB = False
```

**Step 3: Add PDF endpoint**

```python
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

    # Gather data
    meds = await db.medications.find({"user_id": patient_id, "active": True}, {"_id": 0}).to_list(50)
    intakes = await db.medication_intake_logs.find({"user_id": patient_id, "taken_at": {"$gte": cutoff}}, {"_id": 0}).to_list(5000)
    moods = await db.mood_checkins.find({"user_id": patient_id, "created_at": {"$gte": cutoff}}, {"_id": 0}).to_list(500)
    alerts = await db.safety_alerts.find({"user_id": patient_id, "created_at": {"$gte": cutoff}}, {"_id": 0}).to_list(500)

    # Build PDF
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, topMargin=0.75*inch, bottomMargin=0.75*inch)
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle('ReportTitle', parent=styles['Title'], fontSize=18, spaceAfter=20)
    heading_style = ParagraphStyle('SectionHead', parent=styles['Heading2'], fontSize=14, spaceAfter=10, textColor=colors.HexColor('#7c3aed'))

    story = []
    story.append(Paragraph(f"AlzaHelp Care Report — {patient_name}", title_style))
    story.append(Paragraph(f"Report period: Last {days} days | Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d')}", styles['Normal']))
    story.append(Spacer(1, 20))

    # Medications section
    story.append(Paragraph("Medications", heading_style))
    if meds:
        med_data = [["Medication", "Dosage", "Schedule"]]
        for m in meds:
            med_data.append([m.get("name", ""), m.get("dosage", ""), ", ".join(m.get("scheduled_times", []))])
        t = Table(med_data, colWidths=[2.5*inch, 1.5*inch, 2*inch])
        t.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#7c3aed')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f5f3ff')]),
        ]))
        story.append(t)
    else:
        story.append(Paragraph("No active medications.", styles['Normal']))
    story.append(Spacer(1, 15))

    # Adherence summary
    total_doses = len(intakes)
    story.append(Paragraph("Medication Adherence", heading_style))
    story.append(Paragraph(f"Total recorded intakes in period: {total_doses}", styles['Normal']))
    story.append(Spacer(1, 15))

    # Mood summary
    story.append(Paragraph("Mood & Wellbeing", heading_style))
    if moods:
        avg_mood = sum(m.get("mood_score", 0) for m in moods) / len(moods)
        avg_energy = sum(m.get("energy_level", 0) for m in moods) / len(moods)
        story.append(Paragraph(f"Check-ins: {len(moods)} | Avg Mood: {avg_mood:.1f}/3 | Avg Energy: {avg_energy:.1f}/3", styles['Normal']))
    else:
        story.append(Paragraph("No mood check-ins recorded.", styles['Normal']))
    story.append(Spacer(1, 15))

    # Safety alerts
    story.append(Paragraph("Safety Events", heading_style))
    story.append(Paragraph(f"Alerts in period: {len(alerts)}", styles['Normal']))
    if alerts:
        for a in alerts[:10]:
            story.append(Paragraph(f"• {a.get('type', 'alert')} — {a.get('created_at', '')[:10]}: {a.get('message', '')[:100]}", styles['Normal']))
    story.append(Spacer(1, 20))

    story.append(Paragraph("Generated by AlzaHelp — alzahelp.app", styles['Normal']))

    doc.build(story)
    buffer.seek(0)

    return StreamingResponse(
        buffer,
        media_type="application/pdf",
        headers={"Content-Disposition": f"attachment; filename=alzahelp_report_{patient_name.replace(' ', '_')}_{datetime.now(timezone.utc).strftime('%Y%m%d')}.pdf"}
    )
```

**Step 3: Commit**

```bash
git add backend/server.py backend/requirements.txt
git commit -m "feat: add PDF care report generation for doctor visits"
```

---

### Task 12: Improved Onboarding (5 steps)

**Files:**
- Modify: `frontend/src/components/OnboardingWizard.jsx` (expand to 5 steps)

**Step 1: Expand onboarding to 5 steps**

Rewrite OnboardingWizard with these steps:
1. Welcome — greeting + explanation
2. Add Photo — upload a profile or family photo (uses existing uploadApi)
3. Add Medication — name + dosage (existing)
4. Add Family Member — name + relationship (existing)
5. Install App — PWA install prompt (reuse beforeinstallprompt logic)

Each step gets a checkmark animation on completion. The progress bar shows 5 dots. All steps remain skippable.

Use `useTranslation` for all text strings (from Task 8 translations).

**Step 2: Commit**

```bash
git add frontend/src/components/OnboardingWizard.jsx
git commit -m "feat: expand onboarding to 5 steps with photo upload and PWA install"
```

---

### Task 13: Referral Sharing

**Files:**
- Modify: `backend/server.py` (add referral endpoints)
- Modify: `frontend/src/services/api.js` (add referral API)
- Modify: `frontend/src/components/sections/CaregiverPortalSection.jsx` (add share card)

**Step 1: Add referral endpoints to backend**

```python
@api_router.post("/referral/generate")
async def generate_referral(current_user: User = Depends(get_current_user)):
    """Generate a unique referral code for the current user."""
    guard_demo_write(current_user)

    # Check if user already has a referral code
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
```

**Step 2: Update register endpoint to accept referral_code**

In the `UserCreate` model, add:

```python
    referral_code: Optional[str] = None
```

In the register function, after creating the user doc, add:

```python
    if user_data.referral_code:
        referrer = await db.users.find_one({"referral_code": user_data.referral_code})
        if referrer:
            new_user["referred_by"] = user_data.referral_code
```

**Step 3: Add referral API to frontend api.js**

```javascript
export const referralApi = {
  generate: () => fetchWithAuth('/api/referral/generate', { method: 'POST' }),
  stats: () => fetchWithAuth('/api/referral/stats'),
};
```

**Step 4: Add "Invite Family" share card to CaregiverPortalSection**

Add a card with a share button that uses `navigator.share()` (with clipboard fallback):

```jsx
const handleShare = async () => {
  const { code } = await referralApi.generate();
  const url = `${window.location.origin}/login?ref=${code}`;
  const text = t('referral.shareText', { url });

  if (navigator.share) {
    await navigator.share({ title: 'AlzaHelp', text, url });
  } else {
    await navigator.clipboard.writeText(url);
    toast.success(t('referral.copied'));
  }
};
```

**Step 5: Commit**

```bash
git add backend/server.py frontend/src/services/api.js frontend/src/components/sections/CaregiverPortalSection.jsx
git commit -m "feat: add referral sharing system with deeplinks"
```

---

### Task 14: Add Referral Translations + Final i18n Pass

**Files:**
- Modify: `frontend/src/i18n/en.json`
- Modify: `frontend/src/i18n/es.json`

**Step 1: Add referral translations**

English:
```json
"referral": {
  "title": "Invite Your Family",
  "description": "Share AlzaHelp with family members who care for your loved one",
  "shareButton": "Share Invite Link",
  "copied": "Link copied to clipboard!",
  "shareText": "I use AlzaHelp to help care for my loved one. Join me: {{url}}",
  "count": "{{count}} people joined through your link"
}
```

Spanish:
```json
"referral": {
  "title": "Invita a Tu Familia",
  "description": "Comparte AlzaHelp con familiares que cuidan a tu ser querido",
  "shareButton": "Compartir Enlace",
  "copied": "¡Enlace copiado!",
  "shareText": "Uso AlzaHelp para cuidar a mi ser querido. Únete: {{url}}",
  "count": "{{count}} personas se unieron con tu enlace"
}
```

**Step 2: Commit**

```bash
git add frontend/src/i18n/en.json frontend/src/i18n/es.json
git commit -m "feat: add referral translations (en + es)"
```

---

### Task 15: Final Integration & Cleanup

**Files:**
- Modify: `frontend/src/pages/LandingPage.jsx` (language switcher on landing)
- Modify: `frontend/src/services/api.js` (add digest + report API)

**Step 1: Add digest and report APIs to frontend**

```javascript
export const careReportApi = {
  getDailyDigest: (patientId) => fetchWithAuth(`/api/care/patients/${patientId}/daily-digest`),
  downloadReport: (patientId, days = 30) =>
    fetch(`${BASE_URL}/api/care/patients/${patientId}/report?days=${days}`, { credentials: 'include' })
      .then(r => r.blob()),
};
```

**Step 2: Add language switcher to LandingPage header**

Add a small EN|ES toggle in the landing page navigation area using `useTranslation`.

**Step 3: Final commit**

```bash
git add frontend/src/pages/LandingPage.jsx frontend/src/services/api.js
git commit -m "feat: add digest/report API + landing page language switcher"
```
