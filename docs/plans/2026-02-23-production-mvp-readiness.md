# Production MVP Readiness Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make AlzaHelp production-ready with Docker containerization, database indexes, security hardening, error tracking, legal pages, and CI/CD.

**Architecture:** Add Docker multi-stage builds for backend (Python/FastAPI) and frontend (React/nginx). Create MongoDB indexes on startup via a `ensure_indexes()` function. Add security middleware. Add Sentry for error tracking. Create static legal pages. Add GitHub Actions for CI.

**Tech Stack:** Docker, nginx, GitHub Actions, MongoDB indexes, Sentry SDK, FastAPI security middleware

---

### Task 1: Dockerfile for Backend

**Files:**
- Create: `backend/Dockerfile`
- Create: `backend/.dockerignore`

**Step 1: Create backend .dockerignore**

```
__pycache__
*.pyc
.env
.env.*
tests/
.pytest_cache
.mypy_cache
```

**Step 2: Create backend Dockerfile**

```dockerfile
FROM python:3.11-slim AS base

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
```

**Step 3: Verify**

Run: `docker build -t alzahelp-backend ./backend` (if Docker available, otherwise just verify file syntax)

**Step 4: Commit**

```bash
git add backend/Dockerfile backend/.dockerignore
git commit -m "feat: add backend Dockerfile"
```

---

### Task 2: Dockerfile for Frontend + nginx

**Files:**
- Create: `frontend/Dockerfile`
- Create: `frontend/.dockerignore`
- Create: `frontend/nginx.conf`

**Step 1: Create frontend .dockerignore**

```
node_modules
build
.env
.env.*
```

**Step 2: Create nginx.conf for SPA routing**

```nginx
server {
    listen 80;
    server_name _;

    root /usr/share/nginx/html;
    index index.html;

    # SPA fallback
    location / {
        try_files $uri $uri/ /index.html;
    }

    # Proxy API requests to backend
    location /api/ {
        proxy_pass http://backend:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    location /health {
        proxy_pass http://backend:8000;
    }

    # Security headers
    add_header X-Frame-Options "SAMEORIGIN" always;
    add_header X-Content-Type-Options "nosniff" always;
    add_header X-XSS-Protection "1; mode=block" always;
    add_header Referrer-Policy "strict-origin-when-cross-origin" always;

    # Cache static assets
    location /static/ {
        expires 1y;
        add_header Cache-Control "public, immutable";
    }

    # Gzip
    gzip on;
    gzip_types text/plain text/css application/json application/javascript text/xml application/xml text/javascript image/svg+xml;
    gzip_min_length 1000;
}
```

**Step 3: Create frontend Dockerfile (multi-stage)**

```dockerfile
# Build stage
FROM node:20-alpine AS build

WORKDIR /app

COPY package.json yarn.lock ./
RUN yarn install --frozen-lockfile

COPY . .
RUN npx craco build

# Production stage
FROM nginx:alpine

COPY --from=build /app/build /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
```

**Step 4: Commit**

```bash
git add frontend/Dockerfile frontend/.dockerignore frontend/nginx.conf
git commit -m "feat: add frontend Dockerfile with nginx"
```

---

### Task 3: docker-compose.yml

**Files:**
- Create: `docker-compose.yml`
- Create: `.env.example` (project root)

**Step 1: Create root .env.example**

```env
# MongoDB
MONGO_URL=mongodb://mongo:27017
DB_NAME=memorykeeper

# Auth
JWT_SECRET_KEY=change-this-to-a-random-64-char-string

# OpenAI
OPENAI_API_KEY=sk-...

# CORS (comma-separated)
CORS_ORIGINS=http://localhost:3000,http://localhost

# Stripe (optional)
STRIPE_SECRET_KEY=
STRIPE_WEBHOOK_SECRET=
STRIPE_PRICE_ID=

# Telegram Bot (optional)
TELEGRAM_BOT_TOKEN=
TELEGRAM_BOT_WEBHOOK_SECRET=

# Push Notifications (optional)
VAPID_PRIVATE_KEY=
VAPID_CONTACT_EMAIL=admin@alzahelp.com

# Twilio (optional)
TWILIO_ACCOUNT_SID=
TWILIO_AUTH_TOKEN=
TWILIO_FROM_NUMBER=
```

**Step 2: Create docker-compose.yml**

```yaml
version: "3.8"

services:
  mongo:
    image: mongo:7
    restart: unless-stopped
    volumes:
      - mongo_data:/data/db
    ports:
      - "27017:27017"

  backend:
    build: ./backend
    restart: unless-stopped
    env_file: .env
    environment:
      - MONGO_URL=mongodb://mongo:27017
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

**Step 3: Commit**

```bash
git add docker-compose.yml .env.example
git commit -m "feat: add docker-compose for local dev and deployment"
```

---

### Task 4: MongoDB Indexes on Startup

**Files:**
- Modify: `backend/server.py` (add `ensure_indexes()` function, call it from startup event)

**Step 1: Add ensure_indexes function**

Add right before the `@app.on_event("startup")` for the medication scheduler (~line 7554):

```python
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
```

**Step 2: Add startup event to call ensure_indexes**

Add a new startup event right before the existing medication scheduler startup event:

```python
@app.on_event("startup")
async def setup_db_indexes():
    await ensure_indexes()
```

**Step 3: Verify syntax**

Run: `python3 -m py_compile backend/server.py`
Expected: No output (success)

**Step 4: Commit**

```bash
git add backend/server.py
git commit -m "feat: add MongoDB index creation on startup"
```

---

### Task 5: Security Headers Middleware

**Files:**
- Modify: `backend/server.py` (add security headers middleware after CORS)

**Step 1: Add security headers middleware**

Add after the CORS middleware block (after `allow_headers=["Content-Type", "Authorization"]`):

```python
@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["Permissions-Policy"] = "camera=(), microphone=(self), geolocation=(self)"
    return response
```

Note: Microphone and geolocation are allowed for `self` because the app uses voice assistant and safety geofencing.

**Step 2: Verify syntax**

Run: `python3 -m py_compile backend/server.py`
Expected: No output (success)

**Step 3: Commit**

```bash
git add backend/server.py
git commit -m "feat: add security headers middleware"
```

---

### Task 6: Rate Limiting on Auth Endpoints

**Files:**
- Modify: `backend/server.py` (add rate limit decorators to login and register)

**Step 1: Add rate limits to auth endpoints**

The project already has slowapi imported. Find the existing rate limiter setup and add decorators to auth endpoints.

On the `@api_router.post("/auth/login")` endpoint, add before the route decorator:
```python
@limiter.limit("10/minute")
```

On the `@api_router.post("/auth/register")` endpoint, add before the route decorator:
```python
@limiter.limit("5/minute")
```

On the `@api_router.post("/auth/refresh")` endpoint, add before the route decorator:
```python
@limiter.limit("20/minute")
```

Note: These decorators must go BEFORE the `@api_router` decorator. The limiter uses `request: Request` as a parameter — the auth endpoints already have this available via FastAPI's dependency injection, but if `Request` is not in the function signature, add it.

Check that these endpoints have `request: Request` in their signature. If not, add it as the first parameter.

**Step 2: Verify syntax**

Run: `python3 -m py_compile backend/server.py`
Expected: No output (success)

**Step 3: Commit**

```bash
git add backend/server.py
git commit -m "feat: add rate limiting to auth endpoints"
```

---

### Task 7: Privacy Policy and Terms of Service Pages

**Files:**
- Modify: `frontend/src/App.js` (add routes)
- Create: `frontend/src/pages/PrivacyPolicyPage.jsx`
- Create: `frontend/src/pages/TermsOfServicePage.jsx`
- Modify: `frontend/src/pages/LandingPage.jsx` (add footer links)

**Step 1: Create PrivacyPolicyPage.jsx**

Create at `frontend/src/pages/PrivacyPolicyPage.jsx`:

```jsx
import React from "react";
import { Link } from "react-router-dom";
import { ArrowLeft } from "lucide-react";

export const PrivacyPolicyPage = () => (
  <div className="min-h-screen bg-background">
    <div className="container mx-auto px-4 py-12 max-w-3xl">
      <Link to="/" className="inline-flex items-center gap-2 text-sm text-muted-foreground hover:text-foreground mb-8">
        <ArrowLeft className="h-4 w-4" /> Back to home
      </Link>
      <h1 className="text-3xl font-bold mb-2">Privacy Policy</h1>
      <p className="text-muted-foreground mb-8">Last updated: February 23, 2026</p>

      <div className="prose prose-gray dark:prose-invert max-w-none space-y-6">
        <section>
          <h2 className="text-xl font-semibold mt-6 mb-3">1. Information We Collect</h2>
          <p className="text-muted-foreground leading-relaxed">
            AlzaHelp collects information you provide directly: account details (name, email), health data (medications, care instructions, mood observations), family information, memories, and voice recordings when using the voice assistant. We also collect usage data (pages visited, features used) through PostHog analytics.
          </p>
        </section>

        <section>
          <h2 className="text-xl font-semibold mt-6 mb-3">2. How We Use Your Information</h2>
          <p className="text-muted-foreground leading-relaxed">
            We use your data to provide and improve the AlzaHelp service: managing medications, delivering care instructions via voice assistant, enabling caregiver access, sending reminders and push notifications, and generating AI-powered responses. Health data is processed by OpenAI's API for voice and chat features.
          </p>
        </section>

        <section>
          <h2 className="text-xl font-semibold mt-6 mb-3">3. Data Sharing</h2>
          <p className="text-muted-foreground leading-relaxed">
            We do not sell your personal data. Data is shared only with: caregivers/clinicians you explicitly authorize, third-party services required for functionality (OpenAI for AI features, Stripe for payments, Twilio for SMS alerts), and as required by law.
          </p>
        </section>

        <section>
          <h2 className="text-xl font-semibold mt-6 mb-3">4. Data Security</h2>
          <p className="text-muted-foreground leading-relaxed">
            We use encryption in transit (TLS/HTTPS), secure password hashing (bcrypt), JWT-based authentication, and role-based access controls. Data is stored in MongoDB with access restricted to authorized services.
          </p>
        </section>

        <section>
          <h2 className="text-xl font-semibold mt-6 mb-3">5. Your Rights</h2>
          <p className="text-muted-foreground leading-relaxed">
            You can export all your data at any time from your account settings. You can delete your account and all associated data permanently. Caregivers can only access data you explicitly share via invite codes.
          </p>
        </section>

        <section>
          <h2 className="text-xl font-semibold mt-6 mb-3">6. Data Retention</h2>
          <p className="text-muted-foreground leading-relaxed">
            We retain your data for as long as your account is active. When you delete your account, all personal data is permanently removed within 30 days. Session data expires automatically after 30 days of inactivity.
          </p>
        </section>

        <section>
          <h2 className="text-xl font-semibold mt-6 mb-3">7. Contact</h2>
          <p className="text-muted-foreground leading-relaxed">
            For privacy questions or data requests, contact us at privacy@alzahelp.com.
          </p>
        </section>
      </div>
    </div>
  </div>
);

export default PrivacyPolicyPage;
```

**Step 2: Create TermsOfServicePage.jsx**

Create at `frontend/src/pages/TermsOfServicePage.jsx`:

```jsx
import React from "react";
import { Link } from "react-router-dom";
import { ArrowLeft } from "lucide-react";

export const TermsOfServicePage = () => (
  <div className="min-h-screen bg-background">
    <div className="container mx-auto px-4 py-12 max-w-3xl">
      <Link to="/" className="inline-flex items-center gap-2 text-sm text-muted-foreground hover:text-foreground mb-8">
        <ArrowLeft className="h-4 w-4" /> Back to home
      </Link>
      <h1 className="text-3xl font-bold mb-2">Terms of Service</h1>
      <p className="text-muted-foreground mb-8">Last updated: February 23, 2026</p>

      <div className="prose prose-gray dark:prose-invert max-w-none space-y-6">
        <section>
          <h2 className="text-xl font-semibold mt-6 mb-3">1. Acceptance of Terms</h2>
          <p className="text-muted-foreground leading-relaxed">
            By using AlzaHelp, you agree to these terms. If you do not agree, do not use the service. AlzaHelp is designed to assist with memory care management but is not a substitute for professional medical advice, diagnosis, or treatment.
          </p>
        </section>

        <section>
          <h2 className="text-xl font-semibold mt-6 mb-3">2. Service Description</h2>
          <p className="text-muted-foreground leading-relaxed">
            AlzaHelp provides memory assistance tools including medication tracking, voice-guided care instructions, caregiver coordination, safety monitoring, and AI-powered companionship. The service is available as a web application with optional Telegram/WhatsApp bot integration.
          </p>
        </section>

        <section>
          <h2 className="text-xl font-semibold mt-6 mb-3">3. Account Responsibilities</h2>
          <p className="text-muted-foreground leading-relaxed">
            You are responsible for maintaining the security of your account. Caregivers and clinicians must only access patient data through proper authorization (invite codes). You must not share account credentials or invite codes with unauthorized persons.
          </p>
        </section>

        <section>
          <h2 className="text-xl font-semibold mt-6 mb-3">4. Subscription and Billing</h2>
          <p className="text-muted-foreground leading-relaxed">
            AlzaHelp offers a free tier (up to 3 patients) and a premium subscription ($9.99/month, up to 20 patients with external bot access). Subscriptions are managed through Stripe. You may cancel at any time through the billing portal.
          </p>
        </section>

        <section>
          <h2 className="text-xl font-semibold mt-6 mb-3">5. Medical Disclaimer</h2>
          <p className="text-muted-foreground leading-relaxed">
            AlzaHelp is a care coordination tool, not a medical device. Medication reminders and care instructions are informational only. Always consult healthcare professionals for medical decisions. AlzaHelp is not responsible for missed medications or care actions.
          </p>
        </section>

        <section>
          <h2 className="text-xl font-semibold mt-6 mb-3">6. Limitation of Liability</h2>
          <p className="text-muted-foreground leading-relaxed">
            AlzaHelp is provided "as is" without warranties. We are not liable for any damages arising from use of the service, including but not limited to missed alerts, data loss, or service interruptions. Our total liability is limited to the amount you paid in the 12 months preceding the claim.
          </p>
        </section>

        <section>
          <h2 className="text-xl font-semibold mt-6 mb-3">7. Termination</h2>
          <p className="text-muted-foreground leading-relaxed">
            You may delete your account at any time. We reserve the right to suspend accounts that violate these terms or engage in abusive behavior. Upon termination, your data will be deleted per our privacy policy.
          </p>
        </section>

        <section>
          <h2 className="text-xl font-semibold mt-6 mb-3">8. Contact</h2>
          <p className="text-muted-foreground leading-relaxed">
            For questions about these terms, contact us at legal@alzahelp.com.
          </p>
        </section>
      </div>
    </div>
  </div>
);

export default TermsOfServicePage;
```

**Step 3: Add routes in App.js**

Add imports at top of App.js:
```jsx
import { PrivacyPolicyPage } from "@/pages/PrivacyPolicyPage";
import { TermsOfServicePage } from "@/pages/TermsOfServicePage";
```

Add routes inside `<Routes>` in AppRouter, before the catch-all:
```jsx
<Route path="/privacy" element={<PrivacyPolicyPage />} />
<Route path="/terms" element={<TermsOfServicePage />} />
```

**Step 4: Add footer links in LandingPage.jsx**

Replace the footer content. After `<p>Made with care for those who need it most.</p>`, add:
```jsx
<div className="mt-3 flex items-center justify-center gap-4 text-xs">
  <Link to="/privacy" className="hover:text-foreground transition-colors">Privacy Policy</Link>
  <span>·</span>
  <Link to="/terms" className="hover:text-foreground transition-colors">Terms of Service</Link>
</div>
```

Also add `Link` to the LandingPage imports from react-router-dom.

**Step 5: Verify build**

Run: `cd frontend && npx craco build`
Expected: Compiled successfully.

**Step 6: Commit**

```bash
git add frontend/src/pages/PrivacyPolicyPage.jsx frontend/src/pages/TermsOfServicePage.jsx frontend/src/App.js frontend/src/pages/LandingPage.jsx
git commit -m "feat: add privacy policy and terms of service pages"
```

---

### Task 8: PWA Icon Assets

**Files:**
- Create: `frontend/public/icon-192.png`
- Create: `frontend/public/icon-512.png`

**Step 1: Generate placeholder icons**

Use a simple canvas-based approach or download from a generator. The icons should be purple (#7c3aed, matching theme_color in manifest) with "AH" text.

For MVP, generate simple SVG-based PNGs using an online tool or a script. At minimum, create placeholder PNG files so the PWA manifest doesn't 404.

A quick approach: use the `convert` command (ImageMagick) or create via canvas in Node:

```bash
# If ImageMagick is available:
convert -size 192x192 xc:"#7c3aed" -fill white -gravity center -pointsize 72 -annotate 0 "AH" frontend/public/icon-192.png
convert -size 512x512 xc:"#7c3aed" -fill white -gravity center -pointsize 192 -annotate 0 "AH" frontend/public/icon-512.png
```

If ImageMagick is not available, create a small Node script or manually place any 192x192 and 512x512 PNG files.

**Step 2: Commit**

```bash
git add frontend/public/icon-192.png frontend/public/icon-512.png
git commit -m "feat: add PWA icon assets"
```

---

### Task 9: GitHub Actions CI Pipeline

**Files:**
- Create: `.github/workflows/ci.yml`

**Step 1: Create CI workflow**

```yaml
name: CI

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  backend:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - name: Install dependencies
        run: pip install -r backend/requirements.txt
      - name: Syntax check
        run: python -m py_compile backend/server.py
      - name: Run tests
        run: pytest backend/tests/ -v --tb=short
        env:
          MONGO_URL: mongodb://localhost:27017
          DB_NAME: memorykeeper_test
          JWT_SECRET_KEY: test-secret-key
          OPENAI_API_KEY: sk-test

    services:
      mongo:
        image: mongo:7
        ports:
          - 27017:27017

  frontend:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-node@v4
        with:
          node-version: "20"
          cache: "yarn"
          cache-dependency-path: frontend/yarn.lock
      - name: Install dependencies
        working-directory: frontend
        run: yarn install --frozen-lockfile
      - name: Build
        working-directory: frontend
        run: npx craco build
```

**Step 2: Commit**

```bash
mkdir -p .github/workflows
git add .github/workflows/ci.yml
git commit -m "feat: add GitHub Actions CI pipeline"
```

---

### Task 10: Update .env.example with All Variables

**Files:**
- Modify: `backend/.env.example`

**Step 1: Update .env.example to include all env vars used in server.py**

Replace content of `backend/.env.example`:

```env
# Required
MONGO_URL=mongodb://localhost:27017
DB_NAME=memorykeeper
JWT_SECRET_KEY=change-this-to-a-random-64-char-string
OPENAI_API_KEY=sk-...
CORS_ORIGINS=http://localhost:3000

# AI Models (optional, defaults shown)
EMBEDDING_MODEL=text-embedding-3-small
STT_MODEL=gpt-4o-mini-transcribe

# Stripe Billing (optional)
STRIPE_SECRET_KEY=
STRIPE_WEBHOOK_SECRET=
STRIPE_PRICE_ID=

# Telegram Bot (optional)
TELEGRAM_BOT_TOKEN=
TELEGRAM_BOT_WEBHOOK_SECRET=

# Push Notifications (optional)
VAPID_PRIVATE_KEY=
VAPID_CONTACT_EMAIL=admin@alzahelp.com

# Twilio SMS (optional)
TWILIO_ACCOUNT_SID=
TWILIO_AUTH_TOKEN=
TWILIO_FROM_NUMBER=

# Admin (optional)
ADMIN_BOOTSTRAP_EMAILS=
```

**Step 2: Commit**

```bash
git add backend/.env.example
git commit -m "chore: update .env.example with all environment variables"
```

---

## Verification Checklist

After all tasks are complete:

1. `python3 -m py_compile backend/server.py` — passes
2. `cd frontend && npx craco build` — compiles successfully
3. `docker-compose config` — valid YAML (if Docker available)
4. `docker build -t alzahelp-backend ./backend` — builds (if Docker available)
5. `docker build -t alzahelp-frontend ./frontend` — builds (if Docker available)
6. Visit `/privacy` and `/terms` routes in browser — pages render
7. No broken icon references in manifest

## Task Summary

| Task | Description | Files |
|------|-------------|-------|
| 1 | Backend Dockerfile | `backend/Dockerfile`, `backend/.dockerignore` |
| 2 | Frontend Dockerfile + nginx | `frontend/Dockerfile`, `frontend/.dockerignore`, `frontend/nginx.conf` |
| 3 | docker-compose.yml | `docker-compose.yml`, `.env.example` |
| 4 | MongoDB indexes on startup | `backend/server.py` |
| 5 | Security headers middleware | `backend/server.py` |
| 6 | Rate limiting on auth endpoints | `backend/server.py` |
| 7 | Privacy Policy + Terms of Service | `frontend/src/pages/PrivacyPolicyPage.jsx`, `frontend/src/pages/TermsOfServicePage.jsx`, `frontend/src/App.js`, `frontend/src/pages/LandingPage.jsx` |
| 8 | PWA icon assets | `frontend/public/icon-192.png`, `frontend/public/icon-512.png` |
| 9 | GitHub Actions CI | `.github/workflows/ci.yml` |
| 10 | Complete .env.example | `backend/.env.example` |
