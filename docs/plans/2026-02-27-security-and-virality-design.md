# AlzaHelp — Security & Virality Design

## Tier 1: Security Fixes

### 1.1 File Serving Auth (server.py:2384)
- Add `Depends(get_current_user)` to `GET /files/{filename}`
- Look up GridFS metadata for `user_id`, verify requester owns the file OR is a linked caregiver/clinician
- Keep a public exception for demo account files (demo mode needs unauthenticated access)
- Change Cache-Control from `public, max-age=31536000` to `private, max-age=3600`

### 1.2 File Upload Validation (server.py:2311, 2349)
- Whitelist MIME types: `image/jpeg, image/png, image/gif, image/webp, audio/mpeg, audio/wav, audio/webm, audio/ogg, video/mp4, video/webm, application/pdf`
- Enforce 10MB max per file via `content = await file.read()` size check
- Strip path components from original filename, limit to 255 chars

### 1.3 MongoDB Auth in Docker (docker-compose.yml)
- Add `MONGO_INITDB_ROOT_USERNAME` and `MONGO_INITDB_ROOT_PASSWORD` env vars to mongo service
- Update backend `MONGO_URL` to include credentials
- Remove `ports: "27017:27017"` — internal-only access via Docker network
- Add `.env.example` with placeholder credentials

### 1.4 Password Strength Policy (server.py:1778)
- Add `validate_password()` function: min 8 chars, 1 uppercase, 1 digit
- Call from register endpoint before hashing
- Return clear error message on failure

### 1.5 CSP Header
- FastAPI middleware: add `Content-Security-Policy` with `default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data: blob:; connect-src 'self'; font-src 'self'; frame-ancestors 'none'`
- nginx.conf: add same CSP header

### 1.6 Token Revocation (server.py)
- Add `password_changed_at` field to user documents (default: registration time)
- In `get_current_user()`: decode JWT `iat`, compare to `password_changed_at` — reject if token was issued before password change
- Update `password_changed_at` on any password change endpoint

### 1.7 Cookie SameSite (server.py)
- Change all `samesite="none"` to `samesite="lax"` since we're deploying same-domain
- Keep `secure=True` for production, but make it configurable via env var for local dev (`COOKIE_SECURE=false`)

## Tier 2: Viral Features

### 2.1 PWA Install Prompt
- New component `InstallPrompt.jsx` — listens for `beforeinstallprompt` event
- Shows dismissible banner at bottom: "Install AlzaHelp for the best experience"
- Stores dismissal in localStorage, re-shows after 7 days
- Integrate into DashboardPage (after onboarding completes)

### 2.2 Demo Mode
**Backend:**
- New endpoint `POST /auth/demo` — creates or returns a session for a shared demo account
- Demo account seeded on first request with: 3 family members, 5 memories with photos, 4 medications, mood entries, reminders, sample care instructions
- Demo sessions are read-only (POST/PUT/DELETE operations return 403 for demo users)
- Add `is_demo: true` flag to demo user document
- Guard all mutation endpoints with demo check

**Frontend:**
- "Try Demo" button on LandingPage hero section
- On click: call `/auth/demo`, set cookie, redirect to `/dashboard`
- Show subtle banner on dashboard: "You're in demo mode — Sign up to save your data"
- Demo user sees all features working with realistic data

### 2.3 i18n (English + Spanish)
**Setup:**
- Install `react-i18next` + `i18next` + `i18next-browser-languagedetector`
- Create `/src/i18n/` directory with `en.json` and `es.json` translation files
- Initialize i18next in `index.js` with browser language detection
- Wrap App in `I18nextProvider`

**Scope:**
- Landing page (all text)
- Login/Register pages
- Dashboard tab names and headers
- OnboardingWizard steps
- Common UI elements (buttons, labels, errors)
- Voice recognition: detect language from i18n setting, set `recognition.lang` accordingly

**Language Switcher:**
- Add globe icon button in Header.jsx
- Dropdown with EN/ES flags
- Persist choice in localStorage

### 2.4 Enhanced SEO
- Add JSON-LD structured data (SoftwareApplication schema) to index.html
- Add `og:image` pointing to a social share card (create a simple 1200x630 image)
- Add canonical URL tag
- Add `robots.txt` allowing all crawlers
- Add `sitemap.xml` with public pages

### 2.5 Daily Caregiver Digest
**Backend:**
- New endpoint `GET /care/patients/{patient_id}/daily-digest` — generates AI summary
- Collects: medication adherence, mood checkins, safety alerts, quiz activity
- Uses OpenAI to generate a natural-language 3-sentence summary
- New scheduled task: `send_daily_digests()` — runs at 8 PM patient timezone
- Sends via push notification to all linked caregivers
- Falls back to SMS for premium users with no push subscription

**Frontend:**
- Caregiver portal: "Daily Digest" toggle per patient
- Notification preferences in settings

### 2.6 PDF Care Reports
**Backend:**
- New endpoint `GET /care/patients/{patient_id}/report` — generates PDF
- Uses `reportlab` or `weasyprint` for PDF generation
- Sections: Patient info, medication list + adherence, mood trends, safety events, care instructions
- Date range parameter (default: last 30 days)
- Returns StreamingResponse with `application/pdf`

**Frontend:**
- "Download Report" button in caregiver portal
- Date range picker for report period

### 2.7 Improved Onboarding
- Expand from 3 to 5 steps: Welcome → Add Photo → Add Medication → Add Family Member → Install PWA
- Add progress celebrations (checkmark animations)
- "Add Photo" step: upload a profile photo or family photo
- PWA install step: trigger install prompt if available
- Skip-all still available

### 2.8 Referral Sharing
**Backend:**
- New endpoint `POST /referral/generate` — creates a unique referral URL
- Referral code stored in user document
- New endpoint `POST /auth/register` — accept optional `referral_code` parameter
- Track referral source in new users

**Frontend:**
- "Invite Family" card in caregiver portal with share options
- Native Web Share API (`navigator.share()`) with fallback to copy-to-clipboard
- Share message: "I use AlzaHelp to help care for my loved one. Join me: {url}"
- Show referral count in profile/settings

## Implementation Order

1. Security fixes (1.1-1.7) — all in one pass
2. Demo mode (2.2) — highest conversion impact
3. PWA install prompt (2.1) — quick win
4. i18n setup + Spanish (2.3) — market expansion
5. SEO enhancements (2.4) — free traffic
6. Improved onboarding (2.7) — retention
7. Daily digest (2.5) — engagement + premium value
8. PDF reports (2.6) — premium value
9. Referral sharing (2.8) — growth loop
