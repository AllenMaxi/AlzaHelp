# MemoryKeeper - Product Requirements Document

## Overview
MemoryKeeper is a web application designed specifically for people with Alzheimer's disease to store and retrieve their precious memories. The application focuses on accessibility, warmth, and ease of use.

## Target Users
- People with Alzheimer's or memory challenges
- Family members and caregivers who help manage memories

## Core Requirements

### Authentication
- Google OAuth for easy sign-in (no passwords to remember)
- Emergent-managed authentication integration
- Session-based authorization with secure cookies

### Design Philosophy
- Warm earth tones color palette
- Extra-large buttons and text for accessibility
- Simple, clear navigation
- Voice feedback options

### Core Features

#### 1. Family Directory
- Store family member information with photos
- Categorize by relationship (spouse, children, grandchildren, friends, etc.)
- Include contact info, birthday, notes
- Voice notes support

#### 2. Memory Timeline
- Store memories with title, date, location, description
- Tag people who appear in memories
- Photo attachments stored in MongoDB GridFS
- Filter by decade

#### 3. AI Chat Assistant
- RAG-powered chat using keyword-based search
- Answers questions about stored family and memories
- Warm, patient tone appropriate for target users
- Uses Emergent LLM integration

#### 4. Daily Reminders
- Set daily reminders (medications, meals, activities)
- Toggle completion status
- Reset all reminders for new day

### New Features (Completed January 2026)

#### 5. "Who Is This?" Quiz Game
- Memory practice game showing family photos
- Multiple difficulty levels (2-4 choices)
- Score tracking and streak counter
- Voice hints with browser Speech API
- Encouraging feedback messages

#### 6. Photo Stories
- Auto-generated slideshows from memories
- Voice narration using browser Speech API
- Play/pause controls
- Memory descriptions and people tags displayed

#### 7. Week Memories
- View recent activities by day (last 7 days)
- Daily notes functionality (add/view)
- "On This Day" historical memory matching
- Quick family member tagging

### Technical Stack
- **Frontend**: React, Tailwind CSS, shadcn/ui
- **Backend**: Python FastAPI
- **Database**: MongoDB with GridFS for files
- **Authentication**: Google OAuth via Emergent
- **AI**: Emergent LLM integration (GPT-4.1)

## API Endpoints

### Auth
- `POST /api/auth/session` - Exchange session_id for token
- `GET /api/auth/me` - Get current user
- `POST /api/auth/logout` - Logout

### Family
- `GET /api/family` - List family members
- `POST /api/family` - Create family member
- `PUT /api/family/{id}` - Update family member
- `DELETE /api/family/{id}` - Delete family member

### Memories
- `GET /api/memories` - List memories
- `POST /api/memories` - Create memory
- `PUT /api/memories/{id}` - Update memory
- `DELETE /api/memories/{id}` - Delete memory

### Daily Notes
- `GET /api/daily-notes` - List notes (last 30 days)
- `POST /api/daily-notes` - Create/append note
- `GET /api/daily-notes/{date}` - Get note by date

### Reminders
- `GET /api/reminders` - List reminders
- `POST /api/reminders` - Create reminder
- `PUT /api/reminders/{id}/toggle` - Toggle completion
- `DELETE /api/reminders/{id}` - Delete reminder
- `POST /api/reminders/reset` - Reset all for new day

### Files
- `POST /api/upload` - Upload file to GridFS
- `POST /api/upload/multiple` - Upload multiple files
- `GET /api/files/{filename}` - Retrieve file

### Chat
- `POST /api/chat` - Send message to AI assistant
- `GET /api/chat/history/{session_id}` - Get chat history

## Current Status: MVP COMPLETE

All planned features have been implemented and tested:
- ✅ Google OAuth authentication
- ✅ Family member CRUD
- ✅ Memory CRUD with photos
- ✅ AI Chat Assistant with RAG
- ✅ Daily reminders
- ✅ Voice-to-text input
- ✅ "Who Is This?" Quiz
- ✅ Photo Stories
- ✅ Week Memories

## Future Enhancements (P1)
- Text-to-speech narration using ElevenLabs or similar
- Photo upload from mobile camera
- Family member photo recognition AI
- Caregiver portal with multi-user support
- Calendar integration for appointments
- Medication tracking with alerts
