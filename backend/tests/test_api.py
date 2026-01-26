"""
Backend API Tests for MemoryKeeper App
Tests: Family, Memories, Daily Notes, Reminders, Auth endpoints
"""
import pytest
import requests
import os
from datetime import datetime

BASE_URL = os.environ.get('REACT_APP_BACKEND_URL', '').rstrip('/')
SESSION_TOKEN = "sess_test_eba950c34a0140878f212a3ad58d66d2"

@pytest.fixture
def api_client():
    """Shared requests session with auth header"""
    session = requests.Session()
    session.headers.update({
        "Content-Type": "application/json",
        "Authorization": f"Bearer {SESSION_TOKEN}"
    })
    return session


class TestHealthAndAuth:
    """Health check and authentication tests"""
    
    def test_api_root(self, api_client):
        """Test API root endpoint"""
        response = api_client.get(f"{BASE_URL}/api/")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert data["message"] == "MemoryKeeper API"
        print("✓ API root endpoint working")
    
    def test_auth_me(self, api_client):
        """Test authenticated user endpoint"""
        response = api_client.get(f"{BASE_URL}/api/auth/me")
        assert response.status_code == 200
        data = response.json()
        assert "user_id" in data
        assert "email" in data
        assert "name" in data
        print(f"✓ Auth/me working - User: {data['name']}")
    
    def test_auth_me_without_token(self):
        """Test auth endpoint without token returns 401"""
        response = requests.get(f"{BASE_URL}/api/auth/me")
        assert response.status_code == 401
        print("✓ Auth properly rejects unauthenticated requests")


class TestFamilyAPI:
    """Family members CRUD tests"""
    
    def test_get_family_members(self, api_client):
        """Test GET /api/family returns family members"""
        response = api_client.get(f"{BASE_URL}/api/family")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        # Test user has 3 family members
        assert len(data) >= 3
        
        # Verify structure
        for member in data:
            assert "id" in member
            assert "name" in member
            assert "relationship" in member
            assert "relationship_label" in member
            assert "category" in member
        
        # Verify test data exists
        names = [m["name"] for m in data]
        assert "Maria" in names
        assert "Michael" in names
        assert "Sarah" in names
        print(f"✓ GET /api/family - Found {len(data)} family members")
    
    def test_create_family_member(self, api_client):
        """Test POST /api/family creates a new family member"""
        payload = {
            "name": "TEST_Uncle Bob",
            "relationship": "uncle",
            "relationship_label": "Your Uncle",
            "category": "other",
            "notes": "Test family member"
        }
        response = api_client.post(f"{BASE_URL}/api/family", json=payload)
        assert response.status_code == 200
        data = response.json()
        
        assert data["name"] == "TEST_Uncle Bob"
        assert data["relationship"] == "uncle"
        assert "id" in data
        
        # Verify persistence with GET
        get_response = api_client.get(f"{BASE_URL}/api/family")
        assert get_response.status_code == 200
        members = get_response.json()
        created = next((m for m in members if m["name"] == "TEST_Uncle Bob"), None)
        assert created is not None
        
        # Cleanup
        api_client.delete(f"{BASE_URL}/api/family/{data['id']}")
        print("✓ POST /api/family - Create and verify persistence")
    
    def test_update_family_member(self, api_client):
        """Test PUT /api/family/{id} updates a family member"""
        # Create test member
        create_payload = {
            "name": "TEST_Update Member",
            "relationship": "friend",
            "relationship_label": "Your Friend",
            "category": "friends"
        }
        create_response = api_client.post(f"{BASE_URL}/api/family", json=create_payload)
        member_id = create_response.json()["id"]
        
        # Update
        update_payload = {"name": "TEST_Updated Name", "notes": "Updated notes"}
        update_response = api_client.put(f"{BASE_URL}/api/family/{member_id}", json=update_payload)
        assert update_response.status_code == 200
        
        # Verify update persisted
        get_response = api_client.get(f"{BASE_URL}/api/family")
        members = get_response.json()
        updated = next((m for m in members if m["id"] == member_id), None)
        assert updated is not None
        assert updated["name"] == "TEST_Updated Name"
        assert updated["notes"] == "Updated notes"
        
        # Cleanup
        api_client.delete(f"{BASE_URL}/api/family/{member_id}")
        print("✓ PUT /api/family - Update and verify persistence")
    
    def test_delete_family_member(self, api_client):
        """Test DELETE /api/family/{id} removes a family member"""
        # Create test member
        create_payload = {
            "name": "TEST_Delete Member",
            "relationship": "other",
            "relationship_label": "Other",
            "category": "other"
        }
        create_response = api_client.post(f"{BASE_URL}/api/family", json=create_payload)
        member_id = create_response.json()["id"]
        
        # Delete
        delete_response = api_client.delete(f"{BASE_URL}/api/family/{member_id}")
        assert delete_response.status_code == 200
        
        # Verify deletion
        get_response = api_client.get(f"{BASE_URL}/api/family")
        members = get_response.json()
        deleted = next((m for m in members if m["id"] == member_id), None)
        assert deleted is None
        print("✓ DELETE /api/family - Delete and verify removal")


class TestMemoriesAPI:
    """Memories CRUD tests"""
    
    def test_get_memories(self, api_client):
        """Test GET /api/memories returns memories"""
        response = api_client.get(f"{BASE_URL}/api/memories")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) >= 2
        
        # Verify structure
        for memory in data:
            assert "id" in memory
            assert "title" in memory
            assert "date" in memory
            assert "year" in memory
            assert "description" in memory
        
        # Verify test data
        titles = [m["title"] for m in data]
        assert "Our Wedding Day" in titles
        print(f"✓ GET /api/memories - Found {len(data)} memories")
    
    def test_create_memory(self, api_client):
        """Test POST /api/memories creates a new memory"""
        payload = {
            "title": "TEST_Memory",
            "date": "January 1, 2020",
            "year": 2020,
            "location": "Test Location",
            "description": "Test memory description",
            "people": ["Maria"],
            "category": "other"
        }
        response = api_client.post(f"{BASE_URL}/api/memories", json=payload)
        assert response.status_code == 200
        data = response.json()
        
        assert data["title"] == "TEST_Memory"
        assert data["year"] == 2020
        assert "id" in data
        
        # Verify persistence
        get_response = api_client.get(f"{BASE_URL}/api/memories")
        memories = get_response.json()
        created = next((m for m in memories if m["title"] == "TEST_Memory"), None)
        assert created is not None
        
        # Cleanup
        api_client.delete(f"{BASE_URL}/api/memories/{data['id']}")
        print("✓ POST /api/memories - Create and verify persistence")
    
    def test_delete_memory(self, api_client):
        """Test DELETE /api/memories/{id} removes a memory"""
        # Create test memory
        create_payload = {
            "title": "TEST_Delete Memory",
            "date": "January 1, 2020",
            "year": 2020,
            "description": "To be deleted",
            "category": "other"
        }
        create_response = api_client.post(f"{BASE_URL}/api/memories", json=create_payload)
        memory_id = create_response.json()["id"]
        
        # Delete
        delete_response = api_client.delete(f"{BASE_URL}/api/memories/{memory_id}")
        assert delete_response.status_code == 200
        
        # Verify deletion
        get_response = api_client.get(f"{BASE_URL}/api/memories")
        memories = get_response.json()
        deleted = next((m for m in memories if m["id"] == memory_id), None)
        assert deleted is None
        print("✓ DELETE /api/memories - Delete and verify removal")


class TestDailyNotesAPI:
    """Daily notes CRUD tests - NEW FEATURE"""
    
    def test_get_daily_notes(self, api_client):
        """Test GET /api/daily-notes returns notes"""
        response = api_client.get(f"{BASE_URL}/api/daily-notes")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        
        # Verify structure if notes exist
        if len(data) > 0:
            note = data[0]
            assert "id" in note
            assert "date" in note
            assert "note" in note
            assert "created_at" in note
        print(f"✓ GET /api/daily-notes - Found {len(data)} notes")
    
    def test_create_daily_note(self, api_client):
        """Test POST /api/daily-notes creates a new note"""
        today = datetime.now().strftime("%Y-%m-%d")
        payload = {
            "date": "2026-01-25",  # Use a different date to avoid conflicts
            "note": "TEST_Daily note content"
        }
        response = api_client.post(f"{BASE_URL}/api/daily-notes", json=payload)
        assert response.status_code == 200
        data = response.json()
        
        assert "id" in data
        assert data["date"] == "2026-01-25"
        assert "TEST_Daily note content" in data["note"]
        print("✓ POST /api/daily-notes - Create note")
    
    def test_get_daily_note_by_date(self, api_client):
        """Test GET /api/daily-notes/{date} returns specific note"""
        # First create a note
        payload = {
            "date": "2026-01-24",
            "note": "TEST_Specific date note"
        }
        api_client.post(f"{BASE_URL}/api/daily-notes", json=payload)
        
        # Get by date
        response = api_client.get(f"{BASE_URL}/api/daily-notes/2026-01-24")
        assert response.status_code == 200
        data = response.json()
        assert data["date"] == "2026-01-24"
        print("✓ GET /api/daily-notes/{date} - Get note by date")
    
    def test_append_to_existing_note(self, api_client):
        """Test POST /api/daily-notes appends to existing note"""
        # Create initial note
        payload1 = {
            "date": "2026-01-23",
            "note": "First note"
        }
        api_client.post(f"{BASE_URL}/api/daily-notes", json=payload1)
        
        # Append to same date
        payload2 = {
            "date": "2026-01-23",
            "note": "Second note"
        }
        response = api_client.post(f"{BASE_URL}/api/daily-notes", json=payload2)
        assert response.status_code == 200
        data = response.json()
        
        # Should contain both notes
        assert "First note" in data["note"]
        assert "Second note" in data["note"]
        print("✓ POST /api/daily-notes - Append to existing note")


class TestRemindersAPI:
    """Reminders CRUD tests"""
    
    def test_get_reminders(self, api_client):
        """Test GET /api/reminders returns reminders"""
        response = api_client.get(f"{BASE_URL}/api/reminders")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        print(f"✓ GET /api/reminders - Found {len(data)} reminders")
    
    def test_create_reminder(self, api_client):
        """Test POST /api/reminders creates a new reminder"""
        payload = {
            "title": "TEST_Take medication",
            "time": "9:00 AM",
            "period": "morning",
            "category": "health"
        }
        response = api_client.post(f"{BASE_URL}/api/reminders", json=payload)
        assert response.status_code == 200
        data = response.json()
        
        assert data["title"] == "TEST_Take medication"
        assert data["completed"] == False
        assert "id" in data
        
        # Cleanup
        api_client.delete(f"{BASE_URL}/api/reminders/{data['id']}")
        print("✓ POST /api/reminders - Create reminder")
    
    def test_toggle_reminder(self, api_client):
        """Test PUT /api/reminders/{id}/toggle toggles completion"""
        # Create test reminder
        create_payload = {
            "title": "TEST_Toggle reminder",
            "time": "10:00 AM",
            "period": "morning",
            "category": "activity"
        }
        create_response = api_client.post(f"{BASE_URL}/api/reminders", json=create_payload)
        reminder_id = create_response.json()["id"]
        
        # Toggle to completed
        toggle_response = api_client.put(f"{BASE_URL}/api/reminders/{reminder_id}/toggle")
        assert toggle_response.status_code == 200
        assert toggle_response.json()["completed"] == True
        
        # Toggle back to incomplete
        toggle_response2 = api_client.put(f"{BASE_URL}/api/reminders/{reminder_id}/toggle")
        assert toggle_response2.status_code == 200
        assert toggle_response2.json()["completed"] == False
        
        # Cleanup
        api_client.delete(f"{BASE_URL}/api/reminders/{reminder_id}")
        print("✓ PUT /api/reminders/{id}/toggle - Toggle completion")
    
    def test_delete_reminder(self, api_client):
        """Test DELETE /api/reminders/{id} removes a reminder"""
        # Create test reminder
        create_payload = {
            "title": "TEST_Delete reminder",
            "time": "11:00 AM",
            "period": "morning",
            "category": "meals"
        }
        create_response = api_client.post(f"{BASE_URL}/api/reminders", json=create_payload)
        reminder_id = create_response.json()["id"]
        
        # Delete
        delete_response = api_client.delete(f"{BASE_URL}/api/reminders/{reminder_id}")
        assert delete_response.status_code == 200
        
        # Verify deletion
        get_response = api_client.get(f"{BASE_URL}/api/reminders")
        reminders = get_response.json()
        deleted = next((r for r in reminders if r["id"] == reminder_id), None)
        assert deleted is None
        print("✓ DELETE /api/reminders - Delete and verify removal")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
