"""
Voice intent regression tests for paraphrase robustness.

These tests call the live backend, similar to test_api.py.
They validate high-priority voice intents keep working when wording changes.
"""

import os
import uuid

import pytest
import requests

BASE_URL = os.environ.get("REACT_APP_BACKEND_URL", "").rstrip("/")
SESSION_TOKEN = os.environ.get(
    "VOICE_TEST_SESSION_TOKEN",
    "sess_test_eba950c34a0140878f212a3ad58d66d2"
)

if not BASE_URL:
    pytest.skip("REACT_APP_BACKEND_URL is not configured for integration tests.", allow_module_level=True)


@pytest.fixture
def api_client():
    session = requests.Session()
    session.headers.update(
        {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {SESSION_TOKEN}"
        }
    )
    return session


def call_voice(api_client, text):
    response = api_client.post(
        f"{BASE_URL}/api/voice-command",
        json={
            "text": text,
            "session_id": f"voice_regression_{uuid.uuid4().hex[:10]}"
        },
    )
    assert response.status_code == 200, response.text
    payload = response.json()
    assert "response" in payload
    return payload


@pytest.mark.parametrize(
    "phrase",
    [
        "I would like to play chess",
        "Can we play a chess game now?",
        "I want to practice chess"
    ],
)
def test_voice_unsupported_chess_fallback(api_client, phrase):
    payload = call_voice(api_client, phrase)
    assert payload.get("action") == "speak"
    text = (payload.get("response") or "").lower()
    assert "chess" in text
    assert any(token in text for token in ["faces quiz", "memory match", "sudoku"])


@pytest.mark.parametrize(
    "phrase",
    [
        "Tell me what medicines I have to take today and at what time",
        "What pills should I take now?",
        "Which medication doses are scheduled for today?"
    ],
)
def test_voice_medication_schedule_paraphrases(api_client, phrase):
    payload = call_voice(api_client, phrase)
    assert payload.get("action") == "speak"
    assert payload.get("target") in (None, "")
    text = (payload.get("response") or "").lower()
    assert any(
        token in text
        for token in [
            "medication",
            "medications",
            "dose",
            "doses",
            "take",
            "scheduled",
            "regimen"
        ]
    )


@pytest.mark.parametrize(
    "phrase",
    [
        "Read my instructions for today",
        "What are my care procedures today?",
        "Tell me today's care plan"
    ],
)
def test_voice_today_instruction_paraphrases(api_client, phrase):
    payload = call_voice(api_client, phrase)
    assert payload.get("action") == "speak"
    text = (payload.get("response") or "").lower()
    assert any(
        token in text
        for token in [
            "today",
            "instruction",
            "care",
            "procedure",
            "plan",
            "medication"
        ]
    )


@pytest.mark.parametrize(
    "phrase",
    [
        "Open my medication tracker",
        "Show medications page",
    ],
)
def test_voice_explicit_medication_navigation(api_client, phrase):
    payload = call_voice(api_client, phrase)
    assert payload.get("action") == "navigate"
    assert payload.get("target") == "medications"


# ==================== MEDICATION INTAKE VOICE TESTS ====================

@pytest.mark.parametrize(
    "phrase",
    [
        "I took my medicine",
        "I've taken my medication",
        "I already took my pill",
        "Just took the medicine",
        "I had my medication",
        "Done with my pills",
    ],
)
def test_voice_medication_taken_report(api_client, phrase):
    """Verify that medication taken reports are recognized and handled."""
    payload = call_voice(api_client, phrase)
    assert payload.get("action") == "speak"
    text = (payload.get("response") or "").lower()
    # Should either confirm logging or say no pending doses
    assert any(
        token in text
        for token in [
            "recorded", "got it", "taken", "logged",
            "all your", "doses", "scheduled", "track",
            "medication", "trouble"
        ]
    )


# ==================== MOOD REPORT VOICE TESTS ====================

@pytest.mark.parametrize(
    "phrase",
    [
        "I feel happy today",
        "I'm feeling a bit sad",
        "I feel tired and low energy",
        "I'm doing great today",
    ],
)
def test_voice_mood_report(api_client, phrase):
    """Verify that mood reports are recognized and logged."""
    payload = call_voice(api_client, phrase)
    assert payload.get("action") == "speak"
    text = (payload.get("response") or "").lower()
    assert any(
        token in text
        for token in [
            "thank you", "noted", "feeling", "mood",
            "caregiver", "sharing"
        ]
    )
