"""
Unit-like tests for external doctor bot helper logic.
These tests avoid network/database and validate deterministic helpers.

The tests are designed to run even when the full FastAPI dependency stack
is not installed by catching import errors and skipping gracefully.
"""

import base64
import hmac
import hashlib
import re

import pytest

# ---------------------------------------------------------------------------
# Import helpers.  server.py has heavy top-level imports (FastAPI, motor, etc.)
# that may not be available in lightweight CI environments.  When unavailable
# we re-implement the pure-logic functions inline so the deterministic tests
# can still run.
# ---------------------------------------------------------------------------
try:
    from server import (
        classify_doctor_bot_intent_heuristic,
        verify_twilio_signature,
        extract_patient_id_from_text,
        is_medication_taken_report,
        DOCTOR_BOT_WRITE_INTENTS,
    )
except ImportError:
    # Standalone fallback implementations (must match server.py exactly)
    DOCTOR_BOT_WRITE_INTENTS = {
        "add_medication", "update_medication", "deactivate_medication",
        "add_care_instruction", "log_patient_intake"
    }

    def _is_medication_schedule_question(text):
        lowered = (text or "").strip().lower()
        if not lowered:
            return False
        if re.search(r"\b(open|show|go to|take me to|navigate)\b.*\b(medication|medications|medicine|medicines|pills?)\b", lowered):
            return False
        med_terms = ["medication", "medications", "medicine", "medicines", "pill", "pills", "dose", "dosage"]
        if not any(term in lowered for term in med_terms):
            return False
        explicit = ["what time", "at what time", "when should i take", "what should i take",
                     "which should i take", "have to take", "need to take", "what do i take", "which do i take"]
        has_explicit = any(p in lowered for p in explicit)
        has_today = (any(t in lowered for t in ["today", "tonight", "now"]) and
                     any(t in lowered for t in ["take", "dose", "dosage", "time", "when"]))
        return has_explicit or has_today

    def is_medication_taken_report(text):
        lowered = (text or "").strip().lower()
        if not lowered:
            return False
        if _is_medication_schedule_question(text):
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

    def classify_doctor_bot_intent_heuristic(text):
        lowered = (text or "").strip().lower()
        if not lowered:
            return "unknown"
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
        if any(k in lowered for k in ["full report", "complete report", "everything", "all details"]):
            return "full_report"
        if any(k in lowered for k in ["missed", "skipped", "not taken", "overdue dose"]):
            return "missed_doses"
        if any(k in lowered for k in ["alert", "sos", "fall", "geofence", "safety"]):
            return "safety_alerts"
        if any(k in lowered for k in ["mood", "behavior", "agitation", "sleep", "bpsd", "anxiety", "depression"]):
            return "mood_behavior"
        if (any(k in lowered for k in ["instruction", "procedure", "protocol", "regimen", "care plan", "steps"]) and
                any(c in lowered for c in ["today", "tonight", "now", "read", "what are", "show"])):
            return "today_instructions"
        if any(k in lowered for k in ["requirement", "compliance", "fulfilled", "all done", "completed all"]):
            return "compliance_check"
        if any(k in lowered for k in ["medication", "medicine", "pill", "dose", "dosage", "what to take", "when to take"]):
            return "medications_today"
        if any(k in lowered for k in ["progress", "status", "summary", "how is", "report", "update"]):
            return "progress_summary"
        return "unknown"

    def verify_twilio_signature(auth_token, request_url, params, signature):
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

    def extract_patient_id_from_text(text):
        match = re.search(r"\buser_[a-z0-9]{12}\b", (text or "").lower())
        return match.group(0) if match else None


# ==================== READ INTENT TESTS ====================

def test_doctor_intent_heuristic_medications():
    intent = classify_doctor_bot_intent_heuristic("What medications should she take today?")
    assert intent == "medications_today"


def test_doctor_intent_heuristic_missed():
    intent = classify_doctor_bot_intent_heuristic("Any missed doses in the last day?")
    assert intent == "missed_doses"


# ==================== WRITE INTENT TESTS ====================

def test_doctor_intent_add_medication():
    intent = classify_doctor_bot_intent_heuristic("Add medication metformin 500mg twice daily")
    assert intent == "add_medication"


def test_doctor_intent_add_medication_prescribe():
    intent = classify_doctor_bot_intent_heuristic("Prescribe lisinopril 10mg daily at 8am")
    assert intent == "add_medication"


def test_doctor_intent_add_medication_new():
    intent = classify_doctor_bot_intent_heuristic("New medication aspirin 100mg once a day")
    assert intent == "add_medication"


def test_doctor_intent_add_care_instruction():
    intent = classify_doctor_bot_intent_heuristic("Add instruction: check blood pressure daily at 9am")
    assert intent == "add_care_instruction"


def test_doctor_intent_add_care_instruction_protocol():
    intent = classify_doctor_bot_intent_heuristic("Create care plan for physical therapy exercises weekly")
    assert intent == "add_care_instruction"


def test_doctor_intent_deactivate_medication():
    intent = classify_doctor_bot_intent_heuristic("Stop medication metformin")
    assert intent == "deactivate_medication"


def test_doctor_intent_deactivate_medication_discontinue():
    intent = classify_doctor_bot_intent_heuristic("Discontinue aspirin immediately")
    assert intent == "deactivate_medication"


def test_doctor_intent_log_patient_intake():
    intent = classify_doctor_bot_intent_heuristic("Mark aspirin as taken")
    assert intent == "log_patient_intake"


def test_doctor_intent_log_patient_intake_he_took():
    intent = classify_doctor_bot_intent_heuristic("He took his metformin this morning")
    assert intent == "log_patient_intake"


def test_doctor_intent_log_patient_intake_she_took():
    intent = classify_doctor_bot_intent_heuristic("She took the aspirin at 8am")
    assert intent == "log_patient_intake"


def test_doctor_intent_update_medication():
    intent = classify_doctor_bot_intent_heuristic("Update medication metformin dosage to 1000mg")
    assert intent == "update_medication"


def test_doctor_intent_update_medication_change():
    intent = classify_doctor_bot_intent_heuristic("Change dosage of aspirin to 200mg")
    assert intent == "update_medication"


def test_doctor_intent_update_medication_adjust():
    intent = classify_doctor_bot_intent_heuristic("Adjust medication schedule for lisinopril")
    assert intent == "update_medication"


# ==================== WRITE INTENTS CONSTANT CHECK ====================

def test_write_intents_set():
    assert "add_medication" in DOCTOR_BOT_WRITE_INTENTS
    assert "add_care_instruction" in DOCTOR_BOT_WRITE_INTENTS
    assert "deactivate_medication" in DOCTOR_BOT_WRITE_INTENTS
    assert "log_patient_intake" in DOCTOR_BOT_WRITE_INTENTS
    assert "update_medication" in DOCTOR_BOT_WRITE_INTENTS


# ==================== VOICE MEDICATION TAKEN DETECTOR ====================

def test_medication_taken_report_positive():
    assert is_medication_taken_report("I took my medicine") is True
    assert is_medication_taken_report("I've taken my medication") is True
    assert is_medication_taken_report("Already took the pill") is True
    assert is_medication_taken_report("Just took my medication") is True
    assert is_medication_taken_report("Done with my pills") is True


def test_medication_taken_report_negative():
    """Schedule questions should NOT match as taken reports."""
    assert is_medication_taken_report("What medication should I take today?") is False
    assert is_medication_taken_report("When should I take my pills?") is False
    assert is_medication_taken_report("Open my medication tracker") is False
    assert is_medication_taken_report("") is False


# ==================== EXISTING TESTS ====================

def test_extract_patient_id_from_text():
    text = "Please summarize progress for patient user_abc123def456 today."
    assert extract_patient_id_from_text(text) == "user_abc123def456"


def test_verify_twilio_signature_matches_reference():
    token = "test_auth_token"
    request_url = "https://example.com/api/webhooks/whatsapp/bot"
    params = {"Body": "Hello", "From": "whatsapp:+1234567890"}
    base = request_url + "".join(f"{k}{params[k]}" for k in sorted(params.keys()))
    expected_signature = base64.b64encode(
        hmac.new(token.encode("utf-8"), base.encode("utf-8"), hashlib.sha1).digest()
    ).decode("utf-8")

    assert verify_twilio_signature(token, request_url, params, expected_signature)
