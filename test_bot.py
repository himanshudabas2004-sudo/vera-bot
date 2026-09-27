"""
Comprehensive test script for bot.py validating all endpoints, composition logic,
the 4 post-generation validation rules, Gemini re-prompting, and fallback behaviors.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
import pytest
from fastapi.testclient import TestClient

import bot
from bot import (
    app,
    contexts,
    conversations,
    conversation_fingerprints,
    conversation_contexts,
    conversation_auto_reply_count,
    merchant_auto_reply_count,
    conversation_incoming_messages,
    compose,
    _deterministic_fallback_compose,
    validate_anti_repetition,
    validate_single_cta,
    validate_fact_grounding,
    validate_language,
    validate_composed_message,
    record_body_fingerprint,
    ComposedMessage,
    REQUEST_TIMEOUT_SECONDS,
)

client = TestClient(app)


def setup_function():
    contexts.clear()
    conversations.clear()
    conversation_fingerprints.clear()
    conversation_contexts.clear()
    conversation_auto_reply_count.clear()
    merchant_auto_reply_count.clear()
    conversation_incoming_messages.clear()


# =============================================================================
# Endpoint Tests
# =============================================================================

def test_healthz_empty():
    res = client.get("/v1/healthz")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "ok"
    assert "uptime_seconds" in data
    assert data["contexts_loaded"] == {"category": 0, "merchant": 0, "customer": 0, "trigger": 0}


def test_metadata():
    res = client.get("/v1/metadata")
    assert res.status_code == 200
    data = res.json()
    assert "team_name" in data
    assert isinstance(data["team_members"], list)
    assert data["model"] == "gemini-2.5-flash"
    assert "approach" in data
    assert "contact_email" in data
    assert "version" in data
    assert "submitted_at" in data


def test_context_push_and_idempotency():
    res = client.post("/v1/context", json={
        "scope": "merchant",
        "context_id": "m_001_drmeera",
        "version": 1,
        "payload": {
            "merchant_id": "m_001_drmeera",
            "category_slug": "dentists",
            "identity": {"name": "Dr. Meera's Clinic"}
        },
        "delivered_at": "2026-04-26T10:00:00Z"
    })
    assert res.status_code == 200
    assert res.json()["accepted"] is True

    # Same version -> 409
    res_stale = client.post("/v1/context", json={
        "scope": "merchant",
        "context_id": "m_001_drmeera",
        "version": 1,
        "payload": {"merchant_id": "m_001_drmeera"},
        "delivered_at": "2026-04-26T10:05:00Z"
    })
    assert res_stale.status_code == 409
    assert res_stale.json()["accepted"] is False
    assert res_stale.json()["reason"] == "stale_version"

    # Higher version -> 200 replaces
    res_higher = client.post("/v1/context", json={
        "scope": "merchant",
        "context_id": "m_001_drmeera",
        "version": 2,
        "payload": {
            "merchant_id": "m_001_drmeera",
            "category_slug": "dentists",
            "identity": {"name": "Dr. Meera's Clinic Updated"}
        },
        "delivered_at": "2026-04-26T10:10:00Z"
    })
    assert res_higher.status_code == 200
    assert contexts[("merchant", "m_001_drmeera")]["version"] == 2


def test_context_invalid_scope():
    res = client.post("/v1/context", json={
        "scope": "invalid_scope_name",
        "context_id": "xyz",
        "version": 1,
        "payload": {},
        "delivered_at": "2026-04-26T10:00:00Z"
    })
    assert res.status_code == 400
    assert res.json()["accepted"] is False
    assert res.json()["reason"] == "invalid_scope"


# =============================================================================
# Post-Generation Validation Layer Unit Tests
# =============================================================================

def test_validation_rule1_anti_repetition():
    conv_id = "conv_rep_01"
    body1 = "Dr. Meera, we noticed 2400 views on your listing. Would you like to review? Reply YES to confirm."
    
    # 1. Fresh message passes
    ok, reason = validate_anti_repetition(body1, conv_id)
    assert ok is True
    assert reason is None

    # Record fingerprint
    record_body_fingerprint(conv_id, body1)

    # 2. Duplicate body (even with varied punctuation or casing) fails
    body_dup = "dr. meera, we noticed 2400 views on your listing. would you like to review? reply yes to confirm!"
    ok_dup, reason_dup = validate_anti_repetition(body_dup, conv_id)
    assert ok_dup is False
    assert "prior message fingerprint" in reason_dup

    # 3. Different message in same conversation passes
    body2 = "Dr. Meera, JIDA Oct 2026 research is out. Want me to draft a summary? Reply YES to see."
    ok2, reason2 = validate_anti_repetition(body2, conv_id)
    assert ok2 is True


def test_validation_rule2_single_cta():
    # 1. Multiple question marks proposing different actions fail
    multi_q = "Do you want to see who looked at your profile? Should we launch a new offer today?"
    ok, reason = validate_single_cta(multi_q)
    assert ok is False
    assert "question marks" in reason

    # 2. Multi-choice reply options fail
    multi_reply = "Reply 1 for Dental Cleaning, Reply 2 for Teeth Whitening."
    ok, reason = validate_single_cta(multi_reply)
    assert ok is False
    assert "multiple distinct" in reason or "multi-option" in reason

    multi_reply2 = "Reply YES for option A or NO for option B."
    ok2, reason2 = validate_single_cta(multi_reply2)
    assert ok2 is False

    # 3. Single clean CTA passes
    single_cta = "Would you like me to feature Dental Cleaning @ ₹299 on your profile? Reply YES to confirm."
    ok_valid, reason_valid = validate_single_cta(single_cta)
    assert ok_valid is True
    assert reason_valid is None


def test_validation_rule3_fact_grounding():
    cat = {"slug": "dentists", "offer_catalog": [{"title": "Dental Cleaning @ ₹299"}]}
    merch = {
        "merchant_id": "m_001_drmeera",
        "identity": {"name": "Dr. Meera Clinic", "locality": "Lajpat Nagar"},
        "performance": {"views": 2400, "calls": 20, "delta_7d": {"views_pct": 0.20}},
        "offers": [{"title": "Dental Cleaning @ ₹299", "status": "active"}]
    }
    trg = {
        "id": "trg_01",
        "kind": "research_digest",
        "payload": {"top_item": {"title": "Trial with 2100 patients", "source": "JIDA Oct 2026"}}
    }

    # 1. Grounded numbers (2400 views, 20% views_pct, 299 price, 2100 trial) pass
    grounded_body = "Dr. Meera, your listing reached 2400 views with 20% growth. We also have 2100 trial data. Reply YES to review."
    ok, ungrounded = validate_fact_grounding(grounded_body, cat, merch, trg, None)
    assert ok is True
    assert ungrounded == []

    # 2. Hallucinated number (e.g. 9999 or 88%) fails
    hallucinated_body = "Dr. Meera, your views jumped by 88% and you missed 9999 search queries. Reply YES to see."
    ok_fail, ungrounded_fail = validate_fact_grounding(hallucinated_body, cat, merch, trg, None)
    assert ok_fail is False
    assert "9999" in ungrounded_fail
    assert "88" in ungrounded_fail


def test_validation_rule4_language():
    # Merchant specifies "hi" language
    merch_hi = {
        "identity": {"name": "Dr. Meera Clinic", "languages": ["en", "hi"]}
    }
    # Customer specifies "hi-en mix"
    cust_hi = {
        "identity": {"name": "Priya", "language_pref": "hi-en mix"}
    }
    merch_en = {
        "identity": {"name": "English Clinic", "languages": ["en"]}
    }

    # 1. Pure English message for Hindi merchant fails
    pure_en = "Dr. Meera, your listing reached 2400 views this month. Would you like to review the report? Reply YES to see."
    ok_en, reason_en = validate_language(pure_en, merch_hi, None)
    assert ok_en is False
    assert "pure English" in reason_en

    # 2. Hinglish message for Hindi merchant passes
    hinglish_msg = "Dr. Meera, aapki listing par 2400 views aaye hain. Kya hum naya post publish karein? Reply YES to confirm."
    ok_hi, reason_hi = validate_language(hinglish_msg, merch_hi, None)
    assert ok_hi is True
    assert reason_hi is None

    # 3. Pure English message for English-only merchant passes
    ok_pure, reason_pure = validate_language(pure_en, merch_en, None)
    assert ok_pure is True
    assert reason_pure is None

    # 4. Hinglish for customer with "hi-en mix" passes
    cust_hinglish = "Hi Priya, aapka clinic checkup due hai. Kya hum slot book karein? Reply YES to confirm."
    ok_cust, _ = validate_language(cust_hinglish, merch_en, cust_hi)
    assert ok_cust is True


# =============================================================================
# Composition & Re-Generation Workflow Tests
# =============================================================================

def test_compose_regeneration_on_validation_failure():
    """
    Simulates Gemini returning an invalid message (hallucinated number + pure English)
    on the first attempt, and a valid corrected message on the second attempt.
    """
    cat = {"slug": "dentists", "offer_catalog": [{"title": "Dental Cleaning @ ₹299"}]}
    merch = {
        "merchant_id": "m_001",
        "identity": {"name": "Dr. Meera Clinic", "languages": ["en", "hi"]},
        "offers": [{"title": "Dental Cleaning @ ₹299", "status": "active"}]
    }
    trg = {"id": "trg_001", "suppression_key": "suppress:001"}

    # Attempt 1: Invalid (pure English + hallucinated number 9999)
    resp1 = MagicMock()
    resp1.text = """{
        "body": "Dr. Meera, we found 9999 missed queries in your area. Would you like to see? Reply YES to review.",
        "cta": "binary_yes_stop",
        "send_as": "vera",
        "suppression_key": "suppress:001",
        "rationale": "Attempt 1"
    }"""

    # Attempt 2: Valid corrected Hinglish without 9999
    resp2 = MagicMock()
    resp2.text = """{
        "body": "Dr. Meera, aapke clinic ke liye Dental Cleaning @ ₹299 ka post ready hai. Kya hum ise publish karein? Reply YES to confirm.",
        "cta": "binary_yes_stop",
        "send_as": "vera",
        "suppression_key": "suppress:001",
        "rationale": "Corrected attempt with Hinglish and grounded price."
    }"""

    mock_client = MagicMock()
    mock_client.aio.models.generate_content = AsyncMock(side_effect=[resp1, resp2])

    with patch("bot._get_gemini_client", return_value=mock_client):
        action = asyncio.run(compose(cat, merch, trg, None, conversation_id="conv_retry_test"))
        # Should have called generate_content twice
        assert mock_client.aio.models.generate_content.call_count == 2
        # Final action is the corrected one
        assert action["body"] == "Dr. Meera, aapke clinic ke liye Dental Cleaning @ ₹299 ka post ready hai. Kya hum ise publish karein? Reply YES to confirm."
        assert "9999" not in action["body"]


def test_compose_fallback_if_regeneration_still_fails():
    """
    Simulates Gemini returning an invalid message on both attempts.
    The system should gracefully fall back to _deterministic_fallback_compose without crashing.
    """
    cat = {"slug": "dentists", "offer_catalog": [{"title": "Dental Cleaning @ ₹299"}]}
    merch = {
        "merchant_id": "m_001",
        "identity": {"name": "Dr. Meera Clinic", "languages": ["en", "hi"]},
        "offers": [{"title": "Dental Cleaning @ ₹299", "status": "active"}]
    }
    trg = {"id": "trg_001", "suppression_key": "suppress:001"}

    invalid_resp = MagicMock()
    invalid_resp.text = """{
        "body": "Invalid message with 8888 and 9999 hallucinated numbers and multiple questions? Will you reply?",
        "cta": "binary_yes_stop",
        "send_as": "vera",
        "suppression_key": "suppress:001",
        "rationale": "Always invalid"
    }"""

    mock_client = MagicMock()
    mock_client.aio.models.generate_content = AsyncMock(return_value=invalid_resp)

    with patch("bot._get_gemini_client", return_value=mock_client):
        action = asyncio.run(compose(cat, merch, trg, None, conversation_id="conv_fail_fallback"))
        # Fallback produced a valid message
        assert action is not None
        assert "Dental Cleaning @ ₹299" in action["body"]
        assert action["send_as"] == "vera"
        assert action["cta"] in ["binary_yes_stop", "open_ended"]


def test_deterministic_fallback_passes_all_checks():
    """Verifies that _deterministic_fallback_compose natively passes all 4 checks."""
    cat = {"slug": "dentists", "offer_catalog": [{"title": "Dental Cleaning @ ₹299"}]}
    merch = {
        "merchant_id": "m_001_drmeera",
        "identity": {"name": "Dr. Meera Clinic", "owner_first_name": "Meera", "locality": "Lajpat Nagar", "languages": ["en", "hi"]},
        "performance": {"views": 2400, "calls": 20, "delta_7d": {"views_pct": 0.20}},
        "offers": [{"title": "Dental Cleaning @ ₹299", "status": "active"}]
    }
    trg = {
        "id": "trg_digest_01",
        "kind": "research_digest",
        "suppression_key": "suppress:01",
        "payload": {"top_item": {"title": "Fluoride trial", "source": "JIDA Oct 2026"}}
    }

    result = _deterministic_fallback_compose(cat, merch, trg, None)
    is_valid, failures = validate_composed_message(result, cat, merch, trg, None, "conv_det_test")
    assert is_valid is True, f"Deterministic fallback failed checks: {failures}"


def test_tick_and_conversation_store():
    client.post("/v1/context", json={
        "scope": "category",
        "context_id": "dentists",
        "version": 1,
        "payload": {
            "slug": "dentists",
            "voice": {"tone": "peer_clinical"},
            "digest": [{"title": "Fluoride recall trial", "source": "JIDA Oct 2026"}]
        },
        "delivered_at": "2026-04-26T10:00:00Z"
    })
    client.post("/v1/context", json={
        "scope": "merchant",
        "context_id": "m_001_drmeera",
        "version": 1,
        "payload": {
            "merchant_id": "m_001_drmeera",
            "category_slug": "dentists",
            "identity": {"name": "Dr. Meera's Dental Clinic", "owner_first_name": "Meera", "languages": ["en", "hi"]},
            "offers": [{"title": "Dental Cleaning @ ₹299", "status": "active"}]
        },
        "delivered_at": "2026-04-26T10:00:00Z"
    })
    client.post("/v1/context", json={
        "scope": "trigger",
        "context_id": "trg_001",
        "version": 1,
        "payload": {
            "id": "trg_001",
            "scope": "merchant",
            "kind": "research_digest",
            "merchant_id": "m_001_drmeera",
            "suppression_key": "suppress:dentists:001",
            "urgency": 2
        },
        "delivered_at": "2026-04-26T10:00:00Z"
    })

    res = client.post("/v1/tick", json={
        "now": "2026-04-26T10:30:00Z",
        "available_triggers": ["trg_001"]
    })
    assert res.status_code == 200
    data = res.json()
    assert "actions" in data
    assert len(data["actions"]) == 1
    action = data["actions"][0]
    assert action["merchant_id"] == "m_001_drmeera"
    assert action["trigger_id"] == "trg_001"
    assert "body" in action and len(action["body"]) > 0
    assert action["cta"] in ["binary_yes_stop", "open_ended", "none"]
    assert "rationale" in action
    assert "conversation_id" in action


def test_reply_auto_reply_first_and_second_occurrence():
    conv_id = "conv_test_auto"
    # 1st detection: returns action: "send" with short human nudge
    res1 = client.post("/v1/reply", json={
        "conversation_id": conv_id,
        "merchant_id": "m_001_drmeera",
        "customer_id": None,
        "from_role": "merchant",
        "message": "Thank you for contacting us! Our team will respond shortly.",
        "received_at": "2026-04-26T10:45:00Z",
        "turn_number": 2
    })
    assert res1.status_code == 200
    data1 = res1.json()
    assert data1["action"] == "send"
    assert any(term in data1["body"].lower() for term in ["2 minute", "quick", "look", "samajh", "ready"])
    assert "nudge" in data1.get("rationale", "").lower()

    # 2nd detection in same conversation: returns action: "end" with polite closing
    res2 = client.post("/v1/reply", json={
        "conversation_id": conv_id,
        "merchant_id": "m_001_drmeera",
        "customer_id": None,
        "from_role": "merchant",
        "message": "Our automated assistant has received your query. We will get back to you shortly.",
        "received_at": "2026-04-26T10:46:00Z",
        "turn_number": 3
    })
    assert res2.status_code == 200
    data2 = res2.json()
    assert data2["action"] == "end"
    assert any(term in data2["body"].lower() for term in ["owner", "manager", "best wishes", "connect", "shubh din"])


def test_reply_verbatim_identical_incoming_auto_reply():
    conv_id = "conv_test_verbatim"
    # Turn 2: Non-canned message
    res1 = client.post("/v1/reply", json={
        "conversation_id": conv_id,
        "merchant_id": "m_001_drmeera",
        "customer_id": None,
        "from_role": "merchant",
        "message": "Please send more details.",
        "received_at": "2026-04-26T10:45:00Z",
        "turn_number": 2
    })
    assert res1.status_code == 200
    assert res1.json()["action"] == "send"

    # Turn 3: 2nd verbatim-identical incoming message -> 1st auto-reply detection (action: send nudge)
    res2 = client.post("/v1/reply", json={
        "conversation_id": conv_id,
        "merchant_id": "m_001_drmeera",
        "customer_id": None,
        "from_role": "merchant",
        "message": "Please send more details.",
        "received_at": "2026-04-26T10:46:00Z",
        "turn_number": 3
    })
    assert res2.status_code == 200
    assert res2.json()["action"] == "send"

    # Turn 4: 3rd verbatim-identical incoming message -> 2nd auto-reply detection (action: end)
    res3 = client.post("/v1/reply", json={
        "conversation_id": conv_id,
        "merchant_id": "m_001_drmeera",
        "customer_id": None,
        "from_role": "merchant",
        "message": "Please send more details.",
        "received_at": "2026-04-26T10:47:00Z",
        "turn_number": 4
    })
    assert res3.status_code == 200
    assert res3.json()["action"] == "end"


def test_reply_intent_handoff():
    conv_id = "conv_test_intent"
    res = client.post("/v1/reply", json={
        "conversation_id": conv_id,
        "merchant_id": "m_001_drmeera",
        "customer_id": None,
        "from_role": "merchant",
        "message": "Ok lets do it. Whats next?",
        "received_at": "2026-04-26T10:45:00Z",
        "turn_number": 2
    })
    assert res.status_code == 200
    data = res.json()
    assert data["action"] == "send"
    body_lower = data["body"].lower()
    # Must use actioning language per Pattern D
    assert any(w in body_lower for w in ["done", "proceed", "sending", "next", "confirm", "draft"])
    # Must NOT re-qualify or ask hesitation questions
    for q in ["would you", "can you tell", "what if", "how about"]:
        assert q not in body_lower


def test_reply_decline_objection():
    conv_id = "conv_test_decline"
    res = client.post("/v1/reply", json={
        "conversation_id": conv_id,
        "merchant_id": "m_001_drmeera",
        "customer_id": None,
        "from_role": "merchant",
        "message": "Stop messaging me. This is useless spam, not interested.",
        "received_at": "2026-04-26T10:45:00Z",
        "turn_number": 2
    })
    assert res.status_code == 200
    data = res.json()
    assert data["action"] == "end"
    body_lower = data["body"].lower()
    assert any(term in body_lower for term in ["understood", "samajh", "disturb", "best", "shubh din"])
    assert "cleaning" not in body_lower
    assert "offer" not in body_lower


def test_reply_hostile():
    conv_id = "conv_test_hostile"
    res = client.post("/v1/reply", json={
        "conversation_id": conv_id,
        "merchant_id": "m_001_drmeera",
        "customer_id": None,
        "from_role": "merchant",
        "message": "Stop messaging me. This is useless spam.",
        "received_at": "2026-04-26T10:45:00Z",
        "turn_number": 2
    })
    assert res.status_code == 200
    assert res.json()["action"] == "end"


def test_reply_turn_depth_guard():
    conv_id = "conv_test_depth"
    res = client.post("/v1/reply", json={
        "conversation_id": conv_id,
        "merchant_id": "m_001_drmeera",
        "customer_id": None,
        "from_role": "merchant",
        "message": "Can you give me more details about that?",
        "received_at": "2026-04-26T10:50:00Z",
        "turn_number": 5
    })
    assert res.status_code == 200
    data = res.json()
    assert data["action"] == "end"
    assert "turn depth limit" in data.get("rationale", "").lower()


def test_timeout_fallback_simulation():
    original_compose = bot.compose
    
    async def hanging_compose(*args, **kwargs):
        await asyncio.sleep(100)
    
    bot.compose = hanging_compose
    bot.REQUEST_TIMEOUT_SECONDS = 0.05

    client.post("/v1/context", json={
        "scope": "merchant",
        "context_id": "m_test",
        "version": 1,
        "payload": {"merchant_id": "m_test"},
        "delivered_at": "2026-04-26T10:00:00Z"
    })
    client.post("/v1/context", json={
        "scope": "trigger",
        "context_id": "trg_hang",
        "version": 1,
        "payload": {"id": "trg_hang", "merchant_id": "m_test"},
        "delivered_at": "2026-04-26T10:00:00Z"
    })

    res_tick = client.post("/v1/tick", json={
        "now": "2026-04-26T10:30:00Z",
        "available_triggers": ["trg_hang"]
    })
    assert res_tick.status_code == 200
    assert res_tick.json() == {"actions": []}

    # Restore
    bot.compose = original_compose
    bot.REQUEST_TIMEOUT_SECONDS = 25.0


def test_trg_003_recall_due_priya_elapsed_time():
    """
    Test using trg_003_recall_due_priya that asserts the output body contains
    a specific time-elapsed reference derived from customer.relationship.last_visit,
    not just a generic 'due' statement.
    """
    import json
    from pathlib import Path

    base_dir = Path(__file__).parent
    cat = json.load(open(base_dir / "dataset/categories/dentists.json"))
    merchants = json.load(open(base_dir / "dataset/merchants_seed.json"))["merchants"]
    merch = next(m for m in merchants if m["merchant_id"] == "m_001_drmeera_dentist_delhi")
    triggers = json.load(open(base_dir / "dataset/triggers_seed.json"))["triggers"]
    trg = next(t for t in triggers if t["id"] == "trg_003_recall_due_priya")
    customers = json.load(open(base_dir / "dataset/customers_seed.json"))["customers"]
    cust = next(c for c in customers if c["customer_id"] == "c_001_priya_for_m001")

    # Assert customer relationship has last_visit
    assert cust["relationship"]["last_visit"] == "2026-05-12"

    # Call compose
    action = asyncio.run(compose(cat, merch, trg, cust))
    body = action["body"]

    # 1. Assert specific time-elapsed reference derived from customer.relationship.last_visit
    body_lower = body.lower()
    has_elapsed = (
        "5 months" in body_lower
        or "5 mahine" in body_lower
        or "6 months" in body_lower
        or "6 mahine" in body_lower
        or "since your last visit" in body_lower
        or "last visit ko" in body_lower
    )
    assert has_elapsed, f"Body does not contain specific time-elapsed reference derived from last_visit: {body}"

    # 2. Assert output is NOT just a generic 'due' statement
    assert "visit update due hai" not in body_lower, f"Body contains banned vague phrase: {body}"
    assert "time for a check" not in body_lower, f"Body contains banned vague phrase: {body}"
    assert "worth looking at" not in body_lower, f"Body contains banned vague phrase: {body}"
    assert "check-in is due" not in body_lower, f"Body contains banned vague phrase: {body}"

    # 3. Assert it mentions real catalog price and clinic attribution
    assert "₹299" in body or "299" in body, f"Body missing grounded offer price: {body}"
    assert action["send_as"] == "merchant_on_behalf"

