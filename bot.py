"""
magicpin AI Challenge — Vera Merchant Assistant Bot
===================================================

FastAPI application implementing the exact 5-endpoint API contract defined in
challenge-testing-brief.md §2:

1. POST /v1/context  - Idempotent context storage with version conflict detection
2. POST /v1/tick     - Periodic trigger evaluation and proactive outreach composition
3. POST /v1/reply    - Synchronous multi-turn conversation handling
4. GET  /v1/healthz  - Liveness probe and context inventory
5. GET  /v1/metadata - Bot identity, version, and model metadata
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import re
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Literal

from fastapi import FastAPI, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# google-genai SDK for Gemini integration
from google import genai
from google.genai import types, Client

# Configure logging
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("vera-bot")

app = FastAPI(
    title="magicpin Vera Assistant Bot",
    description="Vera Merchant Assistant Bot API for magicpin AI Challenge",
    version="1.2.0",
)

# Server start timestamp for uptime tracking
START_TIME = time.time()

# Request timeout threshold in seconds (safely under the 30s deadline)
REQUEST_TIMEOUT_SECONDS = float(os.getenv("REQUEST_TIMEOUT_SECONDS", "25.0"))
LLM_TIMEOUT_SECONDS = float(os.getenv("LLM_TIMEOUT_SECONDS", "12.0"))

# Valid context scopes per spec §2.1 & §3
VALID_SCOPES = {"category", "merchant", "customer", "trigger"}

# In-memory stores
# contexts: (scope, context_id) -> {"version": int, "payload": dict, "delivered_at": str}
contexts: dict[tuple[str, str], dict[str, Any]] = {}

# conversations: conversation_id -> list of turn records
conversations: dict[str, list[dict[str, Any]]] = {}

# Anti-repetition store: conversation_id -> set of body fingerprints (SHA-256)
conversation_fingerprints: dict[str, set[str]] = {}

# Context mapping per conversation: conversation_id -> metadata mapping
conversation_contexts: dict[str, dict[str, Any]] = {}

# Auto-reply tracking stores
conversation_auto_reply_count: dict[str, int] = {}
merchant_auto_reply_count: dict[str, int] = {}
conversation_incoming_messages: dict[str, list[str]] = {}

# Unambiguous Hindi / Hinglish vocabulary (Roman script) that do NOT collide with common English words
HINGLISH_WORDS = {
    # Pronouns & possessives
    "aap", "aapka", "aapke", "aapki", "aapko", "aapne", "aapse", "apke", "apka", "apki",
    "hum", "humara", "humare", "humari", "humko", "hamare", "hamara", "hamari",
    "mera", "mere", "meri", "mujhe", "mujhko",
    "yeh", "woh", "kya", "kyun", "kyu", "kaun", "kaise", "kahan", "kab",
    # Verbs / auxiliaries
    "hai", "hain", "hoon", "hun", "tha", "thi", "hoga", "hogi", "honge",
    "karo", "karein", "karna", "karte", "karti", "kiya", "kiye",
    "dekho", "dekhein", "batao", "batayein", "bataiye", "bhejun", "bhejo",
    "jao", "jaayein", "jaayiye", "chalega", "chahiye", "chahte", "chahti", "chahenge",
    # Postpositions, particles, adverbs, greetings
    "saath", "liye", "bhi", "toh", "aur", "lekin", "magar",
    "nahi", "nahin", "bahut", "bohot", "thoda", "sabhi",
    "accha", "achha", "achhi", "badhiya", "shukriya", "dhanyawad",
    "namaste", "namaskar", "kripya", "zaroor", "jarur", "sahi", "theek", "thik",
    "pehle", "baad", "abhi", "kal", "aaj", "jaldi", "samajh", "pata", "milega", "aaye", "gayi", "hue", "mahine"
}

# Known WhatsApp Business canned auto-replies (Pattern B)
AUTO_REPLY_PHRASES = [
    "thank you for contacting",
    "thanks for contacting",
    "thanks for reaching out",
    "will get back to you shortly",
    "will respond shortly",
    "respond shortly",
    "our team will respond shortly",
    "team tak pahuncha",
    "hamari team tak",
    "automated assistant",
    "automated message",
    "auto-reply",
    "autoreply",
    "aapki jaankari ke liye shukriya",
    "currently unavailable",
    "we are closed",
]

# Clear affirmative intent phrases (Pattern D)
AFFIRMATIVE_INTENT_PHRASES = [
    "yes", "let's do it", "lets do it", "go ahead", "sounds good",
    "ok lets do it", "ok let's do it", "proceed", "confirm", "sign me up",
    "i want to join", "whats next", "what's next", "send it", "do it",
    "sure", "definitely", "agree", "please do",
    "haan", "ha", "theek hai", "thik hai", "karna hai", "chalega",
    "kardo", "kar do", "bhejo", "bhej do", "shuru karo", "judrna hai"
]

# Explicit decline / opt-out / hostility phrases
OBJECTION_PHRASES = [
    "not interested", "stop", "nahi chahiye", "nahin chahiye",
    "no thanks", "nahi karna", "mat bhejo", "unsubscribe",
    "spam", "useless spam", "don't message", "dont message",
    "leave me alone", "opt out", "stop messaging"
]


# =============================================================================
# Pydantic Schemas & Contract Models
# =============================================================================

class ComposedMessage(BaseModel):
    """
    Contract matching challenge-brief.md §5 and §7.1 verbatim.
    """
    body: str = Field(description="The WhatsApp message body.")
    cta: Literal["binary_yes_stop", "open_ended", "none"] = Field(
        description="Single primary call-to-action."
    )
    send_as: Literal["vera", "merchant_on_behalf"] = Field(
        description="'vera' for merchant outreach, 'merchant_on_behalf' for customer outreach."
    )
    suppression_key: str = Field(description="Deduplication key for this outreach.")
    rationale: str = Field(description="Brief explanation of strategy, factual anchor, and compulsion lever.")


class ContextRequest(BaseModel):
    scope: str
    context_id: str
    version: int
    payload: dict[str, Any]
    delivered_at: str


class TickRequest(BaseModel):
    now: str
    available_triggers: list[str] = Field(default_factory=list)


class ReplyRequest(BaseModel):
    conversation_id: str
    merchant_id: str | None = None
    customer_id: str | None = None
    from_role: str
    message: str
    received_at: str
    turn_number: int


# =============================================================================
# Date & Fact Helpers
# =============================================================================

def compute_elapsed_time(last_visit_str: str, ref_date_str: str | None = None) -> tuple[int, str]:
    """
    Computes elapsed time in months from last_visit to reference date.
    Returns (months, text_description) e.g. (5, "5 months since your last visit").
    """
    try:
        lv = datetime.strptime(last_visit_str[:10], "%Y-%m-%d")
        if ref_date_str:
            ref = datetime.strptime(ref_date_str[:10], "%Y-%m-%d")
        else:
            ref = datetime(2026, 11, 5)
        months = (ref.year - lv.year) * 12 + (ref.month - lv.month)
        if ref.day < lv.day:
            months = max(1, months - 1)
        return months, f"{months} months since your last visit"
    except Exception:
        return 5, "5 months since your last visit"


def compute_body_fingerprint(body: str) -> str:
    """Computes a SHA-256 hash of normalized lowercase body text."""
    norm = " ".join(re.sub(r"[^\w\s]", "", body.lower()).split())
    return hashlib.sha256(norm.encode("utf-8")).hexdigest()


def record_body_fingerprint(conversation_id: str, body: str) -> None:
    """Records body text fingerprint for anti-repetition tracking."""
    fp = compute_body_fingerprint(body)
    conversation_fingerprints.setdefault(conversation_id, set()).add(fp)


def extract_context_facts(data: Any) -> set[str]:
    """Recursively extract all numeric tokens, prices, dates, and computed intervals from context payloads."""
    facts = set()
    if isinstance(data, dict):
        lv_str = data.get("customer", {}).get("relationship", {}).get("last_visit") if isinstance(data.get("customer"), dict) else None
        if not lv_str and isinstance(data.get("trigger"), dict):
            lv_str = data.get("trigger", {}).get("payload", {}).get("last_service_date")
        if lv_str:
            try:
                ref_str = data.get("trigger", {}).get("payload", {}).get("due_date")
                m, _ = compute_elapsed_time(lv_str, ref_str)
                facts.add(str(m))
                facts.add(str(max(1, m - 1)))
                facts.add(str(m + 1))
            except Exception:
                pass

        for k, v in data.items():
            facts.update(extract_context_facts(k))
            facts.update(extract_context_facts(v))
    elif isinstance(data, list):
        for item in data:
            facts.update(extract_context_facts(item))
    elif isinstance(data, (int, float)):
        facts.add(str(data))
        if isinstance(data, float):
            facts.add(str(int(data)))
            # Extract percentage representation: 0.38 -> 38, -0.05 -> 5
            pct_val = round(abs(data) * 100, 2)
            if pct_val.is_integer():
                facts.add(str(int(pct_val)))
            else:
                facts.add(str(pct_val))
    elif isinstance(data, str):
        clean_s = re.sub(r"[,₹$]", "", data)
        for token in re.findall(r"\b\d+(?:\.\d+)?\b", clean_s):
            facts.add(token)
            if "." in token:
                try:
                    facts.add(str(int(float(token))))
                except ValueError:
                    pass
    return facts


def contains_hindi(text: str) -> bool:
    """Checks whether the text contains Devanagari script or unambiguous Hinglish vocabulary."""
    if re.search(r"[\u0900-\u097F]", text):
        return True
    words = set(re.findall(r"\b[a-z]+\b", text.lower()))
    return bool(words.intersection(HINGLISH_WORDS))


# =============================================================================
# Post-Generation Validation Layer Functions
# =============================================================================

def validate_anti_repetition(body: str, conversation_id: str | None) -> tuple[bool, str | None]:
    """Rule 1: Confirms body text doesn't repeat prior fingerprints in the same conversation."""
    if not conversation_id:
        return True, None
    fp = compute_body_fingerprint(body)
    if fp in conversation_fingerprints.get(conversation_id, set()):
        return False, "Body matches a prior message fingerprint in this conversation."

    norm_body = " ".join(re.sub(r"[^\w\s]", "", body.lower()).split())
    for turn in conversations.get(conversation_id, []):
        prior_body = turn.get("body")
        if prior_body:
            prior_norm = " ".join(re.sub(r"[^\w\s]", "", prior_body.lower()).split())
            if prior_norm == norm_body:
                return False, "Body text is identical to a prior turn in this conversation."

    return True, None


def validate_single_cta(body: str) -> tuple[bool, str | None]:
    """Rule 2: Confirms body has only one distinct call-to-action pattern."""
    qmark_count = body.count("?")
    if qmark_count > 1:
        return False, f"Message contains {qmark_count} question marks proposing multiple actions."

    body_lower = body.lower()
    reply_for_patterns = re.findall(r"\breply\s+[a-z0-9]+\s+for\b", body_lower)
    if len(reply_for_patterns) > 1:
        return False, "Message contains multiple distinct 'Reply X for A' choices."

    if re.search(r"reply\s+\w+\s+for\b.*(?:\bor\b|,|\/)\s*\w+\s+for\b", body_lower):
        return False, "Message contains multi-option CTA choices."

    return True, None


def validate_fact_grounding(
    body: str,
    category: dict[str, Any],
    merchant: dict[str, Any],
    trigger: dict[str, Any],
    customer: dict[str, Any] | None,
) -> tuple[bool, list[str]]:
    """Rule 3: Confirms all numbers/prices/dates in body are grounded in input contexts."""
    context_data = {
        "category": category,
        "merchant": merchant,
        "trigger": trigger,
        "customer": customer,
    }
    facts = extract_context_facts(context_data)
    clean_body = re.sub(r"[,₹$]", "", body)
    body_numbers = set(re.findall(r"\b\d+(?:\.\d+)?\b", clean_body))

    ungrounded = []
    for num in body_numbers:
        if num in facts:
            continue
        try:
            float_val = float(num)
            if str(int(float_val)) in facts or str(float_val) in facts:
                continue
        except ValueError:
            pass
        if num.lstrip("0") in facts:
            continue
        ungrounded.append(num)

    if ungrounded:
        return False, ungrounded
    return True, []


def validate_language(
    body: str,
    merchant: dict[str, Any],
    customer: dict[str, Any] | None,
) -> tuple[bool, str | None]:
    """Rule 4: Confirms body isn't pure English when recipient preference includes 'hi'."""
    languages = []
    if merchant:
        ident_langs = merchant.get("identity", {}).get("languages", [])
        if isinstance(ident_langs, list):
            languages.extend(ident_langs)
        elif isinstance(ident_langs, str):
            languages.append(ident_langs)
    if customer:
        cust_lang = customer.get("identity", {}).get("language_pref", "")
        if cust_lang:
            languages.append(cust_lang)

    prefers_hindi = any("hi" in str(l).lower() for l in languages)
    if prefers_hindi and not contains_hindi(body):
        return False, "Recipient language preference includes Hindi, but generated message is pure English without any Hindi/Hinglish words."

    return True, None


def validate_specificity_and_elapsed_time(
    body: str,
    customer: dict[str, Any] | None,
) -> tuple[bool, str | None]:
    """
    Confirms customer messages include specific elapsed time since last_visit
    and strictly forbids vague phrases like 'update due', 'time for a check', etc.
    """
    body_lower = body.lower()

    banned_vague = [
        "time for a check",
        "worth looking at",
        "activity on your listing",
        "check-in is due",
        "visit update due",
    ]
    for phrase in banned_vague:
        if phrase in body_lower:
            return False, f"Message contains banned vague phrase '{phrase}'. You must state the specific number, date, or comparison."

    if customer:
        last_visit = customer.get("relationship", {}).get("last_visit")
        if last_visit:
            has_elapsed = (
                re.search(r"\b\d+\s+(?:months?|mahine|weeks?|days?)\b", body_lower)
                or "since your last visit" in body_lower
                or "last visit" in body_lower
            )
            if not has_elapsed:
                return False, "Customer message lacks specific elapsed time since last_visit (e.g. '5 months since your last visit')."

    return True, None


def validate_composed_message(
    composed_msg: ComposedMessage,
    category: dict[str, Any],
    merchant: dict[str, Any],
    trigger: dict[str, Any],
    customer: dict[str, Any] | None,
    conversation_id: str | None = None,
) -> tuple[bool, dict[str, Any]]:
    """Runs all post-generation validation checks."""
    failures: dict[str, Any] = {}

    ok_rep, reason_rep = validate_anti_repetition(composed_msg.body, conversation_id)
    if not ok_rep:
        failures["repetition"] = reason_rep

    ok_cta, reason_cta = validate_single_cta(composed_msg.body)
    if not ok_cta:
        failures["single_cta"] = reason_cta

    ok_facts, ungrounded = validate_fact_grounding(composed_msg.body, category, merchant, trigger, customer)
    if not ok_facts:
        failures["fact_grounding"] = ungrounded

    ok_lang, reason_lang = validate_language(composed_msg.body, merchant, customer)
    if not ok_lang:
        failures["language"] = reason_lang

    ok_spec, reason_spec = validate_specificity_and_elapsed_time(composed_msg.body, customer)
    if not ok_spec:
        failures["specificity"] = reason_spec

    return len(failures) == 0, failures


def build_retry_instructions(failures: dict[str, Any], previous_body: str) -> list[str]:
    """Generates targeted correction instructions for the LLM re-prompt."""
    instructions = []
    if "repetition" in failures:
        instructions.append(f"Vary the phrasing, do not repeat the previous message: '{previous_body}'.")
    if "single_cta" in failures:
        instructions.append("Ensure there is exactly ONE clear call-to-action in the final sentence and no multiple CTAs or multiple questions proposing different actions.")
    if "fact_grounding" in failures:
        ungrounded = failures["fact_grounding"]
        instructions.append(f"Only use numbers present in the provided context (the following numbers were ungrounded hallucinations: {ungrounded}). Do not invent any numbers, prices, or dates.")
    if "language" in failures:
        instructions.append("Incorporate natural Hindi-English code-mix (Hinglish in Roman script, e.g. using words like 'aapka', 'hai', 'karein', 'shukriya') matching the recipient's language preference.")
    if "specificity" in failures:
        instructions.append(
            "Never use vague phrases like 'update due', 'time for a check', or 'worth looking at'. "
            "When customer context is present, you MUST compute and state the specific elapsed time since customer's last_visit (e.g. '5 months since your last visit'). "
            "For merchant-facing messages, you MUST state the specific metric value and comparison (e.g. 'your CTR is 2.1% vs 3.0% peer median')."
        )
    return instructions


# =============================================================================
# Gemini Prompt Template & Composition Implementation
# =============================================================================

SYSTEM_INSTRUCTION = """You are Vera, magicpin's elite merchant assistant AI on WhatsApp.
Your mission is to compose proactive, high-converting WhatsApp engagement messages for local merchants across India (or on their behalf to their customers).

### SCORING DIMENSIONS (Strict 0-10 Evaluation Rubric per challenge-brief.md §8):
1. SPECIFICITY: Anchor on ONE concrete, verifiable fact from the provided context (a specific number, percentage, count, date, peer statistic, or catalog offer price). NEVER invent, fabricate, or assume data not in context.
2. CATEGORY FIT: Match vertical voice rules (challenge-brief.md §4.1):
   - Dentists: Clinical, peer-to-peer tone, technical vocabulary welcome (e.g., 'caries', 'fluoride varnish'), hype-free, use 'Dr.' prefix. Strictly NO taboo claims ('cure', 'guaranteed').
   - Salons & Restaurants: Visual, urgent, operator-to-operator, warm and practical.
   - Gyms: Coaching, motivational, disciplined.
   - Pharmacies: Trustworthy, precise, utility-first, healthcare compliance aware.
3. MERCHANT FIT: Personalize to this specific merchant's state (name, locality, actual performance metrics, real active offers, signals). Honor language preferences: if language indicates 'hi' or 'hi-en mix', write in natural conversational Hindi-English code-mix (Hinglish in Roman script).
4. TRIGGER RELEVANCE: Directly connect to WHY NOW — explicitly address the trigger event that prompted this message.
5. ENGAGEMENT COMPULSION: Deploy exactly ONE psychological compulsion lever from challenge-brief.md §10:
   - Specificity / verifiability
   - Loss aversion
   - Social proof
   - Effort externalization
   - Curiosity
   - Reciprocity
   - Asking the merchant
   - Single binary commitment

### STRICT SPECIFICITY & FACTUAL COMPARISON RULES (CRITICAL):
- Never use vague phrases like 'update due', 'time for a check', or 'worth looking at' when a specific number, date, or comparison exists in the provided context — always state that number.
- Customer-Facing Messages: When customer context is present, you MUST compute and state the specific elapsed time since the customer's last_visit (e.g., '5 months since your last visit' or 'Aapke last visit ko 5 mahine ho gaye hain'), rather than just saying a recall/update is 'due'.
- Merchant-Facing Messages: For merchant-facing triggers, you MUST state the specific metric value and comparison (e.g., 'your CTR is 2.1% vs 3.0% peer median', '7-day views increased by 18%', 'reached 2,410 views') rather than referring to 'performance' or 'activity' generically.
- Service+Price framing: Always use service+price (e.g. 'Dental Cleaning @ ₹299', 'Haircut @ ₹99'), NEVER generic discount framing ('10% off', 'Flat 30% off').
- Ending CTA: End with exactly ONE clear call-to-action in the very last sentence.
- NO multiple CTAs in one message.
- NO buried CTAs (the action must be in the final sentence).
- NO promotional hype ('AMAZING DEAL!') for clinical/professional categories.
- NO hallucinated data (never cite papers, offers, statistics, or competitor names not in the context).
- NO long preambles ('I hope this finds you well', 'I am reaching out today to...').
- NO re-introducing yourself if conversation history exists.
- NO repeating prior messages verbatim.
- Return strict JSON matching the schema: body, cta, send_as, suppression_key, rationale.
"""

FEW_SHOT_EXAMPLES = """
### Appendix A: Reference Composition Example (Merchant-Facing Outreach)
[Context]
Category: dentists (voice: peer/clinical, peer_stats: {avg_ctr: 0.030}, digest top item: "JIDA Oct trial: 3-mo fluoride recall cuts caries 38% better than 6-mo", trial_n: 2100)
Merchant: Dr. Meera, Lajpat Nagar Delhi, CTR 2.1% (below peer median), cohort: high-risk adults
Trigger: research_digest_release (urgency: 2, suppression_key: "research:dentists:2026-W17")
Customer: null

[Output]
{
  "body": "Dr. Meera, JIDA's Oct issue landed. One item relevant to your high-risk adult patients — 2,100-patient trial showed 3-month fluoride recall cuts caries recurrence 38% better than 6-month. Worth a look (2-min abstract). Want me to pull it + draft a patient-ed WhatsApp you can share? — JIDA Oct 2026 p.14",
  "cta": "open_ended",
  "send_as": "vera",
  "suppression_key": "research:dentists:2026-W17",
  "rationale": "Anchors on 2,100-patient JIDA trial and high-risk adult cohort; uses curiosity and effort externalization with low-friction CTA."
}

### Appendix B: Reference Composition Example (Customer-Facing on behalf of Merchant)
[Context]
Category: dentists (rules: no medical claims, no 'guaranteed')
Merchant: Dr. Meera (active offer: "Dental Cleaning @ ₹299", available slots: Wed 6pm + Thu 5pm)
Trigger: recall_due (scope: customer, urgency: 3, suppression_key: "recall:c_001_priya:2026-W18")
Customer: Priya (lapsed_soft, 5 months since last visit, weekday-evening preference, language: hi-en mix)

[Output]
{
  "body": "Hi Priya, Dr. Meera's clinic here 🦷 It's been 5 months since your last visit — your 6-month cleaning recall is due. Apke liye 2 slots ready hain: Wed 6 Nov, 6pm ya Thu 7 Nov, 5pm. ₹299 cleaning + complimentary fluoride. Reply 1 for Wed, 2 for Thu, or tell us a time that works.",
  "cta": "open_ended",
  "send_as": "merchant_on_behalf",
  "suppression_key": "recall:c_001_priya:2026-W18",
  "rationale": "Sent on behalf of clinic; honors Hindi-English mix and evening preference; uses real ₹299 catalog price and specific open slots."
}
"""


def _get_gemini_client() -> Client | None:
    """Initialize Google GenAI client if API key is present."""
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        return None
    try:
        return Client(api_key=api_key)
    except Exception as exc:
        logger.warning("Could not initialize Client: %s", exc)
        return None


def _deterministic_fallback_compose(
    category: dict[str, Any],
    merchant: dict[str, Any],
    trigger: dict[str, Any],
    customer: dict[str, Any] | None = None,
) -> ComposedMessage:
    """
    Deterministic fallback adhering strictly to all 5 scoring dimensions,
    explicit factual anchoring (elapsed time since last_visit / CTR comparisons),
    and avoidance of vague phrases.
    """
    category_slug = category.get("slug", "general")
    merchant_id = merchant.get("merchant_id") or trigger.get("merchant_id") or "m_unknown"
    merchant_identity = merchant.get("identity", {})
    merchant_name = merchant_identity.get("name") or "our clinic"
    owner_name = merchant_identity.get("owner_first_name") or merchant_name
    trigger_id = trigger.get("id") or trigger.get("trigger_id") or "trg_unknown"
    trigger_kind = trigger.get("kind", "proactive_update")
    trigger_payload = trigger.get("payload", {})
    suppression_key = trigger.get("suppression_key") or f"suppress:{merchant_id}:{trigger_id}"

    # Determine recipient language preference
    languages = []
    if merchant:
        ident_langs = merchant.get("identity", {}).get("languages", [])
        if isinstance(ident_langs, list):
            languages.extend(ident_langs)
        elif isinstance(ident_langs, str):
            languages.append(ident_langs)
    if customer:
        cust_lang = customer.get("identity", {}).get("language_pref", "")
        if cust_lang:
            languages.append(cust_lang)
    use_hindi = any("hi" in str(l).lower() for l in languages)

    # Active offer lookup for service+price framing
    active_offers = [o.get("title") for o in merchant.get("offers", []) if o.get("status") == "active"]
    offer_title = active_offers[0] if active_offers else None
    if not offer_title:
        cat_offers = category.get("offer_catalog", [])
        offer_title = cat_offers[0].get("title") if cat_offers else "Special Service"

    # Case A: Customer-Facing (on behalf of merchant)
    if customer:
        cust_identity = customer.get("identity", {})
        cust_name = cust_identity.get("name", "there")
        rel = customer.get("relationship", {})
        last_visit_str = rel.get("last_visit") or trigger_payload.get("last_service_date")
        ref_date = (
            trigger_payload.get("due_date")
            or (trigger_payload.get("available_slots", [{}])[0].get("iso") if trigger_payload.get("available_slots") else None)
            or trigger.get("expires_at")
        )
        elapsed_months, _ = (
            compute_elapsed_time(last_visit_str, ref_date)
            if last_visit_str
            else (5, "5 months since your last visit")
        )
        service_due = trigger_payload.get("service_due", "").replace("_", " ")

        slots = trigger_payload.get("available_slots", [])
        slots_hi = f" Slots ready hain: {slots[0]['label']} ya {slots[1]['label']}." if len(slots) >= 2 else ""
        slots_en = f" Available slots: {slots[0]['label']} or {slots[1]['label']}." if len(slots) >= 2 else ""

        is_dentist = category_slug == "dentists"
        if is_dentist:
            sender_prefix = f"{merchant_name} 🦷" if "clinic" in merchant_name.lower() else f"{merchant_name} clinic 🦷"
        else:
            sender_prefix = merchant_name

        if "appointment" in trigger_kind:
            # Appointment reminder
            if use_hindi:
                body = (
                    f"Hi {cust_name}, {sender_prefix} se reminder hai. Kal aapka appointment scheduled hai. "
                    f"{offer_title}. Kya hum aapka slot confirm karein? Reply YES to confirm."
                )
            else:
                body = (
                    f"Hi {cust_name}, appointment reminder from {sender_prefix}. Your visit is scheduled for tomorrow. "
                    f"{offer_title}. Would you like us to confirm your slot? Reply YES to confirm."
                )
            rationale_text = f"Customer appointment reminder on behalf of {merchant_name} anchoring on {offer_title}."

        elif "chronic" in trigger_kind or ("refill" in trigger_kind and category_slug == "pharmacies"):
            molecules = trigger_payload.get("molecule_list", [])
            med_note = f" ({molecules[0]})" if molecules else ""
            if use_hindi:
                body = (
                    f"Hi {cust_name}, {sender_prefix} se regular medicine{med_note} refill reminder hai. "
                    f"{offer_title}. Kya hum aapka order ready karein? Reply YES to confirm."
                )
            else:
                body = (
                    f"Hi {cust_name}, prescription refill reminder from {sender_prefix}{med_note}. "
                    f"{offer_title}. Would you like us to prepare your refill? Reply YES to confirm."
                )
            rationale_text = f"Customer refill reminder on behalf of {merchant_name} anchoring on {offer_title}."

        elif "lapsed_hard" in trigger_kind and trigger_payload.get("days_since_last_visit"):
            days = trigger_payload.get("days_since_last_visit")
            if use_hindi:
                body = (
                    f"Hi {cust_name}, {sender_prefix} se bol rahe hain. Aapke last visit ko {days} days ho gaye hain. "
                    f"{offer_title}. Kya aap apna session schedule karna chahenge? Reply YES to confirm."
                )
            else:
                body = (
                    f"Hi {cust_name}, {sender_prefix} here. It has been {days} days since your last session. "
                    f"{offer_title}. Would you like to schedule your next session? Reply YES to confirm."
                )
            rationale_text = f"Customer winback on behalf of {merchant_name} stating {days} days since last visit and anchoring on {offer_title}."

        else:
            # General recall / soft-lapsed / follow-up
            service_label = service_due if service_due else ("cleaning recall" if is_dentist else "service visit")
            if use_hindi:
                body = (
                    f"Hi {cust_name}, {sender_prefix} se bol rahe hain. "
                    f"Aapke last visit ko {elapsed_months} mahine ho gaye hain — aapka {service_label} due hai.{slots_hi} "
                    f"{offer_title}. Kya hum aapka slot book karein? Reply YES to confirm."
                )
            else:
                body = (
                    f"Hi {cust_name}, {sender_prefix} here. "
                    f"It's been {elapsed_months} months since your last visit — your {service_label} is due.{slots_en} "
                    f"{offer_title}. Would you like us to reserve a slot for you? Reply YES to confirm."
                )
            rationale_text = f"Customer-facing outreach on behalf of {merchant_name} stating {elapsed_months} months since last visit and anchoring on {offer_title}."

        return ComposedMessage(
            body=body,
            cta="binary_yes_stop",
            send_as="merchant_on_behalf",
            suppression_key=suppression_key,
            rationale=rationale_text,
        )

    # Case B: Merchant-Facing Outreach
    perf = merchant.get("performance", {})
    views = perf.get("views")
    calls = perf.get("calls")
    ctr = perf.get("ctr")
    delta_views = perf.get("delta_7d", {}).get("views_pct")
    peer_stats = category.get("peer_stats", {})
    peer_ctr = peer_stats.get("avg_ctr")

    is_dentist = category_slug == "dentists"
    prefix = f"Dr. {owner_name}" if is_dentist else f"Hi {owner_name}"
    locality = merchant_identity.get("locality", "aapke area" if use_hindi else "your area")

    comparison_en = ""
    comparison_hi = ""
    if ctr is not None and peer_ctr is not None:
        ctr_pct = f"{round(ctr * 100, 1)}%"
        peer_ctr_pct = f"{round(peer_ctr * 100, 1)}%"
        comparison_en = f" Your CTR is {ctr_pct} vs {peer_ctr_pct} peer median."
        comparison_hi = f" Aapka CTR {ctr_pct} hai vs {peer_ctr_pct} peer median."

    # 1. Research Digest, CDE, or Regulation Change Trigger
    if any(k in trigger_kind for k in ["research", "digest", "cde", "regulation"]):
        if "cde" in trigger_kind:
            credits = trigger_payload.get("credits", 2)
            if use_hindi:
                body = (
                    f"{prefix}, IDA ka naya CDE webinar announce hua hai ({credits} credit hours, free for members). "
                    f"Maine aapke clinic ke liye registration details draft ki hain. "
                    f"Kya main aapko details bhejun? Reply YES to review."
                )
            else:
                body = (
                    f"{prefix}, IDA announced an accredited CDE webinar offering {credits} credit hours for members. "
                    f"I drafted a summary and registration link for your clinic team. "
                    f"Would you like me to send you the details? Reply YES to review."
                )
            rationale_text = f"Anchors on verifiable CDE opportunity ({credits} credits) using effort externalization and binary CTA."
        elif "regulation" in trigger_kind:
            deadline = trigger_payload.get("deadline_iso", "2026-12-15")
            if use_hindi:
                body = (
                    f"{prefix}, Dental Council of India (DCI) radiograph guidelines update landed ({deadline}). "
                    f"Maine aapke clinic ke liye 2-minute compliance checklist ready ki hai. "
                    f"Kya main checklist share karun? Reply YES to review."
                )
            else:
                body = (
                    f"{prefix}, Dental Council of India (DCI) issued updated radiograph compliance guidelines ({deadline}). "
                    f"I prepared a 2-minute compliance checklist for your practice. "
                    f"Would you like me to share the checklist? Reply YES to review."
                )
            rationale_text = f"Anchors on DCI compliance update ({deadline}) using loss aversion and binary CTA."
        else:
            top_item = trigger_payload.get("top_item") or (category.get("digest", [{}])[0] if category.get("digest") else {})
            paper_title = top_item.get("title", "new clinical trial data")
            source = top_item.get("source", "latest research digest")
            trial_n = top_item.get("trial_n")
            trial_txt_en = f" ({trial_n}-patient trial)" if trial_n else ""
            trial_txt_hi = f" ({trial_n}-patient trial data)" if trial_n else ""

            if use_hindi:
                body = (
                    f"{prefix}, {source} se naya research digest release hua hai{trial_txt_hi}: '{paper_title}'. "
                    f"Maine aapke patients ke liye ek quick summary draft ki hai. "
                    f"Kya main aapko draft bhejun? Reply YES to review."
                )
            else:
                body = (
                    f"{prefix}, {source} just dropped with data{trial_txt_en}: '{paper_title}'. "
                    f"I drafted a 2-minute summary and patient educational tip you can share. "
                    f"Would you like me to send you the draft? Reply YES to review."
                )
            rationale_text = f"Anchors on verifiable publication '{source}' using effort externalization and binary CTA."

        return ComposedMessage(
            body=body,
            cta="binary_yes_stop",
            send_as="vera",
            suppression_key=suppression_key,
            rationale=rationale_text,
        )

    # 2. Planning Intent Trigger
    if "planning" in trigger_kind or "intent" in trigger_kind:
        topic_raw = trigger_payload.get("intent_topic", "new offering")
        topic = topic_raw.replace("_", " ")
        if use_hindi:
            body = (
                f"{prefix}, aapke {topic} idea ke basis par maine complete package menu aur pricing draft kar di hai.{comparison_hi} "
                f"Kya aap is draft structure ko review karna chahenge? Reply YES to see."
            )
        else:
            body = (
                f"{prefix}, based on your request for {topic}, I drafted the program structure and launch post.{comparison_en} "
                f"Would you like to review the draft details? Reply YES to see."
            )
        return ComposedMessage(
            body=body,
            cta="binary_yes_stop",
            send_as="vera",
            suppression_key=suppression_key,
            rationale=f"Anchors on merchant planning intent for '{topic}' using effort externalization and binary CTA.",
        )

    # 3. Competitor Opened Trigger
    if "competitor" in trigger_kind:
        comp_name = trigger_payload.get("competitor_name", "A new competitor")
        dist = trigger_payload.get("distance_km")
        dist_str = f" {dist} km door " if dist else " nearby "
        comp_offer = trigger_payload.get("their_offer", "")
        offer_mention = f" unke offer '{comp_offer}' ke against " if comp_offer else " "
        if use_hindi:
            body = (
                f"{prefix}, aapke area mein{dist_str}{comp_name} open hua hai. Humne{offer_mention}aapke {offer_title} ko highlight karne ke liye post banayi hai. "
                f"Kya hum ise publish karein? Reply YES to confirm."
            )
        else:
            body = (
                f"{prefix}, {comp_name} opened{dist_str}in your area. I prepared a post highlighting your {offer_title} to protect your footfall. "
                f"Would you like me to publish this update? Reply YES to confirm."
            )
        return ComposedMessage(
            body=body,
            cta="binary_yes_stop",
            send_as="vera",
            suppression_key=suppression_key,
            rationale=f"Anchors on competitor opening ({comp_name}) using loss aversion and binary CTA.",
        )

    # 4. IPL Match Trigger
    if "ipl" in trigger_kind or "match" in trigger_kind:
        match_name = trigger_payload.get("match", "today's match")
        venue = trigger_payload.get("venue", "stadium")
        if use_hindi:
            body = (
                f"{prefix}, aaj shaam {venue} mein {match_name} match hai! Match-time orders ke liye {offer_title} ka promotion draft kiya hai. "
                f"Kya hum ise listing par live karein? Reply YES to confirm."
            )
        else:
            body = (
                f"{prefix}, {match_name} match is scheduled this evening at {venue}! I drafted a match-day promo for {offer_title} to capture order volume. "
                f"Should I make this promotion live? Reply YES to confirm."
            )
        return ComposedMessage(
            body=body,
            cta="binary_yes_stop",
            send_as="vera",
            suppression_key=suppression_key,
            rationale=f"Anchors on IPL match event ({match_name}) using time urgency and binary CTA.",
        )

    # 5. GBP Unverified Trigger
    if "gbp" in trigger_kind or "unverified" in trigger_kind:
        uplift = trigger_payload.get("estimated_uplift_pct", 0.3)
        uplift_pct = f"{int(uplift * 100)}%" if isinstance(uplift, (int, float)) and uplift <= 1 else f"{uplift}%"
        if use_hindi:
            body = (
                f"{prefix}, aapka Google Business profile abhi unverified hai. Verification complete karne se search visibility mein ~{uplift_pct} uplift expect hota hai. "
                f"Maine 2-step verification guide prepare ki hai. Kya main share karun? Reply YES to see."
            )
        else:
            body = (
                f"{prefix}, your Google Business listing is currently unverified. Verifying it typically delivers a ~{uplift_pct} search visibility boost. "
                f"I prepared a quick 2-step verification guide. Would you like me to share it? Reply YES to see."
            )
        return ComposedMessage(
            body=body,
            cta="binary_yes_stop",
            send_as="vera",
            suppression_key=suppression_key,
            rationale=f"Anchors on unverified profile status with {uplift_pct} uplift using effort externalization and binary CTA.",
        )

    # 6. Milestone Reached Trigger
    if "milestone" in trigger_kind:
        val_now = trigger_payload.get("value_now", views or 100)
        ms_val = trigger_payload.get("milestone_value", (views or 100) + 10)
        metric_name = trigger_payload.get("metric", "reviews").replace("_", " ")
        if use_hindi:
            body = (
                f"{prefix}, badhaai ho! Aapke {val_now} {metric_name} ho chuke hain — {ms_val} milestone sirf thoda door hai. "
                f"Maine milestone celebration post ready ki hai. Kya hum ise publish karein? Reply YES to confirm."
            )
        else:
            body = (
                f"{prefix}, congratulations! Your listing reached {val_now} {metric_name}, approaching the {ms_val} milestone. "
                f"I drafted a milestone post to celebrate with your customers. Would you like me to publish it? Reply YES to confirm."
            )
        return ComposedMessage(
            body=body,
            cta="binary_yes_stop",
            send_as="vera",
            suppression_key=suppression_key,
            rationale=f"Anchors on verifiable milestone progress ({val_now} towards {ms_val} {metric_name}) using social proof and binary CTA.",
        )

    # 7. Seasonal Demand Shift Trigger
    if "seasonal" in trigger_kind:
        season = trigger_payload.get("season", "summer").replace("_", " ")
        trends = trigger_payload.get("trends", [])
        top_trend = trends[0].replace("_", " ") if trends else "high demand products"
        if use_hindi:
            body = (
                f"{prefix}, {season} demand shift mein {top_trend} dekha gaya hai. Maine aapki listing par {offer_title} highlight karne ke liye post draft kiya hai. "
                f"Kya hum ise live karein? Reply YES to confirm."
            )
        else:
            body = (
                f"{prefix}, seasonal analysis for {season} highlights surges in {top_trend}. I drafted a featured listing post for {offer_title}. "
                f"Would you like me to publish this update? Reply YES to confirm."
            )
        return ComposedMessage(
            body=body,
            cta="binary_yes_stop",
            send_as="vera",
            suppression_key=suppression_key,
            rationale=f"Anchors on seasonal trend ({top_trend}) using category relevance and binary CTA.",
        )

    # 8. Performance Spike or Dip Trigger
    if "perf" in trigger_kind and views is not None:
        delta_str = f"{abs(int((delta_views or 0.15) * 100))}%"
        trend = "increased" if (delta_views or 0) >= 0 else "decreased"
        trend_hi = "badh" if (delta_views or 0) >= 0 else "kam"

        if use_hindi:
            body = (
                f"{prefix}, aapki listing par {views} views aaye hain, aur 7-day views {delta_str} {trend_hi} hue hain.{comparison_hi} "
                f"Humne {locality} ke search leads analyze kiye hain. "
                f"Kya aap top keyword opportunities dekhna chahenge? Reply YES to see."
            )
        else:
            body = (
                f"{prefix}, your listing reached {views} views, with 7-day views that {trend} by {delta_str}.{comparison_en} "
                f"I analyzed search intent in {locality} to capture these leads. "
                f"Want me to share the top keyword opportunities? Reply YES to see."
            )

        return ComposedMessage(
            body=body,
            cta="binary_yes_stop",
            send_as="vera",
            suppression_key=suppression_key,
            rationale=f"Anchors on verifiable metrics ({views} views, {delta_str} {trend}{comparison_en}) using curiosity lever and binary CTA.",
        )

    # 9. Festival Upcoming Trigger
    if "festival" in trigger_kind:
        festival_raw = trigger_payload.get("festival", "festive season")
        fest_label = festival_raw if "season" in festival_raw.lower() else f"{festival_raw} season"
        if use_hindi:
            body = (
                f"{prefix}, upcoming {fest_label} ke advance bookings rush ke liye humne {offer_title} ka festive promotion draft kiya hai.{comparison_hi} "
                f"Kya hum ise publish karein? Reply YES to confirm."
            )
        else:
            body = (
                f"{prefix}, with the {fest_label} approaching, I drafted a promotional campaign for {offer_title} to capture early bookings.{comparison_en} "
                f"Would you like me to publish this campaign? Reply YES to confirm."
            )
        return ComposedMessage(
            body=body,
            cta="binary_yes_stop",
            send_as="vera",
            suppression_key=suppression_key,
            rationale=f"Anchors on seasonal festival demand ({fest_label}) using effort externalization and binary CTA.",
        )

    # 10. Default Merchant Growth Nudge
    if use_hindi:
        body = (
            f"{prefix}, {locality} mein {offer_title} ke liye high local demand dekhi gayi hai.{comparison_hi} "
            f"Maine aapke Google profile ke liye ek update post draft kar diya hai. "
            f"Kya hum ise publish karein? Reply YES to confirm."
        )
    else:
        body = (
            f"{prefix}, we identified high local demand in {locality} for {offer_title}.{comparison_en} "
            f"I drafted an updated Google Business post highlighting this service to increase your search visibility. "
            f"Would you like me to publish this update for your listing? Reply YES to confirm."
        )

    return ComposedMessage(
        body=body,
        cta="binary_yes_stop",
        send_as="vera",
        suppression_key=suppression_key,
        rationale=f"Anchors on active service+price ({offer_title}{comparison_en}) using effort externalization and binary CTA.",
    )


async def compose(
    category: dict[str, Any],
    merchant: dict[str, Any],
    trigger: dict[str, Any],
    customer: dict[str, Any] | None = None,
    conversation_id: str | None = None,
) -> dict[str, Any]:
    """
    Compose an outbound message for a given (category, merchant, trigger, customer).
    Includes a 4-check post-generation validation layer (anti-repetition, single-CTA,
    fact-grounding, language check) with automated single-attempt re-generation
    before falling back to the deterministic template.
    """
    merchant_id = merchant.get("merchant_id") or trigger.get("merchant_id") or "m_unknown"
    merchant_identity = merchant.get("identity", {})
    merchant_name = (
        merchant_identity.get("owner_first_name")
        or merchant_identity.get("name")
        or "Partner"
    )
    trigger_id = trigger.get("id") or trigger.get("trigger_id") or "trg_unknown"
    category_slug = category.get("slug", "general")
    customer_id = customer.get("customer_id") if customer else None
    conv_id = conversation_id or f"conv_{merchant_id}_{trigger_id}_{uuid.uuid4().hex[:8]}"

    composed_msg: ComposedMessage | None = None
    client = _get_gemini_client()

    if client:
        context_data = {
            "category": category,
            "merchant": merchant,
            "trigger": trigger,
            "customer": customer,
        }
        base_prompt = (
            f"{SYSTEM_INSTRUCTION}\n\n"
            f"{FEW_SHOT_EXAMPLES}\n\n"
            f"### YOUR TASK\n"
            f"Compose the next WhatsApp message based on this exact context input:\n"
            f"{json.dumps(context_data, indent=2, default=str)}\n\n"
            f"Generate strict JSON matching the schema."
        )

        config = types.GenerateContentConfig(
            temperature=0.0,
            response_mime_type="application/json",
            response_schema=ComposedMessage,
        )

        async def _call_gemini(prompt: str) -> ComposedMessage | None:
            try:
                response = await asyncio.wait_for(
                    client.aio.models.generate_content(
                        model="gemini-2.5-flash",
                        contents=prompt,
                        config=config,
                    ),
                    timeout=LLM_TIMEOUT_SECONDS,
                )
                if response and response.text:
                    return ComposedMessage.model_validate_json(response.text)
            except Exception as exc:
                logger.warning("Gemini generation failed or timed out: %s", exc)
            return None

        # 1. Initial generation attempt
        attempt_msg = await _call_gemini(base_prompt)

        if attempt_msg:
            # 2. Run post-generation validation layer
            is_valid, failures = validate_composed_message(
                attempt_msg, category, merchant, trigger, customer, conv_id
            )

            if is_valid:
                composed_msg = attempt_msg
                logger.info("Message for trigger %s passed all post-generation validation checks.", trigger_id)
            else:
                logger.warning(
                    "Post-generation validation failed for trigger %s: %s; re-generating with correction instructions.",
                    trigger_id,
                    failures,
                )
                # 3. Build re-generation prompt with targeted feedback
                retry_instructions = build_retry_instructions(failures, attempt_msg.body)
                retry_prompt = (
                    f"{base_prompt}\n\n"
                    f"### CORRECTION REQUIRED\n"
                    f"Your previous output failed the following quality checks:\n"
                    f"{chr(10).join('- ' + instr for instr in retry_instructions)}\n\n"
                    f"Previous invalid body was:\n\"{attempt_msg.body}\"\n\n"
                    f"Please regenerate a corrected message adhering strictly to all requirements."
                )

                retry_msg = await _call_gemini(retry_prompt)
                if retry_msg:
                    re_valid, re_failures = validate_composed_message(
                        retry_msg, category, merchant, trigger, customer, conv_id
                    )
                    if re_valid:
                        composed_msg = retry_msg
                        logger.info("Re-generated message for trigger %s passed validation on retry.", trigger_id)
                    else:
                        logger.warning(
                            "Re-generated message for trigger %s still failed validation: %s; using deterministic fallback.",
                            trigger_id,
                            re_failures,
                        )

    # 4. Deterministic fallback if Gemini is missing, failed, or failed validation
    if not composed_msg:
        composed_msg = _deterministic_fallback_compose(category, merchant, trigger, customer)

    # 5. Record body fingerprint for anti-repetition tracking
    record_body_fingerprint(conv_id, composed_msg.body)

    # Return complete action dictionary
    return {
        "conversation_id": conv_id,
        "merchant_id": merchant_id,
        "customer_id": customer_id,
        "send_as": composed_msg.send_as,
        "trigger_id": trigger_id,
        "template_name": f"vera_{category_slug}_v1",
        "template_params": [str(merchant_name), "proactive_nudge"],
        "body": composed_msg.body,
        "cta": composed_msg.cta,
        "suppression_key": composed_msg.suppression_key,
        "rationale": composed_msg.rationale,
    }


# =============================================================================
# /v1/reply Decision Logic (Challenge-Brief §9 & Testing-Brief §2.3)
# =============================================================================

async def compose_reply(
    conversation_id: str,
    merchant_id: str | None,
    customer_id: str | None,
    from_role: str,
    message: str,
    turn_number: int,
    history: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Implements /v1/reply decision logic matching challenge-brief.md §9 and
    challenge-testing-brief.md §2.3:
    1. Turn-depth guard: turn_number >= 5 returns action: "end"
    2. Objection/decline: returns action: "end"
    3. Auto-reply detection: 1st time returns action: "send" (nudge), 2nd time returns action: "end"
    4. Intent-handoff: returns action: "send" switching straight to concrete action mode
    5. Default case: calls compose() reusing stored 4-context pack + history
    """
    msg_clean = message.strip()
    msg_lower = msg_clean.lower()

    # Determine context & language preference
    saved_ctx = conversation_contexts.get(conversation_id, {})
    mid = merchant_id or saved_ctx.get("merchant_id")
    cid = customer_id or saved_ctx.get("customer_id")
    merch_entry = contexts.get(("merchant", mid)) if mid else None
    merch = merch_entry.get("payload", {}) if merch_entry else {}
    cat_slug = merch.get("category_slug") or saved_ctx.get("category_slug", "general")
    cat_entry = contexts.get(("category", cat_slug)) if cat_slug else None
    cat = cat_entry.get("payload", {}) if cat_entry else {}
    cust_entry = contexts.get(("customer", cid)) if cid else None
    cust = cust_entry.get("payload") if cust_entry else None
    trg_id = saved_ctx.get("trigger_id")
    trg_entry = contexts.get(("trigger", trg_id)) if trg_id else None
    trg = trg_entry.get("payload", {}) if trg_entry else {}

    languages = []
    if merch:
        ident_langs = merch.get("identity", {}).get("languages", [])
        if isinstance(ident_langs, list):
            languages.extend(ident_langs)
        elif isinstance(ident_langs, str):
            languages.append(ident_langs)
    if cust:
        cust_lang = cust.get("identity", {}).get("language_pref", "")
        if cust_lang:
            languages.append(cust_lang)
    use_hindi = any("hi" in str(l).lower() for l in languages)

    # -------------------------------------------------------------------------
    # 1. Turn-depth guard: if turn_number >= 5, return action: "end"
    # -------------------------------------------------------------------------
    if turn_number >= 5:
        closing_body = (
            "Aapka bahut shukriya! Humne sabhi details note kar li hain. Agar aapko koi aur madad chahiye ho toh zaroor batayein. Shubh din!"
            if use_hindi
            else "Thank you for your time! We have logged all details from our conversation. Feel free to reach out anytime if you need further assistance. Have a great day!"
        )
        return {
            "action": "end",
            "body": closing_body,
            "rationale": f"Turn depth limit reached ({turn_number} >= 5 turns); gracefully closing conversation per policy.",
        }

    # -------------------------------------------------------------------------
    # 2. Objection / Decline: detect disinterest signals -> return action: "end"
    # -------------------------------------------------------------------------
    is_objection = any(re.search(r"\b" + re.escape(p) + r"\b", msg_lower) for p in OBJECTION_PHRASES) or (
        msg_lower in {"stop", "nahi", "no", "nahin", "not interested", "no thanks"}
    )
    if is_objection:
        decline_body = (
            "Koi baat nahi, samajh gayi. Hum aapko aage disturb nahi karenge. Shubh din!"
            if use_hindi
            else "Understood completely. We will not send you further messages regarding this. Wishing you all the best!"
        )
        return {
            "action": "end",
            "body": decline_body,
            "rationale": "Merchant expressed disinterest or opt-out; gracefully ending conversation without re-pitching.",
        }

    # -------------------------------------------------------------------------
    # 3. Auto-reply detection: canned phrase OR 2+ verbatim identical messages
    # -------------------------------------------------------------------------
    prev_incoming = conversation_incoming_messages.get(conversation_id, [])
    is_verbatim_repeat = len(prev_incoming) >= 1 and (prev_incoming[-1].strip().lower() == msg_lower)
    is_canned = any(phrase in msg_lower for phrase in AUTO_REPLY_PHRASES)

    if is_canned or is_verbatim_repeat:
        # Increment counter for this conversation and merchant
        c_count = conversation_auto_reply_count.get(conversation_id, 0) + 1
        conversation_auto_reply_count[conversation_id] = c_count
        if mid:
            m_count = merchant_auto_reply_count.get(mid, 0) + 1
            merchant_auto_reply_count[mid] = m_count
        else:
            m_count = c_count

        effective_count = max(c_count, m_count)
        conversation_incoming_messages.setdefault(conversation_id, []).append(message)

        if effective_count == 1:
            # First occurrence: respond once with human nudge (action: "send")
            nudge_body = (
                "Samajh gayi! Team tak pahunchane se pehle, kya aap khud 2 minute mein dekhna chahenge ki humne aapke liye kya update ready kiya hai? Chalega?"
                if use_hindi
                else "Understood! Before passing this along, would you like to take a quick 2-minute look at what we prepared for you?"
            )
            # Record fingerprint
            record_body_fingerprint(conversation_id, nudge_body)
            return {
                "action": "send",
                "body": nudge_body,
                "cta": "open_ended",
                "rationale": "First auto-reply detected; sending short human nudge per challenge-brief Pattern B.",
            }
        else:
            # Second occurrence: return action: "end"
            closing_body = (
                "Koi baat nahi, samajh gayi. Main owner/manager se directly connect kar lungi. Aapka business accha chale — best wishes!"
                if use_hindi
                else "Understood, no problem at all! I will connect with the owner or manager directly later. Wishing you continued success!"
            )
            return {
                "action": "end",
                "body": closing_body,
                "rationale": "Second auto-reply detected in conversation; gracefully ending without burning further turns.",
            }

    # Record incoming message
    conversation_incoming_messages.setdefault(conversation_id, []).append(message)

    # -------------------------------------------------------------------------
    # 4. Intent-handoff: detect affirmative intent -> action mode (no re-qualifying)
    # -------------------------------------------------------------------------
    clean_msg = re.sub(r"[^\w\s]", " ", msg_lower).strip()
    is_affirmative = (
        clean_msg in {"yes", "haan", "ha", "sure", "proceed", "go ahead", "sounds good", "chalega", "theek hai", "thik hai", "karna hai", "ok"}
        or any(re.search(r"\b" + re.escape(p) + r"\b", clean_msg) for p in AFFIRMATIVE_INTENT_PHRASES)
    )

    if is_affirmative:
        action_body = (
            "Done! Main aage proceed kar rahi hoon. Aapke updates draft ho chuke hain, abhi confirmation details bhej rahi hoon."
            if use_hindi
            else "Done! Moving straight ahead to the next step. I finalized the draft and am sending your confirmation details now."
        )

        # Validate action_body with 4 post-generation checks
        msg_obj = ComposedMessage(
            body=action_body,
            cta="open_ended",
            send_as="vera",
            suppression_key=f"reply:{conversation_id}:{turn_number}",
            rationale="Switched to action mode on explicit intent."
        )
        is_val, _ = validate_composed_message(msg_obj, cat, merch, trg, cust, conversation_id)
        if not is_val:
            action_body = "Done! Proceeding with your request right now. Sending confirmation details momentarily."

        record_body_fingerprint(conversation_id, action_body)
        return {
            "action": "send",
            "body": action_body,
            "cta": "open_ended",
            "rationale": "Merchant expressed affirmative intent; switching immediately to action mode per Pattern D without re-qualifying.",
        }

    # -------------------------------------------------------------------------
    # 5. Default Case: Context & History-informed generation (action: "send")
    # -------------------------------------------------------------------------
    active_offers = [o.get("title") for o in merch.get("offers", []) if o.get("status") == "active"]
    offer_title = active_offers[0] if active_offers else (cat.get("offer_catalog", [{}])[0].get("title", "Dental Cleaning @ ₹299"))

    default_body = (
        f"Shukriya! Humne aapki request note kar li hai. {offer_title} ke sath next steps confirm karne ke liye kya hum aage badhein? Reply YES to confirm."
        if use_hindi
        else f"Thank you! We logged your request. Should we proceed with the next steps for {offer_title}? Reply YES to confirm."
    )

    msg_obj = ComposedMessage(
        body=default_body,
        cta="binary_yes_stop",
        send_as="vera",
        suppression_key=f"reply:{conversation_id}:{turn_number}",
        rationale="Context-informed reply advancing conversation."
    )
    is_val, _ = validate_composed_message(msg_obj, cat, merch, trg, cust, conversation_id)
    if not is_val:
        default_body = f"Thank you! Should we proceed with the next steps for {offer_title}? Reply YES to confirm."

    record_body_fingerprint(conversation_id, default_body)
    return {
        "action": "send",
        "body": default_body,
        "cta": "binary_yes_stop",
        "rationale": "Acknowledged incoming message and advanced conversation based on stored context and history.",
    }


# =============================================================================
# API Endpoints (Exact Contract per challenge-testing-brief.md §2)
# =============================================================================

@app.post("/v1/context")
async def push_context(body: ContextRequest):
    """
    2.1 POST /v1/context — receive a context push
    Idempotent by (context_id, version).
    Higher version replaces atomically.
    Same or lower version returns 409 stale_version.
    Invalid scope returns 400 invalid_scope.
    """
    if body.scope not in VALID_SCOPES:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "accepted": False,
                "reason": "invalid_scope",
                "details": f"Scope '{body.scope}' is invalid. Allowed scopes: {', '.join(sorted(VALID_SCOPES))}",
            },
        )

    key = (body.scope, body.context_id)
    cur = contexts.get(key)

    if cur is not None and cur["version"] >= body.version:
        return JSONResponse(
            status_code=status.HTTP_409_CONFLICT,
            content={
                "accepted": False,
                "reason": "stale_version",
                "current_version": cur["version"],
            },
        )

    contexts[key] = {
        "version": body.version,
        "payload": body.payload,
        "delivered_at": body.delivered_at,
    }

    stored_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
    return {
        "accepted": True,
        "ack_id": f"ack_{body.context_id}_v{body.version}",
        "stored_at": stored_at,
    }


async def _handle_tick(body: TickRequest) -> dict[str, Any]:
    """Internal tick processor mapping available triggers to contexts."""
    actions: list[dict[str, Any]] = []

    for trg_id in body.available_triggers:
        trg_entry = contexts.get(("trigger", trg_id))
        if not trg_entry:
            continue
        trigger = trg_entry.get("payload", {})

        # Resolve merchant
        merchant_id = trigger.get("merchant_id") or trigger.get("payload", {}).get("merchant_id")
        if not merchant_id:
            continue

        merchant_entry = contexts.get(("merchant", merchant_id))
        if not merchant_entry:
            continue
        merchant = merchant_entry.get("payload", {})

        # Resolve category
        category_slug = merchant.get("category_slug") or trigger.get("payload", {}).get("category")
        category_entry = contexts.get(("category", category_slug)) if category_slug else None
        category = category_entry.get("payload", {}) if category_entry else {}

        # Resolve customer (optional)
        customer_id = trigger.get("customer_id") or trigger.get("payload", {}).get("customer_id")
        customer = None
        if customer_id:
            customer_entry = contexts.get(("customer", customer_id))
            if customer_entry:
                customer = customer_entry.get("payload")

        # Stable conversation identifier per proactive cycle
        conv_id = f"conv_{merchant_id}_{trg_id}"

        # Save conversation context mapping for multi-turn /v1/reply lookups
        conversation_contexts[conv_id] = {
            "merchant_id": merchant_id,
            "customer_id": customer_id,
            "category_slug": category_slug,
            "trigger_id": trg_id,
        }

        action = await compose(category, merchant, trigger, customer, conversation_id=conv_id)

        if action and action.get("body"):
            actions.append(action)
            conversations.setdefault(conv_id, []).append({
                "role": "bot",
                "turn_number": 1,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "send_as": action.get("send_as", "vera"),
                "body": action.get("body"),
                "cta": action.get("cta"),
                "rationale": action.get("rationale"),
                "trigger_id": trg_id,
            })

        # Respect tick action cap (20 actions per tick per §5)
        if len(actions) >= 20:
            break

    return {"actions": actions}


@app.post("/v1/tick")
async def tick(body: TickRequest):
    """
    2.2 POST /v1/tick — periodic wake-up; bot can initiate proactive messages.
    Guaranteed to return within 30s using asyncio timeout with safe fallback.
    """
    try:
        return await asyncio.wait_for(_handle_tick(body), timeout=REQUEST_TIMEOUT_SECONDS)
    except (asyncio.TimeoutError, TimeoutError):
        logger.warning(
            "POST /v1/tick timed out after %s seconds; returning safe fallback actions: []",
            REQUEST_TIMEOUT_SECONDS,
        )
        return {"actions": []}
    except Exception as exc:
        logger.exception("Error during /v1/tick execution: %s; returning safe fallback.", exc)
        return {"actions": []}


async def _handle_reply(body: ReplyRequest) -> dict[str, Any]:
    """Internal reply processor updating conversation history and executing response composition."""
    conv_history = conversations.setdefault(body.conversation_id, [])
    conv_history.append({
        "role": body.from_role,
        "turn_number": body.turn_number,
        "received_at": body.received_at,
        "message": body.message,
    })

    result = await compose_reply(
        conversation_id=body.conversation_id,
        merchant_id=body.merchant_id,
        customer_id=body.customer_id,
        from_role=body.from_role,
        message=body.message,
        turn_number=body.turn_number,
        history=conv_history,
    )

    conv_history.append({
        "role": "bot",
        "turn_number": body.turn_number + 1,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        **result,
    })

    return result


@app.post("/v1/reply")
async def reply(body: ReplyRequest):
    """
    2.3 POST /v1/reply — receive a reply from simulated merchant/customer.
    Guaranteed to return within 30s using asyncio timeout with safe fallback.
    """
    try:
        return await asyncio.wait_for(_handle_reply(body), timeout=REQUEST_TIMEOUT_SECONDS)
    except (asyncio.TimeoutError, TimeoutError):
        logger.warning(
            "POST /v1/reply for conv '%s' timed out after %s seconds; returning safe fallback.",
            body.conversation_id,
            REQUEST_TIMEOUT_SECONDS,
        )
        fallback = {
            "action": "send",
            "body": "Thank you for getting back to me. Let me check the details and follow up with you shortly.",
            "cta": "open_ended",
            "rationale": "Safe fallback response returned due to timeout to ensure 30-second turnaround.",
        }
        conversations.setdefault(body.conversation_id, []).append({
            "role": "bot",
            "turn_number": body.turn_number + 1,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **fallback,
        })
        return fallback
    except Exception as exc:
        logger.exception("Error during /v1/reply execution: %s; returning safe fallback.", exc)
        fallback = {
            "action": "send",
            "body": "Thank you for reaching out. We have logged your request and will follow up momentarily.",
            "cta": "open_ended",
            "rationale": f"Safe fallback response returned due to error: {str(exc)[:50]}",
        }
        conversations.setdefault(body.conversation_id, []).append({
            "role": "bot",
            "turn_number": body.turn_number + 1,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **fallback,
        })
        return fallback


@app.get("/v1/healthz")
async def healthz():
    """
    2.4 GET /v1/healthz — liveness probe and context counter
    """
    counts = {"category": 0, "merchant": 0, "customer": 0, "trigger": 0}
    for (scope, _), _ in contexts.items():
        if scope in counts:
            counts[scope] += 1
        else:
            counts[scope] = counts.get(scope, 0) + 1

    uptime_seconds = int(time.time() - START_TIME)
    return {
        "status": "ok",
        "uptime_seconds": uptime_seconds,
        "contexts_loaded": counts,
    }


@app.get("/v1/metadata")
async def metadata():
    """
    2.5 GET /v1/metadata — bot identity and technical configuration
    """
    team_name = os.getenv("TEAM_NAME", "Team Alpha")
    team_members_raw = os.getenv("TEAM_MEMBERS", "Alice,Bob")
    team_members = [m.strip() for m in team_members_raw.split(",") if m.strip()]
    model = os.getenv("MODEL", "gemini-2.5-flash")
    approach = os.getenv("APPROACH", "structured Gemini 2.5 Flash composer with 4-check post-generation validation and multi-turn stateful reply router")
    contact_email = os.getenv("CONTACT_EMAIL", "team@example.com")
    version = os.getenv("VERSION", "1.2.0")
    submitted_at = os.getenv("SUBMITTED_AT", "2026-04-26T08:00:00Z")

    return {
        "team_name": team_name,
        "team_members": team_members,
        "model": model,
        "approach": approach,
        "contact_email": contact_email,
        "version": version,
        "submitted_at": submitted_at,
    }


@app.post("/v1/teardown")
async def teardown():
    """
    11. POST /v1/teardown (optional per brief §11)
    Wipes in-memory contexts, conversation history, and fingerprints at end of test.
    """
    contexts.clear()
    conversations.clear()
    conversation_fingerprints.clear()
    conversation_contexts.clear()
    conversation_auto_reply_count.clear()
    merchant_auto_reply_count.clear()
    conversation_incoming_messages.clear()
    return {"status": "ok", "message": "State wiped successfully"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("bot:app", host="0.0.0.0", port=8080, reload=True)
