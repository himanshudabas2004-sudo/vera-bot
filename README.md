# Vera — magicpin AI Engagement Assistant

Vera is an autonomous AI merchant engagement engine on WhatsApp for magicpin. She identifies timely merchant triggers, anchors outreach on verifiable local facts and catalog offers, and manages multi-turn WhatsApp conversations through structured intent-routing.

---

## 1. Approach

- **Runtime & Framework**: Async Python service built on **FastAPI** and **Uvicorn**, designed to guarantee responsive responses (<30s SLA) with `asyncio.wait_for` timeouts and safe fallbacks.
- **LLM Backbone**: Google **Gemini 2.5 Flash** called via the `google-genai` SDK for low-latency reasoning.
- **Deterministic Structured Output**: Evaluated at `temperature=0` with `response_mime_type="application/json"` and Pydantic `response_schema` enforcing the `ComposedMessage` schema (`body`, `cta`, `send_as`, `suppression_key`, `rationale`).
- **Standardized Contract**: Full implementation of all 5 endpoints (`/v1/context`, `/v1/tick`, `/v1/reply`, `/v1/healthz`, `/v1/metadata`, plus `/v1/teardown`).

---

## 2. Architecture & Pipeline

### 4-Context Pack Resolution
Incoming events dynamically resolve a 4-tuple context pack:
1. **Category**: Vertical tone, taboo terminology, allowed clinical vocab, and benchmark stats.
2. **Merchant**: Identity, location, active offers (`service @ price`), and performance metrics.
3. **Trigger**: Specific event signal (why now), urgency, and suppression keys.
4. **Customer** *(optional)*: Visit recency (`last_visit`), relationship depth, and language preference.

Contexts are maintained with strict version idempotency (higher version replaces atomically, same/lower returns HTTP 409 `stale_version`).

### 4-Rule Post-Generation Validation Layer
Every generated message passes through a post-generation validation gate:
1. **Anti-Repetition**: Tracks normalized SHA-256 body fingerprints per conversation to prevent verbatim or near-verbatim re-pitching.
2. **Single-CTA Rule**: Enforces exactly one low-friction commitment ask positioned in the message's final sentence.
3. **Fact-Grounding Check**: Verifies that every number, price, metric, or date in the body is strictly grounded in the context payload—eliminating hallucinations.
4. **Language Matching**: Honors language preference (e.g. natural Romanized Hindi-English mix `Hinglish`).

*Resilience Loop*: If validation fails, the bot executes **one targeted re-generation attempt** incorporating the specific validator error into the prompt before falling back to a category-aligned deterministic template.

---

## 3. Multi-Turn Conversation Handling (`/v1/reply`)

Implemented per `challenge-brief.md` §9 and `challenge-testing-brief.md` §2.3:
- **Auto-Reply Detection (Pattern B)**: Matches WhatsApp Business canned auto-replies (*"team will respond shortly"*, *"automated assistant"*, etc.) and detects **2+ verbatim-identical** incoming messages. 
  - *1st Detection*: Nudges for human engagement (`action: "send"`).
  - *2nd Detection*: Clean exit (`action: "end"`), preventing infinite bot-to-bot loops.
- **Intent-Handoff (Pattern D)**: Recognizes explicit affirmative signals (*"yes"*, *"let's do it"*, *"haan"*, *"chalega"*), instantly switching to execution mode (*"Done! Moving straight ahead..."*) without re-qualifying questions.
- **Objection / Opt-Out**: Flags disinterest (*"stop"*, *"not interested"*, *"nahi chahiye"*), immediately closing the conversation (`action: "end"`) without re-pitching.
- **Turn-Depth Guard**: Enforces a strict guard terminating conversations at `turn_number >= 5` (`action: "end"`).

---

## 4. Engineering Tradeoffs

| Decision | Tradeoff Chosen | Rationale |
|---|---|---|
| **Single Re-Prompt (Not Unlimited)** | Max 1 Gemini retry, then fallback | Prevents cascading retry latency and guarantees SLA (<30s). |
| **In-Memory Store** | Memory dicts vs. external database | Zero operational overhead for testing; at production scale, Redis/Postgres would manage distributed state. |
| **Temperature = 0** | Determinism over stylistic variance | Prioritizes fact-grounding, policy compliance, and schema stability over generative novelty. |
| **Strict Substring Fact Grounding** | Rejects ungrounded numeric hallucination | Errs on the side of caution; a safe fallback is superior to hallucinating unverified discount figures. |

---

## 5. What Additional Context Would Have Helped Most

1. **Merchant Active Engagement Windows**: Knowing each owner's historical peak WhatsApp response hours (e.g., post-lunch 3–5 PM for restaurants vs. morning pre-OPD 8–9 AM for doctors) would allow scheduling ticks when open rates are highest.
2. **Extended Conversation Archives**: Having prior multi-week message histories rather than isolated turn windows would allow Vera to reference historical merchant preferences, accepted offers, and past objections.
3. **Peer Benchmark Distributions**: Full quartile or percentile distributions (P25, P50, P75) rather than only a single peer median would help craft more persuasive peer comparison framing (e.g., *"top 10% in your pin code"*).
