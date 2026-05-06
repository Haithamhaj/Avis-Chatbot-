# Avis Saudi Structured KB Chatbot Demo

## Epic name
Avis Saudi Structured KB Chatbot Demo

## Goal
Build a controlled conversational intelligence demo for Avis Saudi. The customer should experience a natural assistant named Khalid, while operational truth remains deterministic: structured KB lookup, calculators, workflow state, rule validation, escalation, payment/booking timing, and response guard.

## Architecture summary
The app is Python-first and split into reusable core logic, service adapters, and UI/API shells.

- `avis_ai_demo/core/`: Streamlit-free business and conversation core.
- `avis_ai_demo/data/`: structured KB files and non-operational metadata.
- `avis_ai_demo/services/`: OpenAI wrapper and mock external service adapters.
- `avis_ai_demo/adapters/`: UI adapter helpers.
- `app_streamlit.py`: Streamlit chat demo and internal trace view.
- `avis_ai_demo/api.py`: FastAPI HTTP backend using the same `handle_message` core as Streamlit.
- `avis_ai_demo/tests/`: contract, workflow, guard, routing, semantic search, API, and UI tests.

The main runtime path is:

```text
User message
-> language detection
-> short conversation history context
-> conversation management
-> safety/high-risk overrides
-> operational routing
-> GPT extraction with deterministic fallback
-> structured KB lookup
-> calculators/rules/workflow/mock services
-> Khalid persona response composer
-> response guard
-> customer response + internal trace
```

## Control boundaries
- GPT may classify intent, extract entities, detect ambiguity, generate clarifying questions, and phrase Khalid's response.
- GPT must not calculate totals, invent prices, invent availability, decide payment/booking status, invent branch data, invent policy rules, or override workflow/escalation decisions.
- Python deterministic modules own lookup, calculations, workflow phase transitions, pricing, validation, escalation, external-action timing, and response guarding.
- Response guard validates every final answer and falls back to deterministic safe templates if needed.

## Implementation sequence completed
1. Scaffolded Python package, pytest harness, and Streamlit shell.
2. Encoded structured KB modules for branches, fleet, daily/monthly pricing, drop-off fees, requirements, card/deposit policy, roadside, escalation, and FAQs.
3. Added strict GPT extraction and conversation-classification contracts.
4. Added explicit daily and roadside phase enums and phase-based required fields.
5. Implemented deterministic lookup over JSON KB modules.
6. Implemented code-only daily/monthly quote calculators.
7. Added OpenAI service boundary with environment-only API key loading and fallback behavior.
8. Added language detection, confirmation detection, ambiguity handling, and operational routing.
9. Added workflow state transitions for daily rental, payment confirmation, booking, roadside, and rejection flows.
10. Added mock external services: booking request, payment processing, roadside case creation, and agent handover.
11. Added rule engine, Khalid response composer, persona response planner, and response guard.
12. Added one reusable orchestrator entrypoint: `handle_message(state, message, service)`.
13. Added FastAPI `/chat`, `/healthz`, and session reset endpoints using the same core as Streamlit.
14. Added conversation management intents for greeting, small talk, off-topic, thanks, capability, unclear relevant messages, complaints, roadside, and fallback.
15. Added Khalid persona, tone modes, and response phrasing constraints.
16. Added semantic route metadata (`KB16`) as routing hints only, not facts.
17. Added OpenAI embeddings/vector search for informational KB content with lexical fallback.
18. Added enriched Avis Saudi general information in KB14.
19. Added optional Saudi tone lexicon (`KB17`) as controlled style options, not hard-coded mandatory phrases.
20. Added short conversation history: recent turns and compact summary for context only.
21. Refined complaint handling so unclear negative sentiment asks a clarification question before escalation.
22. Refined tone lexicon behavior to avoid stacked greetings such as `هلا وارحب، مرحبًا، تشرفنا`.

## Current files of interest
- `avis_ai_demo/core/orchestrator.py`: shared runtime coordinator for Streamlit and FastAPI.
- `avis_ai_demo/core/conversation_manager.py`: hybrid deterministic/GPT conversation classification.
- `avis_ai_demo/core/conversation_history.py`: bounded short-term conversation memory.
- `avis_ai_demo/core/conversation_prompt.py`: Khalid persona contract.
- `avis_ai_demo/core/persona_response_planner.py`: controlled payload for phrasing.
- `avis_ai_demo/core/response_composer.py`: deterministic response drafts plus Khalid phrasing.
- `avis_ai_demo/core/response_guard.py`: final customer-facing safety validator.
- `avis_ai_demo/core/tone_lexicon.py`: optional Saudi phrase selection and de-duplication.
- `avis_ai_demo/core/vector_search.py`: informational semantic KB retrieval.
- `avis_ai_demo/core/semantic_router.py`: semantic workflow/KB route candidates.
- `avis_ai_demo/services/openai_service.py`: OpenAI extraction, classification, embeddings, and phrasing boundary.
- `avis_ai_demo/api.py`: FastAPI backend.

## Khalid persona state
Khalid is a customer-facing phrasing layer. Khalid may sound warm, adaptive, and lightly Saudi in casual contexts, but he only speaks using facts supplied by the deterministic pipeline.

Current tone modes:
- `friendly_casual`: greetings, small talk, light casual messages.
- `professional_helpful`: booking, prices, branches, requirements, and normal service requests.
- `serious_supportive`: complaints, payment concerns, deposits, frustration, and unclear negative sentiment.
- `safety_first`: accidents, breakdowns, danger, and roadside assistance.
- `light_deflection`: off-topic questions with a short redirect to Avis services.

Saudi phrases are optional style choices. They must not be stacked, repeated, or used as templates. Serious, safety, payment, quote, and high-risk responses block casual phrase injection.

## Conversation history
The bot keeps a bounded short-term memory:

- `conversation_turns`: last 8 user/assistant turns, text-trimmed.
- `conversation_summary`: compact deterministic summary.
- The memory is passed to GPT only as conversational context.
- It is not a source for prices, availability, branch facts, payment status, booking status, or policies.

## How to run
Install dependencies if needed:

```bash
python3 -m pip install -r requirements.txt
```

Run tests:

```bash
python3 -m pytest avis_ai_demo/tests
```

Run Streamlit:

```bash
set -a
source .env.local
set +a
python3 -m streamlit run app_streamlit.py --server.address 127.0.0.1 --server.port 8501
```

Run FastAPI:

```bash
set -a
source .env.local
set +a
python3 -m uvicorn avis_ai_demo.api:app --host 127.0.0.1 --port 8000
```

Example HTTP request:

```bash
curl -sS http://127.0.0.1:8000/chat \
  -H 'Content-Type: application/json' \
  -d '{"session_id":"demo","message":"أبغى يارس من الرياض للدمام يومين"}'
```

## Environment variables
- `OPENAI_API_KEY`: required for live OpenAI classification, extraction, phrasing, and embeddings.
- `OPENAI_MODEL`: optional, defaults to `gpt-5.4`.
- `OPENAI_EMBEDDING_MODEL`: optional, defaults to `text-embedding-3-small`.

Keys are read only from environment variables and must not be committed or printed.

## Current test status
Latest full test command:

```bash
python3 -m pytest avis_ai_demo/tests
```

Latest result:

```text
123 passed
```

Coverage areas include:
- Structured KB contracts.
- GPT JSON schema validation and fallback.
- Conversation management and Khalid persona tone.
- Short conversation history.
- Semantic route metadata and vector search.
- Daily/monthly price separation.
- Code-only calculators.
- Quote and payment workflow boundaries.
- Roadside and escalation flows.
- Response guard blocked categories and fallback.
- FastAPI contract.
- Streamlit/core import separation.
- API key leakage checks.

## Known limitations
- Data remains demo-grade and limited to the structured source material plus provided general Avis Saudi facts.
- External services are mock adapters with controlled reference IDs.
- No branch-level vehicle inventory is implemented by design.
- Semantic search is for informational KB retrieval and routing hints, not operational truth.
- Conversation history is intentionally short and not persisted outside the current in-memory session.

## Next production steps
- Replace mock service adapters with authenticated Avis systems.
- Add persistent session storage with retention controls.
- Add observability for classification, retrieval, guard failures, and handovers.
- Add production data ingestion for branches, prices, fleet, and policies.
- Add a formal prompt/version registry and evaluation snapshots for Khalid tone.
- Expand Arabic date/time normalization and production booking validations.
