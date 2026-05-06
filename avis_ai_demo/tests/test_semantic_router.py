from avis_ai_demo.core.orchestrator import handle_message
from avis_ai_demo.core.semantic_router import semantic_route_candidates
from avis_ai_demo.services.openai_service import OpenAIService


class RouteEmbeddingStubClient:
    def __init__(self, extraction=None):
        self.extraction = extraction or {
            "intent": "fallback_unknown",
            "language": "ar",
            "entities": {},
            "ambiguities": [],
            "clarification_question": None,
            "confidence": 0.5,
        }

    def extract(self, message, state=None):
        return self.extraction

    def embed_texts(self, texts):
        return [self._embed(text) for text in texts]

    def _embed(self, text):
        lowered = text.lower()
        if any(token in lowered for token in ["موتر", "daily_rental_workflow", "rent a car"]):
            return [1.0, 0.0, 0.0, 0.0]
        if any(token in lowered for token in ["وديعة", "deposit_policy", "wrong charge"]):
            return [0.0, 1.0, 0.0, 0.0]
        if any(token in lowered for token in ["سواق", "chauffeur_service"]):
            return [0.0, 0.0, 1.0, 0.0]
        if any(token in lowered for token in ["شركات", "corporate_services", "business account"]):
            return [0.0, 0.0, 0.0, 1.0]
        if any(token in lowered for token in ["مميزات", "why choose avis", "brand_company_info"]):
            return [0.0, 0.0, 0.0, 0.0, 1.0]
        return [0.25, 0.25, 0.25, 0.25, 0.25]


def service(extraction=None):
    return OpenAIService(client=RouteEmbeddingStubClient(extraction))


def test_semantic_route_metadata_returns_workflow_candidate_not_answer():
    candidates = semantic_route_candidates("ودي أرتب موتر بكرة", service())

    assert candidates
    assert candidates[0].route_id == "daily_rental_workflow"
    assert candidates[0].target_intent.value == "daily_rental"
    assert "workflow_daily_rental" in candidates[0].target_modules


def test_orchestrator_uses_semantic_route_candidate_for_unclear_workflow():
    response, _state, trace = handle_message(None, "ودي أرتب موتر بكرة", service())

    assert trace["intent"] == "daily_rental"
    assert trace["semantic_route_selected"]["route_id"] == "daily_rental_workflow"
    assert trace["computed_totals"] is None
    assert trace["missing_fields"] == ["has_valid_license"]


def test_safety_and_dispute_overrides_are_not_weakened_by_semantic_routes():
    response, _state, trace = handle_message(None, "الوديعة ما رجعت", service())

    assert trace["intent"] == "complaint_or_dispute"
    assert trace["semantic_route_selected"] is None
    assert trace["escalation_status"] is True
    assert "الفريق المختص" in response


def test_corporate_question_routes_to_general_faq_instead_of_capability_menu():
    extraction = {
        "intent": "fallback_unknown",
        "language": "ar",
        "entities": {},
        "ambiguities": [],
        "clarification_question": None,
        "confidence": 0.4,
    }

    response, _state, trace = handle_message(None, "عندكم حلول للشركات؟", service(extraction))

    assert trace["intent"] == "general_faq"
    assert "KB12" in trace["kb_modules_used"]
    assert trace["semantic_search"]["source_kb"] == "KB12"
    assert "حجز سيارة" not in response


def test_brand_route_metadata_points_to_kb14_brand_facts():
    candidates = semantic_route_candidates("ايش مميزات آيفس؟", service())

    assert candidates
    assert candidates[0].route_id == "brand_company_info"
    assert candidates[0].target_intent.value == "general_faq"
    assert candidates[0].target_modules == ["KB14"]
