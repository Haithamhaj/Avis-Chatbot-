from avis_ai_demo.core.orchestrator import handle_message
from avis_ai_demo.core.vector_search import semantic_kb_search
from avis_ai_demo.services.openai_service import OpenAIService


class EmbeddingStubClient:
    def __init__(self, extraction=None):
        self.extraction = extraction or {
            "intent": "fallback_unknown",
            "language": "ar",
            "entities": {},
            "ambiguities": [],
            "clarification_question": None,
            "confidence": 0.6,
        }

    def extract(self, message, state=None):
        return self.extraction

    def embed_texts(self, texts):
        return [self._embed(text) for text in texts]

    def _embed(self, text):
        lowered = text.lower()
        if any(token in lowered for token in ["سواق", "سائق", "driver", "chauffeur"]):
            return [1.0, 0.0, 0.0]
        if any(token in lowered for token in ["شركات", "corporate", "leasing", "أسطول"]):
            return [0.0, 1.0, 0.0]
        return [0.0, 0.0, 1.0]


def service(extraction=None):
    return OpenAIService(client=EmbeddingStubClient(extraction))


def test_semantic_kb_search_uses_openai_embeddings_for_informational_kb():
    results = semantic_kb_search("هل عندكم خدمة سيارة مع سواق؟", service())

    assert results
    assert results[0].source_kb == "KB11"
    assert results[0].record_id == "chauffeur_service"
    assert results[0].score == 1.0


def test_general_faq_handler_uses_vector_search_when_exact_alias_is_missing():
    extraction = {
        "intent": "general_faq",
        "language": "ar",
        "entities": {},
        "ambiguities": [],
        "clarification_question": None,
        "confidence": 0.8,
    }

    response, _state, trace = handle_message(None, "هل عندكم خدمة سيارة مع سواق؟", service(extraction))

    assert trace["intent"] == "general_faq"
    assert trace["semantic_search"]["provider"] == "openai_embeddings"
    assert trace["semantic_search"]["source_kb"] == "KB11"
    assert "KB11" in trace["kb_modules_used"]
    assert "السيارة مع سائق" in response


def test_avis_adjacent_fallback_can_use_vector_search_before_generic_fallback():
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
    assert trace["semantic_search"]["source_kb"] == "KB12"
    assert "أفهم طلبك" not in response
    assert "شركة" in response or "الشركة" in response
