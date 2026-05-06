from avis_ai_demo.core.orchestrator import handle_message
from avis_ai_demo.services.openai_service import OpenAIService


class StubClient:
    def __init__(self, intent="fallback_unknown", language="ar", entities=None):
        self.payload = {
            "intent": intent,
            "language": language,
            "entities": entities or {},
            "ambiguities": [],
            "clarification_question": None,
            "confidence": 0.7,
        }

    def extract(self, message, state=None):
        return self.payload


def service(intent="fallback_unknown", language="ar", entities=None):
    return OpenAIService(client=StubClient(intent, language, entities))


def test_tone_lexicon_adds_one_safe_phrase_for_small_talk():
    response, _state, trace = handle_message(None, "كيفك يا اخ", service())

    assert trace["intent"] == "small_talk"
    assert response.count("هلا وارحب") <= 1
    assert "يا حلو" not in response


def test_tone_lexicon_can_add_service_acknowledgement_for_booking_clarification():
    response, _state, trace = handle_message(None, "أبغى سيارة من الرياض بكرة", service())

    assert trace["intent"] == "daily_rental"
    assert response.count("أبشر") <= 1
    assert response.startswith(("أبشر،", "أكيد."))
    assert trace["computed_totals"] is None


def test_tone_lexicon_does_not_stack_greetings_from_phrasing():
    class PhraseClient(StubClient):
        def phrase_khalid_response(self, plan):
            return "هلا وارحب، مرحبًا، تشرفنا. كيف أقدر أساعدك؟"

    response, _state, trace = handle_message(None, "ايش الاخبار", OpenAIService(client=PhraseClient()))

    assert trace["intent"] == "small_talk"
    assert response.count("هلا وارحب") <= 1
    assert "مرحبًا، تشرفنا" not in response
    assert not response.startswith("هلا وارحب، مرحبًا، تشرفنا")


def test_tone_lexicon_is_not_used_for_complaints_or_roadside():
    response, _state, trace = handle_message(None, "الوديعة ما رجعت", service())
    assert trace["intent"] == "complaint_or_dispute"
    assert not response.startswith(("هلا وارحب", "أبشر", "حاضر طال عمرك"))

    response, _state, trace = handle_message(None, "صار لي حادث", service())
    assert trace["intent"] == "roadside_or_accident"
    assert not response.startswith(("هلا وارحب", "أبشر", "حاضر طال عمرك"))
