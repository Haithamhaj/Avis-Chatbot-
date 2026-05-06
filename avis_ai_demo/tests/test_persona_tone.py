from avis_ai_demo.core.conversation_prompt import KHALID_PERSONA_PROMPT
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
            "confidence": 0.6,
        }

    def extract(self, message, state=None):
        return self.payload


def service(intent="fallback_unknown", language="ar", entities=None):
    return OpenAIService(client=StubClient(intent, language, entities))


def test_khalid_persona_prompt_exists_with_boundaries():
    assert "You are Khalid" in KHALID_PERSONA_PROMPT
    assert "Do not calculate totals" in KHALID_PERSONA_PROMPT
    assert "Do not invent availability" in KHALID_PERSONA_PROMPT
    assert "Do not expose KB IDs" in KHALID_PERSONA_PROMPT
    assert "Conversation memory" in KHALID_PERSONA_PROMPT
    assert "Use memory only for conversational continuity" in KHALID_PERSONA_PROMPT
    assert "Do not stack greetings" in KHALID_PERSONA_PROMPT
    assert "Use at most one phrase" in KHALID_PERSONA_PROMPT
    assert 'respond to "السلام عليكم"' in KHALID_PERSONA_PROMPT


def test_operational_request_uses_professional_helpful_tone():
    response, _state, trace = handle_message(None, "أبغى سيارة من الرياض بكرة", service())

    assert trace["intent"] == "daily_rental"
    assert trace["tone_mode"] == "professional_helpful"
    assert "من أي فرع" in response
    assert "كم مدة" in response


def test_greeting_and_off_topic_persona_are_lively_but_scoped():
    response, _state, trace = handle_message(None, "hello", service(language="en"))
    assert trace["intent"] == "greeting"
    assert trace["tone_mode"] == "friendly_casual"
    assert "Hello" in response

    response, _state, trace = handle_message(None, "مين أحسن الهلال ولا النصر؟", service())
    assert trace["intent"] == "off_topic"
    assert trace["tone_mode"] == "light_deflection"
    assert "باب طويل" in response
    assert "أفيس" in response
