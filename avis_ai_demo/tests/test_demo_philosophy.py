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
            "confidence": 0.5,
        }

    def extract(self, message, state=None):
        return self.payload


def service(intent="fallback_unknown", language="ar", entities=None):
    return OpenAIService(client=StubClient(intent, language, entities))


def test_demo_moment_human_small_talk_is_not_fallback():
    response, _state, trace = handle_message(None, "كيفك يا حلو؟", service())

    assert response in {
        "أهلًا، أنا بخير. كيف أقدر أساعدك؟",
        "هلا وارحب، أهلًا، أنا بخير. كيف أقدر أساعدك؟",
    }
    assert "يا حلو" not in response
    assert trace["intent"] == "small_talk"
    assert trace["phase"] == "conversation_management"
    assert trace["tone_mode"] == "friendly_casual"
    assert trace["kb_modules_used"] == []
    assert trace["lookup_records"] == []


def test_demo_moment_one_sentence_booking_understanding_and_calculator_quote():
    message = (
        "أبغى يارس من الرياض للدمام يومين، اسمي هيثم عمري 30 وعندي رخصة، "
        "الاستلام 2026-05-10 الساعة 14:00"
    )
    response, state, trace = handle_message(None, message, service())

    assert trace["intent"] == "daily_rental"
    assert trace["tone_mode"] == "professional_helpful"
    assert trace["phase"] == "awaiting_payment_confirmation"
    assert trace["computed_totals"]["total_online_sar"] == 684.7
    assert trace["computed_totals"]["total_in_branch_sar"] == 716.9
    assert state.quote.calculator_source == "calculate_daily_quote"
    assert "الإجمالي الإلكتروني: 684.7 ريال" in response
    assert "إجمالي الفرع: 716.9 ريال" in response


def test_demo_moment_safe_escalation_for_payment_and_deposit_issues():
    for message in ["الوديعة ما رجعت", "عندي مشكلة في الدفع"]:
        response, _state, trace = handle_message(None, message, service())

        assert trace["intent"] == "complaint_or_dispute"
        assert trace["tone_mode"] == "serious_supportive"
        assert trace["risk_level"] == "high"
        assert trace["escalation_status"] is True
        assert "رقم الحجز أو العقد" in response
        assert "الفريق المختص" in response


def test_demo_moment_accident_uses_safety_first_not_improvisation():
    response, _state, trace = handle_message(None, "صار لي حادث", service())

    assert trace["intent"] == "roadside_or_accident"
    assert trace["tone_mode"] == "safety_first"
    assert trace["risk_level"] == "low"
    assert "سلامتك أولًا" in response
    assert "إصابات" in response
    assert "الطوارئ" in response
