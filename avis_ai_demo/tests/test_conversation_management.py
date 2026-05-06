from avis_ai_demo.core.intent_router import route_intent
from avis_ai_demo.core.orchestrator import handle_message
from avis_ai_demo.core.types import Intent
from avis_ai_demo.services.openai_service import OpenAIService


class StubClient:
    def __init__(self, intent="fallback_unknown", language="ar", entities=None, conversation=None, phrase=None):
        self.payload = {
            "intent": intent,
            "language": language,
            "entities": entities or {},
            "ambiguities": [],
            "clarification_question": None,
            "confidence": 0.6,
        }
        self.conversation = conversation
        self.phrase = phrase

    def extract(self, message, state=None):
        return self.payload

    def classify_conversation(self, message, state=None):
        if self.conversation is not None:
            return self.conversation
        raise RuntimeError("no conversation classifier stub")

    def phrase_khalid_response(self, plan):
        if self.phrase is not None:
            return self.phrase(plan)
        return plan["deterministic_fallback"]


def service(intent="fallback_unknown", language="ar", entities=None, conversation=None, phrase=None):
    return OpenAIService(client=StubClient(intent, language, entities, conversation, phrase))


def test_greetings_do_not_route_to_fallback():
    assert route_intent("مرحبا") == Intent.GREETING
    assert route_intent("hello") == Intent.GREETING

    response, _state, trace = handle_message(None, "مرحبا", service())
    assert trace["intent"] == "greeting"
    assert trace["phase"] == "conversation_management"
    assert trace["tone_mode"] == "friendly_casual"
    assert "مرحبًا" in response
    assert "معك خالد" in response

    response, _state, trace = handle_message(None, "السلام عليكم", service())
    assert trace["intent"] == "greeting"
    assert response.startswith("وعليكم السلام")
    assert "معك خالد" in response
    assert "مرحبًا" not in response


def test_thanks_acknowledgement():
    response, _state, trace = handle_message(None, "شكرا", service())

    assert trace["intent"] == "thanks_acknowledgement"
    assert "على الرحب والسعة" in response


def test_off_topic_redirects_to_scope():
    response, state, trace = handle_message(None, "مين أحسن الهلال ولا النصر؟", service())

    assert trace["intent"] == "off_topic"
    assert trace["tone_mode"] == "light_deflection"
    assert state.next_action == "scope_redirect"
    assert "الفريقين كبار" in response
    assert "أفيس" in response


def test_what_can_you_do_returns_capability_list():
    response, _state, trace = handle_message(None, "What can you do?", service(language="en"))

    assert trace["intent"] == "capability_question"
    assert "daily or monthly prices" in response
    assert "roadside assistance" in response


def test_company_question_answers_from_general_faq_kb():
    response, _state, trace = handle_message(None, "كنت ماشي وشفت لوجو أفيس انتو شركة ايش؟", service())

    assert trace["intent"] == "general_faq"
    assert trace["kb_modules_used"] == ["KB14"]
    assert trace["lookup_records"] == ["company_overview"]
    assert "آيفس السعودية" in response or "أفيس السعودية" in response
    assert "تأجير السيارات" in response or "تأجير سيارات" in response


def test_company_question_with_logo_compliment_gets_contextual_opener():
    response, _state, trace = handle_message(
        None,
        "تعرف مبارح كنت مع بنتي وشفنا واحد من فروعكم كثير عجبنا اللوغو تبعكم انتو شركة ايش",
        service(),
    )

    assert trace["intent"] == "general_faq"
    assert trace["lookup_records"] == ["company_overview"]
    assert response.startswith("يسعدنا أن اللوغو عجبكم.")
    assert "آيفس السعودية" in response


def test_brand_question_uses_enriched_avis_company_facts():
    response, _state, trace = handle_message(None, "ليش أختار آيفس السعودية؟", service())

    assert trace["intent"] == "general_faq"
    assert "KB14" in trace["kb_modules_used"]
    assert trace["lookup_records"] == ["why_avis_saudi"]
    assert "منذ عام 1970" in response
    assert "جودة الخدمة" in response


def test_ambiguous_relevant_message_asks_clarification_not_fallback():
    response, _state, trace = handle_message(None, "أبغى أرتب الموضوع", service())

    assert trace["intent"] == "potentially_relevant_unclear"
    assert "تقصد حجز سيارة" in response
    assert trace["intent"] != "fallback_unknown"


def test_random_unclear_message_uses_fallback_with_service_options():
    response, _state, trace = handle_message(None, "flarble zint qqq", service(language="en"))

    assert trace["intent"] == "fallback_unknown"
    assert "not sure" in response.lower()
    assert "booking a car" in response


def test_small_talk_never_fallback_and_trace_is_clean():
    response, _state, trace = handle_message(None, "كيفك يا حلو؟", service())

    assert trace["intent"] == "small_talk"
    assert trace["phase"] == "conversation_management"
    assert trace["tone_mode"] == "friendly_casual"
    assert trace["kb_modules_used"] == []
    assert trace["lookup_records"] == []
    assert trace["risk_level"] == "low"
    assert trace["next_action"] == "scope_redirect"
    assert trace["escalation_status"] is False
    assert response in {
        "أهلًا، أنا بخير. كيف أقدر أساعدك؟",
        "هلا وارحب، أهلًا، أنا بخير. كيف أقدر أساعدك؟",
    }
    assert "يا حلو" not in response


def test_small_talk_variants_arabic_and_english():
    for message in ["كيفك", "شلونك"]:
        response, _state, trace = handle_message(None, message, service())
        assert trace["intent"] == "small_talk"
        assert "fallback" not in trace["intent"]
        assert response

    response, _state, trace = handle_message(None, "how are you", service(language="en"))
    assert trace["intent"] == "small_talk"
    assert trace["language"] == "en"
    assert "doing well" in response.lower()


def test_casual_followup_like_screenshot_stays_small_talk_not_fallback():
    response, _state, trace = handle_message(None, "انا بحكيلك يا بطل", service())

    assert trace["intent"] == "small_talk"
    assert trace["tone_mode"] == "friendly_casual"
    assert trace["kb_modules_used"] == []
    assert trace["lookup_records"] == []
    assert "ما فهمت" not in response
    assert "أقدر أساعدك في حجز سيارة" not in response
    assert response in {
        "أهلًا، أنا بخير. كيف أقدر أساعدك؟",
        "هلا وارحب، أهلًا، أنا بخير. كيف أقدر أساعدك؟",
    }
    assert "يا بطل" not in response


def test_small_talk_does_not_invent_saved_nicknames():
    for message in ["كيفك يا اخ", "كيفك يا حلو؟", "انا بحكيلك يا بطل", "شلونك يا غالي"]:
        response, _state, trace = handle_message(None, message, service())
        assert trace["intent"] == "small_talk"
        assert "يا حلو" not in response
        assert "يا بطل" not in response
        assert "يا غالي" not in response
        assert response.startswith(("أهلًا", "هلا وارحب"))


def test_semantic_gpt_conversation_classifier_handles_unlisted_casual_phrase():
    response, _state, trace = handle_message(
        None,
        "يا زلمة خلينا نحكي شوي",
        service(
            conversation={
                "intent": "small_talk",
                "tone_mode": "friendly_casual",
                "customer_mood": "casual",
                "next_action": "scope_redirect",
                "confidence": 0.91,
            }
        ),
    )

    assert trace["intent"] == "small_talk"
    assert trace["tone_mode"] == "friendly_casual"
    assert trace["gpt_fallback_used"] is False
    assert "ما فهمت" not in response


def test_low_confidence_semantic_classifier_asks_clarification_not_fallback():
    response, _state, trace = handle_message(
        None,
        "تعرف مبارح كنت ماشي وقلت خليني أسأل",
        service(
            conversation={
                "intent": "small_talk",
                "tone_mode": "friendly_casual",
                "customer_mood": "casual",
                "next_action": "scope_redirect",
                "confidence": 0.34,
            }
        ),
    )

    assert trace["intent"] == "potentially_relevant_unclear"
    assert trace["classifier_confidence"] == 0.34
    assert "تقصد حجز سيارة" in response
    assert trace["intent"] != "fallback_unknown"


def test_persona_phrasing_receives_controlled_plan_only():
    def phrase(plan):
        assert plan["intent"] == "small_talk"
        assert plan["tone_mode"] == "friendly_casual"
        assert plan["customer_mood"] == "casual"
        assert "forbidden_claims" in plan
        assert "computed_totals" in plan
        return "أهلًا، سوالف أفيس لها جو. كيف أقدر أساعدك؟"

    response, _state, trace = handle_message(
        None,
        "يا زلمة خلينا نحكي شوي",
        service(
            conversation={
                "intent": "small_talk",
                "tone_mode": "friendly_casual",
                "customer_mood": "casual",
                "next_action": "scope_redirect",
                "confidence": 0.91,
            },
            phrase=phrase,
        ),
    )

    assert trace["intent"] == "small_talk"
    assert "سوالف أفيس" in response


def test_operational_complaint_and_roadside_tones():
    response, _state, trace = handle_message(None, "الوديعة ما رجعت", service())
    assert trace["intent"] == "complaint_or_dispute"
    assert trace["tone_mode"] == "serious_supportive"
    assert "أفهم عليك" in response
    assert "رقم الحجز أو العقد" in response

    response, _state, trace = handle_message(None, "صار لي حادث", service())
    assert trace["intent"] == "roadside_or_accident"
    assert trace["tone_mode"] == "safety_first"
    assert "سلامتك أولًا" in response
    assert "إصابات" in response


def test_unclear_negative_sentiment_asks_before_escalation():
    response, _state, trace = handle_message(None, "تعرف يا اخي انا ما احب ايفيس احسكم نصابين", service())

    assert trace["intent"] == "potentially_relevant_unclear"
    assert trace["phase"] == "conversation_management"
    assert trace["tone_mode"] == "serious_supportive"
    assert trace["escalation_status"] is False
    assert "عشان أساعدك صح" in response
    assert "رقم الحجز أو العقد" not in response
    assert "رقم الجوال" not in response
    assert "تاريخ العملية" not in response


def test_gpt_complaint_classification_needs_operational_detail_before_escalation():
    response, _state, trace = handle_message(
        None,
        "تعرف يا اخي انا ما احب ايفيس احسكم نصابين",
        service(
            conversation={
                "intent": "complaint_or_dispute",
                "tone_mode": "serious_supportive",
                "customer_mood": "frustrated",
                "next_action": "collect_case_details",
                "confidence": 0.92,
            }
        ),
    )

    assert trace["intent"] == "potentially_relevant_unclear"
    assert trace["escalation_status"] is False
    assert "عشان أساعدك صح" in response
    assert "رقم الحجز أو العقد" not in response


def test_explicit_payment_dispute_still_escalates():
    response, _state, trace = handle_message(None, "انخصم مني مبلغ", service())

    assert trace["intent"] == "complaint_or_dispute"
    assert trace["tone_mode"] == "serious_supportive"
    assert trace["escalation_status"] is True
    assert "رقم الحجز أو العقد" in response


def test_discount_request_is_not_financial_dispute():
    response, _state, trace = handle_message(
        None,
        "شوف انا بصراحة بحب ايفيس ولازم تعطوني خصم مقابل هذا الحب هههههه",
        service(),
    )

    assert trace["intent"] == "general_faq"
    assert trace["escalation_status"] is False
    assert trace["risk_level"] == "low"
    assert "رقم الحجز أو العقد" not in response
    assert "الفريق المختص" not in response
    assert "وصلت المحبة" in response
    assert "الحجز الإلكتروني" in response or "السعر المتاح" in response


def test_unclear_negative_followup_does_not_repeat_same_template():
    first_response, state, first_trace = handle_message(
        None,
        "تعرف يا اخي انا ما احب ايفيس احسكم نصابين",
        service(),
    )
    second_response, _state, second_trace = handle_message(state, "لا انتو كذا بدون سبب", service())

    assert first_trace["intent"] == "potentially_relevant_unclear"
    assert second_trace["intent"] == "potentially_relevant_unclear"
    assert second_trace["next_action"] == "ask_open_clarification"
    assert second_trace["escalation_status"] is False
    assert second_response != first_response
    assert "إيش اللي خلاك" in second_response
    assert "هل الموضوع بخصوص حجز" not in second_response
    assert "رقم الحجز أو العقد" not in second_response


def test_conversation_history_is_short_and_contextual():
    state = None
    for message in [
        "مرحبا",
        "كيفك",
        "تعرف يا اخي انا ما احب ايفيس احسكم نصابين",
        "لا انتو كذا بدون سبب",
        "شكرا",
    ]:
        _response, state, trace = handle_message(state, message, service())

    assert len(state.conversation_turns) <= 8
    assert state.conversation_summary
    assert trace["conversation_history"]["turn_count"] == len(state.conversation_turns)
    assert "last_intent=" in trace["conversation_history"]["summary"]
    assert all(len(turn["text"]) <= 280 for turn in state.conversation_turns)
