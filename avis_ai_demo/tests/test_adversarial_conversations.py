from avis_ai_demo.core.orchestrator import handle_message
from avis_ai_demo.core.types import Intent
from avis_ai_demo.services.openai_service import OpenAIService


class StubClient:
    def extract(self, message, state=None):
        raise RuntimeError("force deterministic extraction")

    def classify_conversation(self, message, state=None):
        raise RuntimeError("force deterministic classification")

    def decide_conversation(self, message, state=None):
        raise RuntimeError("force deterministic decision")

    def phrase_khalid_response(self, plan):
        return plan["deterministic_fallback"]


def service():
    return OpenAIService(client=StubClient())


def test_playful_negative_sentiment_clarifies_then_financial_escalates():
    response, state, trace = handle_message(None, "يا خالد ترى أحبكم بس آخر مرة حسيتكم لعبتوا علي شوي ههههه", service())

    assert trace["intent"] == Intent.POTENTIALLY_RELEVANT_UNCLEAR.value
    assert trace["escalation_status"] is False
    assert "ما فهمت طلبك" not in response

    response, _state, trace = handle_message(state, "يعني عطوني خصم ولا رجعوا المبلغ اللي انخصم زيادة، مدري وش الصح", service())

    assert trace["intent"] in {Intent.COMPLAINT_OR_DISPUTE.value, Intent.COMPLAINT_OR_FINANCIAL_DISPUTE.value}
    assert trace["escalation_status"] is True
    assert "رقم الحجز" in response


def test_family_trip_hidden_booking_does_not_become_price_only_dead_end():
    _response, state, _trace = handle_message(None, "أنا جاي الرياض مع الوالد والعيال وبنلف كم مشوار ويمكن نروح الدمام", service())
    response, _state, trace = handle_message(state, "أبغى شي مريح بكرة العصر يومين بس لا تورطني بسعر غلط", service())

    assert trace["intent"] == Intent.DAILY_RENTAL.value
    assert "رخصة" in response


def test_branch_card_return_multi_intent_mentions_policy_and_quote_next_step():
    _response, state, _trace = handle_message(None, "أنا عند فرع المطار الآن وشفت كامري قدامي", service())
    response, _state, trace = handle_message(state, "ينفع آخذها ببطاقة مدى وأرجعها جدة بعد يومين؟", service())

    assert trace["intent"] == Intent.CARD_DEPOSIT_POLICY.value
    assert "KB08" in trace["kb_modules_used"]
    assert "بطاقة ائتمان" in response or "بطاقات الخصم" in response
    assert "عرض سعر" in response or "السعر" in response


def test_accident_followup_stays_safety_not_branch_or_price():
    _response, state, trace = handle_message(None, "صارلي حادث بسيط بس الحمدلله مافي شي", service())
    assert trace["intent"] in {Intent.ROADSIDE_OR_ACCIDENT.value, Intent.ROADSIDE_ASSISTANCE.value}

    response, _state, trace = handle_message(state, "لا لا خلاص السيارة تمشي، بس أبي أعرف إذا أكمل عليها للفرع ولا لازم أوقف", service())

    assert trace["intent"] in {Intent.ROADSIDE_OR_ACCIDENT.value, Intent.ROADSIDE_ASSISTANCE.value}
    assert "أوقف" in response or "توقف" in response
    assert "عرض سعر" not in response


def test_exact_model_guarantee_question_gets_caveat_not_fallback():
    _response, state, _trace = handle_message(None, "لا تعطيني كلام فئات، أبي كامري بيضاء 2026 بالضبط من فرع السليمانية", service())
    response, _state, trace = handle_message(state, "إذا قلت لك أكد هل تضمنها؟", service())

    assert trace["intent"] == Intent.FLEET_PRICING.value
    assert "ما أقدر أضمن" in response or "التوفر النهائي" in response
    assert "ما فهمت طلبك" not in response


def test_international_deposit_case_escalates_not_policy_answer():
    response, state, trace = handle_message(None, "استأجرت من أفيس دبي ورجعت السيارة في الرياض، والوديعة معلقة", service())

    assert trace["intent"] in {Intent.COMPLAINT_OR_DISPUTE.value, Intent.COMPLAINT_OR_FINANCIAL_DISPUTE.value}
    assert trace["escalation_status"] is True
    assert "5 إلى 10 أيام" not in response

    response, _state, trace = handle_message(state, "أقدر أعطيك رقم العقد بس أبي جواب نهائي الآن", service())

    assert trace["intent"] in {Intent.COMPLAINT_OR_DISPUTE.value, Intent.COMPLAINT_OR_FINANCIAL_DISPUTE.value}
    assert "لا أستطيع" in response or "الفريق المختص" in response


def test_daily_monthly_contradiction_asks_clarification_no_mixed_total():
    response, state, trace = handle_message(None, "أبغى كامري شهر بس يمكن يومين حسب السعر", service())

    assert trace["intent"] == Intent.POTENTIALLY_RELEVANT_UNCLEAR.value
    assert "يومي ولا شهري" in response
    assert trace["computed_totals"] is None

    response, _state, trace = handle_message(state, "احسبها لي من جدة للرياض بكرة 9، وإذا الشهري أرخص خله شهري", service())

    assert trace["intent"] == Intent.POTENTIALLY_RELEVANT_UNCLEAR.value
    assert "يومي ولا شهري" in response
    assert trace["computed_totals"] is None
    assert "totals_without_calculator_output" not in trace["guard_failures"]


def test_angry_vehicle_smoke_feedback_is_service_feedback():
    response, state, trace = handle_message(None, "بصراحة خدمتكم رفعت ضغطي ولا عاد أفكر أستأجر منكم", service())

    assert trace["intent"] in {Intent.SERVICE_EXPERIENCE_FEEDBACK.value, Intent.POTENTIALLY_RELEVANT_UNCLEAR.value}
    assert "رخصة قيادة" not in response

    response, _state, trace = handle_message(state, "الموضوع مو فلوس، السيارة كانت ريحتها دخان والفرع قال عادي", service())

    assert trace["intent"] == Intent.SERVICE_EXPERIENCE_FEEDBACK.value
    assert "ملاحظة" in response or "تجربتك" in response


def test_embedded_daily_quote_request_routes_to_calculator_then_price_followup_uses_context():
    response, state, trace = handle_message(None, "كم كامري يومين من الرياض للدمام؟ عندي رخصة والاستلام بكرة الساعة 10", service())

    assert trace["intent"] == Intent.DAILY_RENTAL.value
    assert trace["computed_totals"] is not None
    assert trace["computed_totals"]["totals_produced"] is True

    response, _state, trace = handle_message(state, "تمام بس قبل أدفع هل السعر يشمل كل شي؟", service())

    assert trace["intent"] in {Intent.CARD_DEPOSIT_POLICY.value, Intent.DAILY_RENTAL.value, Intent.FLEET_PRICING.value}
    assert "أي فئة أو موديل" not in response


def test_roadside_and_future_booking_mixed_context_prioritizes_safety():
    response, state, trace = handle_message(None, "أنا في محطة بين الرياض والدمام والسيارة ترجف، وبنفس الوقت عندي حجز ثاني بكرة", service())

    assert trace["intent"] in {Intent.ROADSIDE_OR_ACCIDENT.value, Intent.ROADSIDE_ASSISTANCE.value}
    assert "سلامتك" in response

    response, _state, trace = handle_message(state, "جوال 0551112222 والعقد RA4433 وما أدري أمشي ولا أوقف", service())

    assert trace["intent"] in {Intent.ROADSIDE_OR_ACCIDENT.value, Intent.ROADSIDE_ASSISTANCE.value}
    assert "بطاقة ائتمان" not in response
    assert "أوقف" in response or "توقف" in response
