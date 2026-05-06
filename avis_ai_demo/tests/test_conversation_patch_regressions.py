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


def test_service_experience_feedback_is_not_financial_dispute():
    response, state, trace = handle_message(None, "يا خالد أمس رحت فرعكم وطلعت متضايق بصراحة", service())

    assert trace["intent"] == Intent.SERVICE_EXPERIENCE_FEEDBACK.value
    assert trace["escalation_status"] is False
    assert "تجربة" in response or "ملاحظة الفرع" in response
    assert "تاريخ العملية" not in response
    assert "رقم الحجز أو العقد، رقم الجوال، وتاريخ العملية" not in response

    response, _state, trace = handle_message(state, "فرع المطار كان زحمة والموظف تأخر علي، بس ما عندي مشكلة فلوس", service())

    assert trace["intent"] == Intent.SERVICE_EXPERIENCE_FEEDBACK.value
    assert trace["escalation_status"] is False
    assert "تاريخ العملية" not in response


def test_deposit_policy_topic_shift_answers_policy_not_license():
    _response, state, _trace = handle_message(None, "كنت بفكر أحجز سيارة بكرة من الرياض", service())
    response, state, trace = handle_message(state, "قبلها سؤال، الوديعة متى ترجع عادة؟", service())

    assert trace["intent"] == Intent.CARD_DEPOSIT_POLICY.value
    assert "5 إلى 10 أيام عمل" in response
    assert "رخصة قيادة" not in response
    assert state.suspended_workflow == Intent.DAILY_RENTAL.value


def test_debit_card_is_card_policy_not_discount_offer():
    response, state, trace = handle_message(None, "عندي بطاقة خصم مو ائتمان، ينفع أستأجر؟", service())

    assert trace["intent"] == Intent.CARD_DEPOSIT_POLICY.value
    assert "بطاقة ائتمان" in response or "بطاقات الخصم" in response
    assert "الخصومات والعروض" not in response

    response, _state, trace = handle_message(state, "ولو ما ينفع، فيه طريقة ثانية؟", service())

    assert trace["intent"] == Intent.CARD_DEPOSIT_POLICY.value
    assert "الخصومات والعروض" not in response


def test_mixed_branch_price_request_answers_both_branch_and_price():
    _response, state, _trace = handle_message(None, "أنا قريب من مطار جدة وبفكر آخذ كامري يوم واحد", service())
    response, _state, trace = handle_message(state, "طيب هل عندكم فرع هناك وكم تقريبًا سعرها؟", service())

    assert trace["intent"] == Intent.FLEET_PRICING.value
    assert "KB01" in trace["kb_modules_used"]
    assert "KB03" in trace["kb_modules_used"]
    assert "فرع" in response or "مطار" in response
    assert "355.35 ريال" in response


def test_route_correction_updates_cities_and_records_trace():
    _response, state, _trace = handle_message(None, "أبغى سيارة من الرياض لجدة يومين", service())
    response, state, trace = handle_message(state, "لا قصدي من جدة للرياض، والاستلام بكرة الساعة 9", service())

    assert trace["intent"] == Intent.DAILY_RENTAL.value
    assert state.entities["pickup_city"] == "Jeddah"
    assert state.entities["dropoff_city"] == "Riyadh"
    assert "pickup_city" in trace["corrections_applied"]
    assert "رخصة قيادة" in response


def test_financial_complaint_details_are_not_requested_again():
    _response, state, _trace = handle_message(None, "انخصم مني مبلغ وأبيك ترجعونه اليوم", service())
    response, state, trace = handle_message(state, "رقم الحجز AV123 وجوالي 0550001111 والعملية أمس", service())

    assert trace["intent"] in {Intent.COMPLAINT_OR_DISPUTE.value, Intent.COMPLAINT_OR_FINANCIAL_DISPUTE.value}
    assert state.support_case["has_minimum_details"] is True
    assert "وصلتني تفاصيل" in response
    assert "أحتاج رقم الحجز أو العقد" not in response


def test_monthly_calculator_prices_do_not_trigger_guard_failure():
    response, _state, trace = handle_message(None, "Can you tell me the monthly Camry price?", service())

    assert trace["intent"] == Intent.MONTHLY_RENTAL.value
    assert trace["computed_totals"]["total_inc_vat_sar"] == 7661.0
    assert "unsupported_price_claim" not in trace["guard_failures"]
    assert "7661" in response
