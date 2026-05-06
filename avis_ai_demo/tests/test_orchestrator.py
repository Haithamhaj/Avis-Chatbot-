from avis_ai_demo.core.orchestrator import handle_message
from avis_ai_demo.core.types import DailyRentalPhase, RoadsidePhase, WorkflowState
from avis_ai_demo.services.openai_service import OpenAIService


class StubClient:
    def __init__(self, payload):
        self.payload = payload

    def extract(self, message, state=None):
        return self.payload


def service(payload):
    return OpenAIService(client=StubClient(payload))


def test_arabic_daily_quote_and_payment_flow():
    extraction = {
        "intent": "daily_rental",
        "language": "ar",
        "entities": {
            "customer_name": "هيثم",
            "age": 30,
            "has_valid_license": True,
            "pickup_city": "Riyadh",
            "dropoff_city": "Dammam",
            "rental_days": 2,
            "pickup_date": "2026-05-10",
            "pickup_time": "14:00",
            "vehicle_query": "Yaris",
        },
        "ambiguities": [],
        "clarification_question": None,
        "confidence": 0.95,
    }
    response, state, trace = handle_message(None, "أبغى يارس من الرياض للدمام يومين", service(extraction))

    assert "ملخص عرض السعر" in response
    assert "الإجمالي الإلكتروني: 684.7 ريال" in response
    assert state.phase == DailyRentalPhase.AWAITING_PAYMENT_CONFIRMATION.value
    assert trace["computed_totals"]["total_online_sar"] == 684.7
    assert "KB03" in trace["kb_modules_used"]

    confirm_extraction = {
        "intent": "daily_rental",
        "language": "ar",
        "entities": {},
        "ambiguities": [],
        "clarification_question": None,
        "confidence": 0.9,
    }
    response, state, trace = handle_message(state, "نعم", service(confirm_extraction))

    assert "تم الدفع بنجاح" in response
    assert state.phase == DailyRentalPhase.BOOKING_COMPLETED.value
    assert len(trace["mock_action_results"]) == 2


def test_english_branch_lookup():
    extraction = {
        "intent": "branch_lookup",
        "language": "en",
        "entities": {"branch_or_city_query": "Sulaymaniyah"},
        "ambiguities": [],
        "clarification_question": None,
        "confidence": 0.9,
    }
    response, state, trace = handle_message(None, "where is Sulaymaniyah branch", service(extraction))

    assert "Riyadh Sulaymaniyah" in response
    assert trace["lookup_records"] == ["riyadh_sulaymaniyah"]


def test_deposit_escalation():
    extraction = {
        "intent": "card_deposit_policy",
        "language": "en",
        "entities": {},
        "ambiguities": [],
        "clarification_question": None,
        "confidence": 0.9,
    }
    response, state, trace = handle_message(None, "deposit not released and I need refund", service(extraction))

    assert "relevant team" in response
    assert trace["escalation_status"] is True
    assert "KB13" in trace["kb_modules_used"]


def test_roadside_assistance_without_notes_records_case():
    extraction = {
        "intent": "roadside_assistance",
        "language": "en",
        "entities": {
            "safety_status": "safe",
            "mobile": "0550000000",
            "plate_or_contract": "ABC123",
            "current_location": "Riyadh",
            "issue_type": "battery",
            "drivable_status": "not_drivable",
        },
        "ambiguities": [],
        "clarification_question": None,
        "confidence": 0.9,
    }
    response, state, trace = handle_message(
        WorkflowState(phase=RoadsidePhase.SAFETY_CHECK.value),
        "battery is dead",
        service(extraction),
    )

    assert "assistance request has been recorded" in response
    assert state.phase == RoadsidePhase.CASE_RECORDED.value
    assert trace["mock_action_results"][0]["action"] == "create_roadside_case"


def test_monthly_pricing_and_branchless_price_answer():
    monthly = {
        "intent": "monthly_rental",
        "language": "en",
        "entities": {"vehicle_query": "Camry"},
        "ambiguities": [],
        "clarification_question": None,
        "confidence": 0.9,
    }
    response, state, trace = handle_message(None, "monthly Camry", service(monthly))
    assert "7661.0" in response
    assert "KB04" in trace["kb_modules_used"]

    price = {
        "intent": "fleet_pricing",
        "language": "en",
        "entities": {"vehicle_query": "Yaris"},
        "ambiguities": [],
        "clarification_question": None,
        "confidence": 0.9,
    }
    response, state, trace = handle_message(None, "how much is Yaris", service(price))
    assert "Online price: 217.35 SAR" in response
    assert "branch" not in trace["missing_fields"]

