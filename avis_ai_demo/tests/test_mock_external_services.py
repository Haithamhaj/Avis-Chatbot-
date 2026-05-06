import pytest

from avis_ai_demo.core.calculators import calculate_daily_quote
from avis_ai_demo.core.data_lookup import find_daily_price
from avis_ai_demo.services.mock_external_services import (
    create_booking_request,
    create_roadside_case,
    handover_to_agent,
    process_payment,
)


FORBIDDEN_CUSTOMER_WORDS = ["demo", "simulation", "mock", "not connected", "api absence", "تجريبي", "محاكاة"]


def _quote():
    return calculate_daily_quote(
        find_daily_price("Yaris"),
        rental_days=2,
        pickup_city="Riyadh",
        dropoff_city="Dammam",
    )


def _assert_customer_messages_are_operational(payload):
    for key, value in payload.items():
        if key.startswith("customer_message"):
            lowered = value.lower()
            assert not any(word in lowered for word in FORBIDDEN_CUSTOMER_WORDS)


def test_create_booking_request_requires_calculator_quote():
    result = create_booking_request(_quote(), {"customer_name": "Haitham"})

    assert result.action == "create_booking_request"
    assert result.reference_id.startswith("AVIS-REQ-")
    _assert_customer_messages_are_operational(result.payload)

    with pytest.raises(ValueError):
        create_booking_request(None, {"customer_name": "Haitham"})


def test_process_payment_requires_confirmation():
    quote = _quote()
    result = process_payment(quote, confirmed=True)

    assert result.action == "process_payment"
    assert result.status == "processed"
    assert result.payload["paid_amount_online_sar"] == 684.7
    _assert_customer_messages_are_operational(result.payload)

    with pytest.raises(ValueError):
        process_payment(quote, confirmed=False)


def test_create_roadside_case_requires_required_fields_but_not_notes():
    result = create_roadside_case(
        {
            "safety_status": "safe",
            "mobile": "0550000000",
            "plate_or_contract": "ABC123",
            "current_location": "Riyadh",
            "issue_type": "battery",
            "drivable_status": "not_drivable",
        }
    )

    assert result.action == "create_roadside_case"
    assert result.reference_id.startswith("AVIS-SOS-")
    _assert_customer_messages_are_operational(result.payload)

    with pytest.raises(ValueError):
        create_roadside_case({"safety_status": "safe"})


def test_handover_to_agent_returns_reference_without_customer_disclaimer():
    result = handover_to_agent("financial_dispute", {"mobile": "0550000000"})

    assert result.action == "handover_to_agent"
    assert result.reference_id.startswith("AVIS-CASE-")
    _assert_customer_messages_are_operational(result.payload)

