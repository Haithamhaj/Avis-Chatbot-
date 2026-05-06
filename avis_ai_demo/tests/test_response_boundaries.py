from avis_ai_demo.core.calculators import calculate_daily_quote
from avis_ai_demo.core.data_lookup import find_branch, find_daily_price, find_fleet_category
from avis_ai_demo.core.response_composer import compose_response
from avis_ai_demo.core.types import AnswerContext, Intent


def test_quote_response_is_invoice_like_and_contains_caveat():
    quote = calculate_daily_quote(
        find_daily_price("Yaris"),
        rental_days=2,
        pickup_city="Riyadh",
        dropoff_city="Dammam",
    )
    response = compose_response(
        AnswerContext(
            intent=Intent.DAILY_RENTAL,
            language="en",
            computed_totals=quote,
            phase="quote_presented",
        )
    )

    assert "Quotation summary" in response
    assert "Total online amount: 684.7 SAR" in response
    assert "Total in-branch amount: 716.9 SAR" in response
    assert "representative" in response.lower()


def test_payment_confirmation_response_only_after_payment_phase():
    quote = calculate_daily_quote(
        find_daily_price("Yaris"),
        rental_days=2,
        pickup_city="Riyadh",
        dropoff_city="Dammam",
    )
    response = compose_response(
        AnswerContext(
            intent=Intent.DAILY_RENTAL,
            language="en",
            computed_totals=quote,
            phase="payment_processed",
        )
    )

    assert "Payment completed successfully." in response
    assert "Booking completed successfully." in response


def test_branch_response_includes_holiday_caveat_not_internal_kb():
    branch = find_branch("Sulaymaniyah")[0]
    response = compose_response(
        AnswerContext(intent=Intent.BRANCH_LOOKUP, language="en"),
        {"branch": branch},
    )

    assert "Hours may differ" in response
    assert "KB" not in response


def test_model_response_uses_category_caveat():
    fleet = find_fleet_category("Yaris")
    response = compose_response(
        AnswerContext(intent=Intent.FLEET_PRICING, language="en"),
        {"fleet": fleet},
    )

    assert "representative" in response.lower()
    assert "guarantee the exact" not in response.lower()


def test_refund_dispute_and_accident_escalate():
    dispute = compose_response(
        AnswerContext(
            intent=Intent.COMPLAINT_OR_FINANCIAL_DISPUTE,
            language="en",
            escalation_required=True,
            risk_level="high",
        )
    )
    assert "relevant team" in dispute

    roadside = compose_response(AnswerContext(intent=Intent.ROADSIDE_ASSISTANCE, language="en"))
    assert "Your safety comes first" in roadside

