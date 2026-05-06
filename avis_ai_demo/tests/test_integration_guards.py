import ast
from pathlib import Path

from avis_ai_demo.core.calculators import calculate_daily_quote, daily_price_summary
from avis_ai_demo.core.data_lookup import find_daily_price
from avis_ai_demo.core.response_composer import compose_response
from avis_ai_demo.core.response_guard import validate_customer_response
from avis_ai_demo.core.types import AnswerContext, Intent
from avis_ai_demo.services.mock_external_services import (
    create_booking_request,
    create_roadside_case,
    process_payment,
)


FORBIDDEN_CUSTOMER_WORDS = [
    "demo",
    "simulation",
    "mock",
    "not connected",
    "no api",
    "api absence",
    "تجريبي",
    "محاكاة",
    "غير متصل",
]


def _customer_outputs():
    quote = calculate_daily_quote(
        find_daily_price("Yaris"),
        rental_days=2,
        pickup_city="Riyadh",
        dropoff_city="Dammam",
    )
    yield compose_response(
        AnswerContext(intent=Intent.DAILY_RENTAL, language="en", computed_totals=quote, phase="quote_presented")
    )
    yield compose_response(
        AnswerContext(intent=Intent.DAILY_RENTAL, language="ar", computed_totals=quote, phase="payment_processed")
    )
    yield create_booking_request(quote, {"customer_name": "Haitham"}).payload["customer_message_en"]
    yield process_payment(quote, confirmed=True).payload["customer_message_en"]
    yield create_roadside_case(
        {
            "safety_status": "safe",
            "mobile": "0550000000",
            "plate_or_contract": "ABC123",
            "current_location": "Riyadh",
            "issue_type": "battery",
            "drivable_status": "not_drivable",
        }
    ).payload["customer_message_en"]


def test_scoped_customer_outputs_do_not_expose_demo_mock_or_api_wording():
    for output in _customer_outputs():
        lowered = output.lower()
        assert not any(word in lowered for word in FORBIDDEN_CUSTOMER_WORDS)


def test_no_quote_total_without_calculator_output_guard():
    context = AnswerContext(
        intent=Intent.FLEET_PRICING,
        language="en",
        computed_totals=daily_price_summary(find_daily_price("Yaris")),
    )
    result = validate_customer_response("Total online amount: 217.35 SAR", context)

    assert "totals_without_calculator_output" in result.failures


def test_ai_cannot_override_totals_in_customer_response_context():
    quote = calculate_daily_quote(
        find_daily_price("Yaris"),
        rental_days=2,
        pickup_city="Riyadh",
        dropoff_city="Dammam",
    )
    context = AnswerContext(intent=Intent.DAILY_RENTAL, language="en", computed_totals=quote)
    safe = compose_response(context)

    assert "999" not in safe
    assert "684.7" in safe


def test_daily_monthly_mixing_and_branch_inventory_regressions():
    response = compose_response(
        AnswerContext(intent=Intent.FLEET_PRICING, language="en"),
        {"daily_price": find_daily_price("Yaris")},
    )

    assert "Online price: 217.35 SAR" in response
    assert "inventory" not in response.lower()
    assert "which branch" not in response.lower()


def test_core_has_no_streamlit_imports_and_no_key_literals():
    root = Path(__file__).resolve().parents[2]
    for path in (root / "avis_ai_demo" / "core").glob("*.py"):
        source = path.read_text(encoding="utf-8")
        assert "sk-" not in source
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                assert all(alias.name != "streamlit" for alias in node.names), path
            if isinstance(node, ast.ImportFrom):
                assert node.module != "streamlit", path

    for folder in ["core", "services", "adapters"]:
        for path in (root / "avis_ai_demo" / folder).rglob("*.py"):
            assert "sk-" not in path.read_text(encoding="utf-8")

        assert "sk-" not in path.read_text(encoding="utf-8")
