from avis_ai_demo.core.calculators import calculate_daily_quote, daily_price_summary
from avis_ai_demo.core.data_lookup import find_daily_price
from avis_ai_demo.core.response_guard import guard_or_fallback, validate_customer_response
from avis_ai_demo.core.types import AnswerContext, Intent


def _context(**kwargs):
    defaults = dict(intent=Intent.DAILY_RENTAL, language="en")
    defaults.update(kwargs)
    return AnswerContext(**defaults)


def test_guard_blocks_demo_mock_api_wording():
    result = validate_customer_response("This is a demo response from a mock API.", _context())
    assert "forbidden_demo_or_api_wording" in result.failures


def test_guard_blocks_payment_success_before_confirmation():
    result = validate_customer_response(
        "Payment completed successfully.",
        _context(phase="awaiting_payment_confirmation"),
    )
    assert "payment_success_before_confirmation" in result.failures


def test_guard_blocks_totals_without_calculator_output():
    result = validate_customer_response("Total online amount: 500 SAR", _context())
    assert "totals_without_calculator_output" in result.failures

    summary = daily_price_summary(find_daily_price("Yaris"))
    result = validate_customer_response(
        "Total online amount: 500 SAR",
        _context(computed_totals=summary),
    )
    assert "totals_without_calculator_output" in result.failures


def test_guard_blocks_kb_and_internal_trace_leakage():
    result = validate_customer_response("Used KB03 retrieved_records internally.", _context())
    assert "internal_trace_or_kb_leak" in result.failures

    result = validate_customer_response("Current phase is awaiting_payment_confirmation.", _context())
    assert "internal_trace_or_kb_leak" in result.failures


def test_guard_blocks_json_leakage_and_unsupported_prices():
    result = validate_customer_response('{"intent":"daily_rental","trace":{}}', _context())
    assert "json_leakage" in result.failures

    result = validate_customer_response("The price is 999 SAR.", _context())
    assert "unsupported_price_claim" in result.failures


def test_guard_blocks_exact_model_guarantees():
    result = validate_customer_response("We guarantee the exact model.", _context())
    assert "exact_model_guarantee" in result.failures


def test_guard_blocks_invented_price_when_allowed_facts_exist():
    result = validate_customer_response(
        "The price is 999 SAR.",
        _context(allowed_facts={"daily_price": {"online_total_vat_sar": 217.35}}),
    )
    assert "unsupported_price_claim" in result.failures


def test_guard_blocks_operational_promises_and_availability_claims():
    result = validate_customer_response("The car is available now.", _context())
    assert "exact_model_guarantee" in result.failures

    result = validate_customer_response("The truck will arrive within 10 minutes.", _context())
    assert "unsupported_operational_promise" in result.failures

    result = validate_customer_response("Refund approved.", _context())
    assert "unsupported_operational_promise" in result.failures


def test_guard_blocks_saved_casual_nicknames():
    result = validate_customer_response("أهلًا يا حلو، كيف أساعدك؟", _context(intent=Intent.SMALL_TALK, language="ar"))
    assert "unsupported_casual_nickname" in result.failures


def test_guard_fallback_replaces_unsafe_response_and_records_trace():
    context = _context()
    safe = guard_or_fallback("This demo uses KB03. Total: 100 SAR", context)

    assert "demo" not in safe.lower()
    assert context.guard_failures
