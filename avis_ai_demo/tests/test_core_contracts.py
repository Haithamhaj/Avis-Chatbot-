import ast
from pathlib import Path

import pytest

from avis_ai_demo.core.required_fields import (
    LicenseDecision,
    REQUIRED_FIELDS_BY_INTENT,
    evaluate_license_status,
    missing_fields_for_phase,
)
from avis_ai_demo.core.schemas import SchemaValidationError, validate_gpt_extraction
from avis_ai_demo.core.types import DailyRentalPhase, RoadsidePhase


def test_gpt_extraction_schema_accepts_valid_contract():
    validate_gpt_extraction(
        {
            "intent": "daily_rental",
            "language": "ar",
            "entities": {"vehicle_query": "يارس", "age": 30},
            "ambiguities": [],
            "clarification_question": None,
            "confidence": 0.8,
        }
    )


def test_gpt_extraction_schema_rejects_extra_and_invalid_fields():
    with pytest.raises(SchemaValidationError):
        validate_gpt_extraction(
            {
                "intent": "daily_rental",
                "language": "ar",
                "entities": {},
                "ambiguities": [],
                "clarification_question": None,
                "confidence": 0.8,
                "calculated_total": 999,
            }
        )


def test_phase_enums_match_required_values():
    assert {phase.value for phase in DailyRentalPhase} == {
        "intake",
        "eligibility_check",
        "quote_ready",
        "quote_presented",
        "awaiting_payment_confirmation",
        "payment_processed",
        "booking_completed",
        "stopped_rejected",
    }
    assert {phase.value for phase in RoadsidePhase} == {
        "safety_check",
        "intake",
        "case_ready",
        "case_recorded",
        "escalated",
    }


def test_daily_rental_required_fields_are_phase_based():
    contract = REQUIRED_FIELDS_BY_INTENT["daily_rental"]
    assert "payment_confirmation" not in contract.quote_required_fields
    assert "payment_confirmation" in contract.payment_required_fields
    assert missing_fields_for_phase(
        "daily_rental",
        "quote",
        {
            "customer_name": "H",
            "age": 30,
            "has_valid_license": True,
            "pickup_city": "Riyadh",
            "dropoff_city": "Dammam",
            "rental_days": 2,
            "pickup_date": "2026-05-10",
            "pickup_time": "14:00",
            "vehicle_query": "Yaris",
        },
    ) == []


def test_roadside_required_and_optional_fields():
    contract = REQUIRED_FIELDS_BY_INTENT["roadside_assistance"]
    assert contract.required_fields == (
        "safety_status",
        "mobile",
        "plate_or_contract",
        "current_location",
        "issue_type",
        "drivable_status",
    )
    assert contract.optional_fields == ("notes",)


def test_unknown_license_clarifies_only_explicit_invalid_rejects():
    assert evaluate_license_status(None) == LicenseDecision.CLARIFY
    assert evaluate_license_status("maybe") == LicenseDecision.CLARIFY
    assert evaluate_license_status("نعم عندي رخصة سارية") == LicenseDecision.ACCEPT
    assert evaluate_license_status("ما عندي رخصة") == LicenseDecision.REJECT


def test_core_modules_do_not_import_streamlit():
    core_dir = Path(__file__).resolve().parents[1] / "core"
    for path in core_dir.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                assert all(alias.name != "streamlit" for alias in node.names), path
            if isinstance(node, ast.ImportFrom):
                assert node.module != "streamlit", path

