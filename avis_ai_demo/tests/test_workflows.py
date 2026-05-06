import pytest

from avis_ai_demo.core.workflow_state import (
    advance_daily_workflow,
    advance_roadside_workflow,
    apply_payment_confirmation,
    complete_booking_after_payment,
    mark_quote_presented,
    transition_daily_phase,
)
from avis_ai_demo.core.types import DailyRentalPhase, RoadsidePhase, WorkflowState


FULL_DAILY_ENTITIES = {
    "customer_name": "Haitham",
    "age": 30,
    "has_valid_license": True,
    "pickup_city": "Riyadh",
    "dropoff_city": "Dammam",
    "rental_days": 2,
    "pickup_date": "2026-05-10",
    "pickup_time": "14:00",
    "vehicle_query": "Yaris",
}


def test_daily_quote_generation_without_payment_confirmation():
    state = advance_daily_workflow(WorkflowState(), FULL_DAILY_ENTITIES)

    assert state.phase == DailyRentalPhase.QUOTE_READY.value
    assert state.quote is not None
    assert state.payment_confirmed is False
    assert state.quote.values["total_online_sar"] == 684.7


def test_valid_and_invalid_daily_phase_transitions():
    state = WorkflowState(phase=DailyRentalPhase.INTAKE.value)
    transition_daily_phase(state, DailyRentalPhase.ELIGIBILITY_CHECK)
    assert state.phase == DailyRentalPhase.ELIGIBILITY_CHECK.value

    with pytest.raises(ValueError):
        transition_daily_phase(state, DailyRentalPhase.PAYMENT_PROCESSED)


def test_below_21_rejects_and_stops():
    entities = {**FULL_DAILY_ENTITIES, "age": 20}
    state = advance_daily_workflow(WorkflowState(), entities)

    assert state.phase == DailyRentalPhase.STOPPED_REJECTED.value
    assert state.rejected_reason == "age_below_21"


def test_missing_license_clarifies_not_rejects():
    entities = {key: value for key, value in FULL_DAILY_ENTITIES.items() if key != "has_valid_license"}
    state = advance_daily_workflow(WorkflowState(), entities)

    assert state.phase == DailyRentalPhase.ELIGIBILITY_CHECK.value
    assert state.next_action == "clarify_license"
    assert state.rejected_reason is None


def test_explicit_invalid_license_rejects():
    entities = {**FULL_DAILY_ENTITIES, "has_valid_license": False}
    state = advance_daily_workflow(WorkflowState(), entities)

    assert state.phase == DailyRentalPhase.STOPPED_REJECTED.value
    assert state.rejected_reason == "invalid_or_missing_license"


def test_quote_before_payment_and_success_after_confirmation():
    state = advance_daily_workflow(WorkflowState(), FULL_DAILY_ENTITIES)
    state = mark_quote_presented(state)

    assert state.phase == DailyRentalPhase.AWAITING_PAYMENT_CONFIRMATION.value
    assert state.payment_confirmed is False

    state = apply_payment_confirmation(state, "yes")
    assert state.phase == DailyRentalPhase.PAYMENT_PROCESSED.value
    assert state.payment_confirmed is True

    state = complete_booking_after_payment(state)
    assert state.phase == DailyRentalPhase.BOOKING_COMPLETED.value


def test_short_yes_does_not_confirm_before_awaiting_payment_phase():
    state = advance_daily_workflow(WorkflowState(), FULL_DAILY_ENTITIES)
    state = apply_payment_confirmation(state, "yes")

    assert state.phase == DailyRentalPhase.QUOTE_READY.value
    assert state.payment_confirmed is False


def test_roadside_workflow_without_optional_notes_reaches_case_ready():
    state = WorkflowState(phase=RoadsidePhase.SAFETY_CHECK.value)
    state = advance_roadside_workflow(
        state,
        {
            "safety_status": "safe",
            "mobile": "0550000000",
            "plate_or_contract": "ABC123",
            "current_location": "Riyadh",
            "issue_type": "battery",
            "drivable_status": "not_drivable",
        },
    )

    assert state.phase == RoadsidePhase.CASE_READY.value
    assert state.missing_fields == []
    assert "notes" not in state.missing_fields

