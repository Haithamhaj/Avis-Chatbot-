from __future__ import annotations

from dataclasses import replace
from typing import Any

from avis_ai_demo.core.calculators import CalculationError, calculate_daily_quote
from avis_ai_demo.core.confirmation import is_confirmation
from avis_ai_demo.core.data_lookup import find_daily_price
from avis_ai_demo.core.required_fields import (
    LicenseDecision,
    REQUIRED_FIELDS_BY_INTENT,
    evaluate_license_status,
    missing_fields_for_phase,
)
from avis_ai_demo.core.types import DailyRentalPhase, Intent, RoadsidePhase, WorkflowState


VALID_DAILY_TRANSITIONS: dict[str, set[str]] = {
    DailyRentalPhase.INTAKE.value: {
        DailyRentalPhase.ELIGIBILITY_CHECK.value,
        DailyRentalPhase.QUOTE_READY.value,
        DailyRentalPhase.STOPPED_REJECTED.value,
    },
    DailyRentalPhase.ELIGIBILITY_CHECK.value: {
        DailyRentalPhase.INTAKE.value,
        DailyRentalPhase.QUOTE_READY.value,
        DailyRentalPhase.STOPPED_REJECTED.value,
    },
    DailyRentalPhase.QUOTE_READY.value: {DailyRentalPhase.QUOTE_PRESENTED.value},
    DailyRentalPhase.QUOTE_PRESENTED.value: {DailyRentalPhase.AWAITING_PAYMENT_CONFIRMATION.value},
    DailyRentalPhase.AWAITING_PAYMENT_CONFIRMATION.value: {DailyRentalPhase.PAYMENT_PROCESSED.value},
    DailyRentalPhase.PAYMENT_PROCESSED.value: {DailyRentalPhase.BOOKING_COMPLETED.value},
    DailyRentalPhase.BOOKING_COMPLETED.value: set(),
    DailyRentalPhase.STOPPED_REJECTED.value: set(),
}


VALID_ROADSIDE_TRANSITIONS: dict[str, set[str]] = {
    RoadsidePhase.SAFETY_CHECK.value: {RoadsidePhase.INTAKE.value, RoadsidePhase.ESCALATED.value},
    RoadsidePhase.INTAKE.value: {RoadsidePhase.CASE_READY.value, RoadsidePhase.ESCALATED.value},
    RoadsidePhase.CASE_READY.value: {RoadsidePhase.CASE_RECORDED.value, RoadsidePhase.ESCALATED.value},
    RoadsidePhase.CASE_RECORDED.value: set(),
    RoadsidePhase.ESCALATED.value: set(),
}


def transition_daily_phase(state: WorkflowState, next_phase: DailyRentalPhase) -> WorkflowState:
    if next_phase.value not in VALID_DAILY_TRANSITIONS.get(state.phase, set()):
        raise ValueError(f"Invalid daily phase transition: {state.phase} -> {next_phase.value}")
    state.phase = next_phase.value
    return state


def transition_roadside_phase(state: WorkflowState, next_phase: RoadsidePhase) -> WorkflowState:
    if next_phase.value not in VALID_ROADSIDE_TRANSITIONS.get(state.phase, set()):
        raise ValueError(f"Invalid roadside phase transition: {state.phase} -> {next_phase.value}")
    state.phase = next_phase.value
    return state


def merge_entities(state: WorkflowState, entities: dict[str, Any]) -> WorkflowState:
    state.entities.update({key: value for key, value in entities.items() if value not in (None, "")})
    return state


def advance_daily_workflow(state: WorkflowState, entities: dict[str, Any]) -> WorkflowState:
    state.intent = Intent.DAILY_RENTAL
    if state.phase not in VALID_DAILY_TRANSITIONS:
        state.phase = DailyRentalPhase.INTAKE.value
    merge_entities(state, entities)

    age = state.entities.get("age")
    if age is not None and int(age) < 21:
        state.phase = DailyRentalPhase.STOPPED_REJECTED.value
        state.rejected_reason = "age_below_21"
        state.next_action = "stop_rejected"
        return state

    license_decision = evaluate_license_status(state.entities.get("has_valid_license"))
    if license_decision == LicenseDecision.REJECT:
        state.phase = DailyRentalPhase.STOPPED_REJECTED.value
        state.rejected_reason = "invalid_or_missing_license"
        state.next_action = "stop_rejected"
        return state
    if license_decision == LicenseDecision.CLARIFY:
        state.phase = DailyRentalPhase.ELIGIBILITY_CHECK.value
        state.missing_fields = ["has_valid_license"]
        state.next_action = "clarify_license"
        return state

    missing = missing_fields_for_phase("daily_rental", "quote", state.entities)
    if missing:
        state.phase = DailyRentalPhase.INTAKE.value
        state.missing_fields = missing
        state.next_action = "collect_quote_fields"
        return state

    price = find_daily_price(str(state.entities["vehicle_query"]))
    if price is None:
        state.phase = DailyRentalPhase.INTAKE.value
        state.missing_fields = ["vehicle_query"]
        state.next_action = "clarify_vehicle"
        return state

    try:
        state.quote = calculate_daily_quote(
            price,
            rental_days=int(state.entities["rental_days"]),
            pickup_city=str(state.entities["pickup_city"]),
            dropoff_city=str(state.entities["dropoff_city"]),
        )
    except CalculationError as exc:
        state.phase = DailyRentalPhase.INTAKE.value
        state.missing_fields = ["dropoff_city"]
        state.next_action = "clarify_route"
        state.entities["calculation_error"] = str(exc)
        return state

    state.phase = DailyRentalPhase.QUOTE_READY.value
    state.missing_fields = []
    state.next_action = "present_quote"
    return state


def mark_quote_presented(state: WorkflowState) -> WorkflowState:
    if state.phase == DailyRentalPhase.QUOTE_READY.value:
        transition_daily_phase(state, DailyRentalPhase.QUOTE_PRESENTED)
    if state.phase == DailyRentalPhase.QUOTE_PRESENTED.value:
        transition_daily_phase(state, DailyRentalPhase.AWAITING_PAYMENT_CONFIRMATION)
    state.next_action = "await_payment_confirmation"
    return state


def apply_payment_confirmation(state: WorkflowState, message: str) -> WorkflowState:
    if is_confirmation(message, state.phase) and state.quote is not None:
        transition_daily_phase(state, DailyRentalPhase.PAYMENT_PROCESSED)
        state.payment_confirmed = True
        state.next_action = "complete_booking"
    return state


def complete_booking_after_payment(state: WorkflowState) -> WorkflowState:
    if state.phase == DailyRentalPhase.PAYMENT_PROCESSED.value and state.payment_confirmed:
        transition_daily_phase(state, DailyRentalPhase.BOOKING_COMPLETED)
        state.next_action = "booking_completed"
    return state


def advance_roadside_workflow(state: WorkflowState, entities: dict[str, Any]) -> WorkflowState:
    state.intent = Intent.ROADSIDE_ASSISTANCE
    if state.phase not in VALID_ROADSIDE_TRANSITIONS:
        state.phase = RoadsidePhase.SAFETY_CHECK.value
    merge_entities(state, entities)

    if state.entities.get("safety_status") in {"injury", "danger", "إصابة", "خطر"}:
        state.phase = RoadsidePhase.ESCALATED.value
        state.risk_level = "high"
        state.next_action = "escalate_emergency"
        return state

    if state.phase == RoadsidePhase.SAFETY_CHECK.value and state.entities.get("safety_status"):
        transition_roadside_phase(state, RoadsidePhase.INTAKE)

    required = REQUIRED_FIELDS_BY_INTENT["roadside_assistance"].required_fields
    missing = [field for field in required if not state.entities.get(field)]
    state.missing_fields = missing
    if missing:
        state.next_action = "collect_roadside_fields"
        return state

    if state.phase == RoadsidePhase.INTAKE.value:
        transition_roadside_phase(state, RoadsidePhase.CASE_READY)
    state.next_action = "record_roadside_case"
    return state

