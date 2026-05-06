from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Literal


@dataclass(frozen=True)
class PhasedRequiredFields:
    quote_required_fields: tuple[str, ...] = ()
    payment_required_fields: tuple[str, ...] = ()
    post_payment_fields: tuple[str, ...] = ()
    required_fields: tuple[str, ...] = ()
    optional_fields: tuple[str, ...] = ()


REQUIRED_FIELDS_BY_INTENT: dict[str, PhasedRequiredFields] = {
    "daily_rental": PhasedRequiredFields(
        quote_required_fields=(
            "has_valid_license",
            "pickup_city",
            "dropoff_city",
            "rental_days",
            "pickup_date",
            "pickup_time",
            "vehicle_query",
        ),
        payment_required_fields=("quote_reference", "payment_confirmation"),
        post_payment_fields=("booking_reference",),
    ),
    "monthly_rental": PhasedRequiredFields(
        quote_required_fields=("vehicle_query", "pickup_city", "pickup_date"),
        payment_required_fields=("quote_reference", "payment_confirmation"),
        post_payment_fields=("booking_reference",),
    ),
    "branch_lookup": PhasedRequiredFields(required_fields=("branch_or_city_query",)),
    "roadside_assistance": PhasedRequiredFields(
        required_fields=(
            "safety_status",
            "mobile",
            "plate_or_contract",
            "current_location",
            "issue_type",
            "drivable_status",
        ),
        optional_fields=("notes",),
    ),
    "complaint_or_financial_dispute": PhasedRequiredFields(
        required_fields=(
            "name",
            "mobile",
            "booking_or_ra_number",
            "issue_details",
        )
    ),
    "card_deposit_policy": PhasedRequiredFields(required_fields=("policy_query",)),
}


class LicenseDecision(str, Enum):
    CLARIFY = "clarify"
    ACCEPT = "accept"
    REJECT = "reject"


def evaluate_license_status(value: object) -> LicenseDecision:
    if value is None:
        return LicenseDecision.CLARIFY
    if value is True:
        return LicenseDecision.ACCEPT
    if value is False:
        return LicenseDecision.REJECT

    text = str(value).strip().lower()
    invalid_tokens = {
        "no",
        "no license",
        "invalid",
        "expired",
        "لا",
        "لا يوجد",
        "ما عندي",
        "منتهية",
        "غير سارية",
    }
    valid_tokens = {"yes", "valid", "i have", "نعم", "عندي", "سارية"}
    if any(token in text for token in invalid_tokens):
        return LicenseDecision.REJECT
    if any(token in text for token in valid_tokens):
        return LicenseDecision.ACCEPT
    return LicenseDecision.CLARIFY


def missing_fields_for_phase(intent: str, phase_name: Literal["quote", "payment", "post_payment"], entities: dict) -> list[str]:
    contract = REQUIRED_FIELDS_BY_INTENT[intent]
    if phase_name == "quote":
        fields = contract.quote_required_fields
    elif phase_name == "payment":
        fields = contract.payment_required_fields
    else:
        fields = contract.post_payment_fields
    return [field for field in fields if not entities.get(field)]
