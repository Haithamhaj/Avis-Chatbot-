from __future__ import annotations

from datetime import datetime
from itertools import count
from typing import Any

from avis_ai_demo.core.required_fields import REQUIRED_FIELDS_BY_INTENT
from avis_ai_demo.core.types import MockServiceResult, QuoteResult


_COUNTER = count(1)


def _reference(prefix: str) -> str:
    year = datetime.now().year
    return f"{prefix}-{year}-{next(_COUNTER):03d}"


def create_booking_request(quote: QuoteResult | None, customer: dict[str, Any]) -> MockServiceResult:
    if quote is None or not quote.values.get("totals_produced"):
        raise ValueError("booking request requires calculator quote output")
    if not customer.get("customer_name"):
        raise ValueError("booking request requires customer_name")
    return MockServiceResult(
        action="create_booking_request",
        reference_id=_reference("AVIS-REQ"),
        status="recorded",
        payload={
            "customer_name": customer["customer_name"],
            "quote": quote.values,
            "customer_message_ar": "تم تسجيل طلب الحجز بناءً على البيانات التي زودتنا بها.",
            "customer_message_en": "Your booking request has been recorded based on the details you provided.",
        },
    )


def process_payment(quote: QuoteResult | None, confirmed: bool) -> MockServiceResult:
    if quote is None or not quote.values.get("totals_produced"):
        raise ValueError("payment requires calculator quote output")
    if not confirmed:
        raise ValueError("payment requires explicit confirmation")
    return MockServiceResult(
        action="process_payment",
        reference_id=_reference("AVIS-PAY"),
        status="processed",
        payload={
            "paid_amount_online_sar": quote.values.get("total_online_sar")
            or quote.values.get("total_inc_vat_sar"),
            "customer_message_ar": "تم الدفع بنجاح.",
            "customer_message_en": "Payment completed successfully.",
        },
    )


def create_roadside_case(fields: dict[str, Any]) -> MockServiceResult:
    required = REQUIRED_FIELDS_BY_INTENT["roadside_assistance"].required_fields
    missing = [field for field in required if not fields.get(field)]
    if missing:
        raise ValueError(f"roadside case missing required fields: {missing}")
    return MockServiceResult(
        action="create_roadside_case",
        reference_id=_reference("AVIS-SOS"),
        status="recorded",
        payload={
            "issue_type": fields["issue_type"],
            "customer_message_ar": "تم تسجيل طلب المساعدة وسيتم التواصل معك من الفريق المختص.",
            "customer_message_en": "Your assistance request has been recorded and the relevant team will contact you.",
        },
    )


def handover_to_agent(case_type: str, fields: dict[str, Any]) -> MockServiceResult:
    if not case_type:
        raise ValueError("handover requires case_type")
    return MockServiceResult(
        action="handover_to_agent",
        reference_id=_reference("AVIS-CASE"),
        status="recorded",
        payload={
            "case_type": case_type,
            "fields": fields,
            "customer_message_ar": "تم تحويل طلبك للفريق المختص.",
            "customer_message_en": "Your request has been handed over to the relevant team.",
        },
    )

