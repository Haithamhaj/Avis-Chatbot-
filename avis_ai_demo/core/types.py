from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal


Language = Literal["ar", "en"]
RiskLevel = Literal["low", "medium", "high"]


class Intent(str, Enum):
    GREETING = "greeting"
    SMALL_TALK = "small_talk"
    OFF_TOPIC = "off_topic"
    THANKS_ACKNOWLEDGEMENT = "thanks_acknowledgement"
    CAPABILITY_QUESTION = "capability_question"
    GENERAL_FAQ = "general_faq"
    CLARIFICATION_REQUEST = "clarification_request"
    POTENTIALLY_RELEVANT_UNCLEAR = "potentially_relevant_unclear"
    OPERATIONAL_REQUEST = "operational_request"
    COMPLAINT_OR_DISPUTE = "complaint_or_dispute"
    ROADSIDE_OR_ACCIDENT = "roadside_or_accident"
    SCOPE_REDIRECT = "scope_redirect"
    DAILY_RENTAL = "daily_rental"
    MONTHLY_RENTAL = "monthly_rental"
    BRANCH_LOOKUP = "branch_lookup"
    FLEET_PRICING = "fleet_pricing"
    CARD_DEPOSIT_POLICY = "card_deposit_policy"
    ROADSIDE_ASSISTANCE = "roadside_assistance"
    COMPLAINT_OR_FINANCIAL_DISPUTE = "complaint_or_financial_dispute"
    FALLBACK_UNKNOWN = "fallback_unknown"


class DailyRentalPhase(str, Enum):
    INTAKE = "intake"
    ELIGIBILITY_CHECK = "eligibility_check"
    QUOTE_READY = "quote_ready"
    QUOTE_PRESENTED = "quote_presented"
    AWAITING_PAYMENT_CONFIRMATION = "awaiting_payment_confirmation"
    PAYMENT_PROCESSED = "payment_processed"
    BOOKING_COMPLETED = "booking_completed"
    STOPPED_REJECTED = "stopped_rejected"


class RoadsidePhase(str, Enum):
    SAFETY_CHECK = "safety_check"
    INTAKE = "intake"
    CASE_READY = "case_ready"
    CASE_RECORDED = "case_recorded"
    ESCALATED = "escalated"


@dataclass(frozen=True)
class RetrievedRecord:
    kb_id: str
    record_id: str
    confidence: float = 1.0


@dataclass
class QuoteResult:
    rental_type: Literal["daily", "monthly"]
    calculator_source: str
    values: dict[str, Any]


@dataclass
class MockServiceResult:
    action: str
    reference_id: str
    status: Literal["recorded", "processed", "failed"]
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass
class WorkflowState:
    intent: Intent = Intent.FALLBACK_UNKNOWN
    language: Language = "ar"
    phase: str = DailyRentalPhase.INTAKE.value
    entities: dict[str, Any] = field(default_factory=dict)
    missing_fields: list[str] = field(default_factory=list)
    quote: QuoteResult | None = None
    payment_confirmed: bool = False
    rejected_reason: str | None = None
    risk_level: RiskLevel = "low"
    next_action: str = "clarify"
    tone_mode: str = "professional_helpful"
    customer_mood: str = "neutral"
    guard_failures: list[str] = field(default_factory=list)
    conversation_turns: list[dict[str, Any]] = field(default_factory=list)
    conversation_summary: str = ""


@dataclass
class AnswerContext:
    intent: Intent
    language: Language
    retrieved_records: list[RetrievedRecord] = field(default_factory=list)
    missing_fields: list[str] = field(default_factory=list)
    risk_level: RiskLevel = "low"
    escalation_required: bool = False
    caveats: list[str] = field(default_factory=list)
    computed_totals: QuoteResult | None = None
    next_action: str = "clarify"
    phase: str | None = None
    tone_mode: str = "professional_helpful"
    user_message: str = ""
    customer_mood: str = "neutral"
    allowed_facts: dict[str, Any] = field(default_factory=dict)
    service_suggestions: list[str] = field(default_factory=list)
    forbidden_claims: list[str] = field(default_factory=list)
    guard_failures: list[str] = field(default_factory=list)
