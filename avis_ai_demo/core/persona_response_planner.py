from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from avis_ai_demo.core.tone_lexicon import candidate_tone_phrases
from avis_ai_demo.core.types import AnswerContext


@dataclass
class PersonaResponsePlan:
    intent: str
    tone_mode: str
    customer_mood: str
    language: str
    phase: str | None
    collected_entities: dict[str, Any] = field(default_factory=dict)
    missing_fields: list[str] = field(default_factory=list)
    allowed_facts: dict[str, Any] = field(default_factory=dict)
    lookup_records: list[dict[str, Any]] = field(default_factory=list)
    computed_totals: dict[str, Any] | None = None
    escalation_status: bool = False
    next_action: str = "clarify"
    service_suggestions: list[str] = field(default_factory=list)
    forbidden_claims: list[str] = field(default_factory=list)
    tone_phrases: list[str] = field(default_factory=list)
    deterministic_fallback: str = ""

    def to_payload(self) -> dict[str, Any]:
        return asdict(self)


DEFAULT_FORBIDDEN_CLAIMS = [
    "Do not invent prices or totals.",
    "Do not calculate totals.",
    "Do not invent availability.",
    "Do not invent payment status.",
    "Do not invent booking status.",
    "Do not invent branch data.",
    "Do not invent policy rules.",
    "Do not expose KB IDs, trace, JSON, or workflow names.",
]


def build_persona_response_plan(
    context: AnswerContext,
    records: dict[str, Any] | None = None,
    deterministic_fallback: str = "",
) -> PersonaResponsePlan:
    records = records or {}
    allowed_facts = dict(context.allowed_facts)
    if "branch" in records and records["branch"]:
        branch = records["branch"]
        allowed_facts["branch"] = {
            "name_ar": branch.get("branch_name_ar"),
            "name_en": branch.get("branch_name_en"),
            "hours": branch.get("hours"),
            "phone": branch.get("phone"),
            "map_url": branch.get("map_url"),
        }
    if "daily_price" in records and records["daily_price"]:
        price = records["daily_price"]
        allowed_facts["daily_price"] = {
            "classification": price.get("classification"),
            "online_total_vat_sar": price.get("online_total_vat_sar"),
            "in_branch_total_vat_sar": price.get("in_branch_total_vat_sar"),
            "models_2026": price.get("models_2026", []),
        }
    if "monthly_price" in records and records["monthly_price"]:
        price = records["monthly_price"]
        allowed_facts["monthly_price"] = {
            "classification": price.get("classification"),
            "online_price_sar": price.get("online_price_sar"),
            "total_inc_vat_sar": price.get("total_inc_vat_sar"),
            "model": price.get("model"),
        }
    if "general_faq" in records and records["general_faq"]:
        faq = records["general_faq"]
        allowed_facts["general_faq"] = {
            "topic": faq.get("topic"),
            "answer_ar": faq.get("answer_ar"),
            "answer_en": faq.get("answer_en"),
            "risk_level": faq.get("risk_level"),
        }

    service_suggestions = list(context.service_suggestions)
    if not service_suggestions and context.intent.value in {"small_talk", "greeting"}:
        service_suggestions = ["If the customer has a trip or rental in mind, invite them lightly to continue."]
    if context.intent.value == "fleet_pricing":
        service_suggestions = ["Offer to continue into a quotation if the customer shares city and duration."]
    if context.intent.value == "branch_lookup":
        service_suggestions = ["Connect branch details to pickup or return if helpful."]
    if context.escalation_required or context.intent.value in {"complaint_or_dispute", "complaint_or_financial_dispute"}:
        service_suggestions = ["Acknowledge briefly and collect the details needed for the relevant team."]
    if context.intent.value in {"roadside_assistance", "roadside_or_accident"}:
        service_suggestions = ["Put safety first before collecting roadside case details."]
    if context.intent.value in {"fallback_unknown", "potentially_relevant_unclear"}:
        service_suggestions = ["Ask one short clarification question before listing broad services."]
        if allowed_facts.get("clarification_turn") == "followup":
            service_suggestions = [
                "This is a follow-up after a clarification question. Do not repeat the same service-category options. Ask an open, empathetic question about what happened."
            ]
    if context.intent.value == "general_faq":
        service_suggestions = ["Answer from the FAQ fact, then invite the customer to ask for a booking, price, branch, or roadside help."]

    return PersonaResponsePlan(
        intent=context.intent.value,
        tone_mode=context.tone_mode,
        customer_mood=context.customer_mood,
        language=context.language,
        phase=context.phase,
        missing_fields=context.missing_fields,
        allowed_facts=allowed_facts,
        lookup_records=[
            {"kb_id": record.kb_id, "record_id": record.record_id}
            for record in context.retrieved_records
        ],
        computed_totals=context.computed_totals.values if context.computed_totals else None,
        escalation_status=context.escalation_required,
        next_action=context.next_action,
        service_suggestions=service_suggestions,
        forbidden_claims=list(context.forbidden_claims or DEFAULT_FORBIDDEN_CLAIMS),
        tone_phrases=candidate_tone_phrases(context),
        deterministic_fallback=deterministic_fallback,
    )
