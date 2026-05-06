from __future__ import annotations

import json
from typing import Any

from avis_ai_demo.core.types import Intent


CONVERSATION_INTENTS = [
    "greeting",
    "small_talk",
    "off_topic",
    "thanks_acknowledgement",
    "capability_question",
    "potentially_relevant_unclear",
    "operational_request",
    "complaint_or_dispute",
    "roadside_or_accident",
    "fallback_unknown",
]

TONE_MODES = [
    "friendly_casual",
    "professional_helpful",
    "serious_supportive",
    "safety_first",
    "light_deflection",
]

INTERACTION_TYPES = [
    "pure_social",
    "social_with_service_hint",
    "faq_or_info",
    "workflow_candidate",
    "workflow_ready",
    "hard_safety",
    "hard_financial_dispute",
    "off_topic",
    "unclear",
]

SERVICE_DOMAINS = [
    "none",
    "booking",
    "pricing",
    "offers",
    "branches",
    "requirements",
    "deposit_policy",
    "roadside",
    "complaint",
    "company_info",
    "fleet",
]

WORKFLOW_READINESS = ["not_applicable", "not_ready", "ready"]

CONVERSATION_DECISION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "interaction_type",
        "service_domain",
        "workflow_candidate",
        "workflow_readiness",
        "confidence",
        "customer_mood",
        "needs_clarification",
        "suggested_dialogue_act",
    ],
    "properties": {
        "interaction_type": {"type": "string", "enum": INTERACTION_TYPES},
        "service_domain": {"type": "string", "enum": SERVICE_DOMAINS},
        "workflow_candidate": {"type": ["string", "null"], "enum": [None, *[intent.value for intent in Intent]]},
        "workflow_readiness": {"type": "string", "enum": WORKFLOW_READINESS},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "customer_mood": {"type": "string"},
        "needs_clarification": {"type": "boolean"},
        "suggested_dialogue_act": {"type": "string"},
    },
}

CONVERSATION_CLASSIFICATION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["intent", "tone_mode", "customer_mood", "next_action", "confidence"],
    "properties": {
        "intent": {"type": "string", "enum": CONVERSATION_INTENTS},
        "tone_mode": {"type": "string", "enum": TONE_MODES},
        "customer_mood": {"type": "string"},
        "next_action": {"type": "string"},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
    },
}


GPT_EXTRACTION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "intent",
        "language",
        "entities",
        "ambiguities",
        "clarification_question",
        "confidence",
    ],
    "properties": {
        "intent": {"type": "string", "enum": [intent.value for intent in Intent]},
        "language": {"type": "string", "enum": ["ar", "en"]},
        "entities": {
            "type": "object",
            "additionalProperties": True,
            "properties": {
                "customer_name": {"type": ["string", "null"]},
                "age": {"type": ["integer", "null"]},
                "has_valid_license": {"type": ["boolean", "null"]},
                "pickup_city": {"type": ["string", "null"]},
                "pickup_branch": {"type": ["string", "null"]},
                "dropoff_city": {"type": ["string", "null"]},
                "dropoff_branch": {"type": ["string", "null"]},
                "rental_days": {"type": ["integer", "null"]},
                "pickup_date": {"type": ["string", "null"]},
                "pickup_time": {"type": ["string", "null"]},
                "vehicle_query": {"type": ["string", "null"]},
                "mobile": {"type": ["string", "null"]},
                "plate_or_contract": {"type": ["string", "null"]},
                "current_location": {"type": ["string", "null"]},
                "issue_type": {"type": ["string", "null"]},
                "drivable_status": {"type": ["string", "null"]},
                "safety_status": {"type": ["string", "null"]},
                "notes": {"type": ["string", "null"]},
            },
        },
        "ambiguities": {"type": "array", "items": {"type": "string"}},
        "clarification_question": {"type": ["string", "null"]},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
    },
}


class SchemaValidationError(ValueError):
    pass


def parse_conversation_classification(raw: str | dict[str, Any]) -> dict[str, Any]:
    data = json.loads(raw) if isinstance(raw, str) else raw
    validate_conversation_classification(data)
    return data


def validate_conversation_classification(data: dict[str, Any]) -> None:
    if not isinstance(data, dict):
        raise SchemaValidationError("Conversation classification must be a JSON object.")
    required = set(CONVERSATION_CLASSIFICATION_SCHEMA["required"])
    missing = required - set(data)
    if missing:
        raise SchemaValidationError(f"Missing conversation classification fields: {sorted(missing)}")
    extra = set(data) - set(CONVERSATION_CLASSIFICATION_SCHEMA["properties"])
    if extra:
        raise SchemaValidationError(f"Unexpected conversation classification fields: {sorted(extra)}")
    if data["intent"] not in CONVERSATION_INTENTS:
        raise SchemaValidationError("Invalid conversation intent.")
    if data["tone_mode"] not in TONE_MODES:
        raise SchemaValidationError("Invalid tone mode.")
    if not isinstance(data["customer_mood"], str) or not data["customer_mood"]:
        raise SchemaValidationError("customer_mood must be a non-empty string.")
    if not isinstance(data["next_action"], str) or not data["next_action"]:
        raise SchemaValidationError("next_action must be a non-empty string.")
    if not isinstance(data["confidence"], (int, float)) or not 0 <= data["confidence"] <= 1:
        raise SchemaValidationError("confidence must be between 0 and 1.")


def parse_conversation_decision(raw: str | dict[str, Any]) -> dict[str, Any]:
    data = json.loads(raw) if isinstance(raw, str) else raw
    validate_conversation_decision(data)
    return data


def validate_conversation_decision(data: dict[str, Any]) -> None:
    if not isinstance(data, dict):
        raise SchemaValidationError("Conversation decision must be a JSON object.")
    required = set(CONVERSATION_DECISION_SCHEMA["required"])
    missing = required - set(data)
    if missing:
        raise SchemaValidationError(f"Missing conversation decision fields: {sorted(missing)}")
    extra = set(data) - set(CONVERSATION_DECISION_SCHEMA["properties"])
    if extra:
        raise SchemaValidationError(f"Unexpected conversation decision fields: {sorted(extra)}")
    if data["interaction_type"] not in INTERACTION_TYPES:
        raise SchemaValidationError("Invalid interaction_type.")
    if data["service_domain"] not in SERVICE_DOMAINS:
        raise SchemaValidationError("Invalid service_domain.")
    candidate = data["workflow_candidate"]
    if candidate is not None and candidate not in {intent.value for intent in Intent}:
        raise SchemaValidationError("Invalid workflow_candidate.")
    if data["workflow_readiness"] not in WORKFLOW_READINESS:
        raise SchemaValidationError("Invalid workflow_readiness.")
    if not isinstance(data["confidence"], (int, float)) or not 0 <= data["confidence"] <= 1:
        raise SchemaValidationError("confidence must be between 0 and 1.")
    if not isinstance(data["customer_mood"], str) or not data["customer_mood"]:
        raise SchemaValidationError("customer_mood must be a non-empty string.")
    if not isinstance(data["needs_clarification"], bool):
        raise SchemaValidationError("needs_clarification must be boolean.")
    if not isinstance(data["suggested_dialogue_act"], str) or not data["suggested_dialogue_act"]:
        raise SchemaValidationError("suggested_dialogue_act must be a non-empty string.")


def parse_gpt_extraction(raw: str | dict[str, Any]) -> dict[str, Any]:
    data = json.loads(raw) if isinstance(raw, str) else raw
    validate_gpt_extraction(data)
    return data


def validate_gpt_extraction(data: dict[str, Any]) -> None:
    if not isinstance(data, dict):
        raise SchemaValidationError("GPT extraction must be a JSON object.")

    required = set(GPT_EXTRACTION_SCHEMA["required"])
    missing = required - set(data)
    if missing:
        raise SchemaValidationError(f"Missing GPT extraction fields: {sorted(missing)}")

    extra = set(data) - set(GPT_EXTRACTION_SCHEMA["properties"])
    if extra:
        raise SchemaValidationError(f"Unexpected GPT extraction fields: {sorted(extra)}")

    if data["intent"] not in {intent.value for intent in Intent}:
        raise SchemaValidationError("Invalid intent.")
    if data["language"] not in {"ar", "en"}:
        raise SchemaValidationError("Invalid language.")
    if not isinstance(data["entities"], dict):
        raise SchemaValidationError("entities must be an object.")
    forbidden_entity_keys = {
        "total",
        "totals",
        "calculated_total",
        "total_online_sar",
        "total_in_branch_sar",
        "online_subtotal_sar",
        "in_branch_subtotal_sar",
        "one_way_dropoff_fee_sar",
    }
    leaked_keys = forbidden_entity_keys & set(data["entities"])
    if leaked_keys:
        raise SchemaValidationError(
            f"GPT extraction cannot provide calculated monetary fields: {sorted(leaked_keys)}"
        )
    if not isinstance(data["ambiguities"], list) or not all(
        isinstance(item, str) for item in data["ambiguities"]
    ):
        raise SchemaValidationError("ambiguities must be a string array.")
    if data["clarification_question"] is not None and not isinstance(
        data["clarification_question"], str
    ):
        raise SchemaValidationError("clarification_question must be string or null.")
    if not isinstance(data["confidence"], (int, float)) or not 0 <= data["confidence"] <= 1:
        raise SchemaValidationError("confidence must be between 0 and 1.")
