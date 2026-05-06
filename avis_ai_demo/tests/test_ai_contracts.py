import ast
from pathlib import Path

from avis_ai_demo.core.entity_extractor import extract_with_gpt_or_fallback
from avis_ai_demo.core.schemas import (
    SchemaValidationError,
    validate_conversation_decision,
    validate_conversation_classification,
    validate_gpt_extraction,
)
from avis_ai_demo.services.openai_service import OpenAIService


class StubClient:
    def __init__(self, payload):
        self.payload = payload

    def extract(self, message, state=None):
        return self.payload


def test_mocked_gpt_valid_json_is_used():
    service = OpenAIService(
        client=StubClient(
            {
                "intent": "daily_rental",
                "language": "en",
                "entities": {"vehicle_query": "Yaris"},
                "ambiguities": [],
                "clarification_question": None,
                "confidence": 0.9,
            }
        )
    )

    result, used_fallback, error = extract_with_gpt_or_fallback("I need a Yaris", service=service)

    assert used_fallback is False
    assert error is None
    assert result["intent"] == "daily_rental"
    assert result["entities"]["vehicle_query"] == "Yaris"


def test_mocked_gpt_invalid_json_falls_back_to_deterministic_clarification():
    service = OpenAIService(client=StubClient("not json"))

    result, used_fallback, error = extract_with_gpt_or_fallback("أبغى يارس", service=service)

    assert used_fallback is True
    assert error
    assert result["entities"]["vehicle_query"] == "Yaris"
    assert result["clarification_question"]


def test_gpt_supplied_calculated_totals_are_rejected():
    unsafe = {
        "intent": "daily_rental",
        "language": "en",
        "entities": {"vehicle_query": "Yaris", "total_online_sar": 999},
        "ambiguities": [],
        "clarification_question": None,
        "confidence": 0.9,
    }

    try:
        validate_gpt_extraction(unsafe)
    except SchemaValidationError as exc:
        assert "calculated monetary fields" in str(exc)
    else:
        raise AssertionError("Unsafe GPT totals were accepted.")


def test_conversation_classification_schema_accepts_only_control_fields():
    validate_conversation_classification(
        {
            "intent": "small_talk",
            "tone_mode": "friendly_casual",
            "customer_mood": "casual",
            "next_action": "scope_redirect",
            "confidence": 0.9,
        }
    )

    try:
        validate_conversation_classification(
            {
                "intent": "small_talk",
                "tone_mode": "friendly_casual",
                "customer_mood": "casual",
                "next_action": "scope_redirect",
                "confidence": 0.9,
                "answer": "hello",
            }
        )
    except SchemaValidationError as exc:
        assert "Unexpected" in str(exc)
    else:
        raise AssertionError("Conversation classifier accepted response text.")


def test_conversation_decision_schema_accepts_only_decision_fields():
    validate_conversation_decision(
        {
            "interaction_type": "social_with_service_hint",
            "service_domain": "offers",
            "workflow_candidate": "general_faq",
            "workflow_readiness": "not_applicable",
            "confidence": 0.88,
            "customer_mood": "playful_positive",
            "needs_clarification": False,
            "suggested_dialogue_act": "acknowledge_then_answer",
        }
    )

    try:
        validate_conversation_decision(
            {
                "interaction_type": "workflow_ready",
                "service_domain": "pricing",
                "workflow_candidate": "daily_rental",
                "workflow_readiness": "ready",
                "confidence": 0.9,
                "customer_mood": "task_focused",
                "needs_clarification": False,
                "suggested_dialogue_act": "route_operational",
                "answer": "I can book it.",
            }
        )
    except SchemaValidationError as exc:
        assert "Unexpected" in str(exc)
    else:
        raise AssertionError("Conversation decision accepted response text.")


def test_openai_key_loaded_from_environment_only(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    service = OpenAIService()

    assert service.api_key == "test-key"


def test_no_hardcoded_openai_key_assignment():
    root = Path(__file__).resolve().parents[2]
    for path in [root / "avis_ai_demo" / "services" / "openai_service.py"]:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                source = ast.get_source_segment(path.read_text(encoding="utf-8"), node) or ""
                assert "sk-" not in source
