from avis_ai_demo.core.conversation_decision import decide_conversation
from avis_ai_demo.core.orchestrator import handle_message
from avis_ai_demo.core.types import Intent
from avis_ai_demo.core.workflow_gate import evaluate_workflow_gate
from avis_ai_demo.services.openai_service import OpenAIService


class StubClient:
    def __init__(self, decision=None, extraction_intent="fallback_unknown"):
        self.decision = decision
        self.payload = {
            "intent": extraction_intent,
            "language": "ar",
            "entities": {},
            "ambiguities": [],
            "clarification_question": None,
            "confidence": 0.6,
        }

    def decide_conversation(self, message, state=None):
        if self.decision is None:
            raise RuntimeError("no decision")
        return self.decision

    def classify_conversation(self, message, state=None):
        raise RuntimeError("no classifier")

    def extract(self, message, state=None):
        return self.payload

    def phrase_khalid_response(self, plan):
        return plan["deterministic_fallback"]


def service(decision=None, extraction_intent="fallback_unknown"):
    return OpenAIService(client=StubClient(decision, extraction_intent))


def test_decision_layer_routes_discount_as_kb_not_dispute():
    response, _state, trace = handle_message(
        None,
        "شوف انا بصراحة بحب ايفيس ولازم تعطوني خصم مقابل هذا الحب هههههه",
        service(),
    )

    assert trace["conversation_decision"]["interaction_type"] == "social_with_service_hint"
    assert trace["workflow_gate"]["reason"] == "service_hint_kb_lookup"
    assert trace["intent"] == "general_faq"
    assert trace["escalation_status"] is False
    assert "رقم الحجز أو العقد" not in response
    assert "وصلت المحبة" in response


def test_workflow_gate_blocks_not_ready_workflow_candidate():
    decision = {
        "interaction_type": "workflow_candidate",
        "service_domain": "booking",
        "workflow_candidate": "daily_rental",
        "workflow_readiness": "not_ready",
        "confidence": 0.83,
        "customer_mood": "task_focused",
        "needs_clarification": True,
        "suggested_dialogue_act": "ask_clarification",
    }
    response, _state, trace = handle_message(None, "ودي أرتب سيارة", service(decision, "daily_rental"))

    assert trace["conversation_decision"]["interaction_type"] == "workflow_candidate"
    assert trace["workflow_gate"]["allow_operational"] is False
    assert trace["intent"] == "potentially_relevant_unclear"
    assert "تقصد حجز سيارة" in response
    assert trace["computed_totals"] is None


def test_workflow_gate_allows_hard_financial_dispute_only_when_explicit():
    decision = decide_conversation("انخصم مني مبلغ")
    gate = evaluate_workflow_gate(decision)

    assert decision.interaction_type == "hard_financial_dispute"
    assert gate.allow_operational is True
    assert gate.intent == Intent.COMPLAINT_OR_DISPUTE


def test_workflow_gate_allows_ready_workflow_candidate():
    decision = {
        "interaction_type": "workflow_ready",
        "service_domain": "booking",
        "workflow_candidate": "daily_rental",
        "workflow_readiness": "ready",
        "confidence": 0.9,
        "customer_mood": "task_focused",
        "needs_clarification": False,
        "suggested_dialogue_act": "route_operational",
    }
    parsed = decide_conversation("أبغى يارس من الرياض يومين", service(decision))
    gate = evaluate_workflow_gate(parsed)

    assert gate.allow_operational is True
    assert gate.intent == Intent.DAILY_RENTAL


def test_deterministic_decision_marks_complete_booking_as_ready():
    decision = decide_conversation("أبغى يارس من الرياض للدمام يومين، الاستلام 2026-05-10 الساعة 14:00")
    gate = evaluate_workflow_gate(decision)

    assert decision.interaction_type == "workflow_ready"
    assert decision.workflow_candidate == Intent.DAILY_RENTAL
    assert gate.allow_operational is True
