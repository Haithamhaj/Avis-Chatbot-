from __future__ import annotations

from dataclasses import dataclass

from avis_ai_demo.core.conversation_decision import ConversationDecision
from avis_ai_demo.core.types import Intent


@dataclass(frozen=True)
class WorkflowGateResult:
    allow_operational: bool
    intent: Intent
    reason: str
    next_action: str

    def to_trace(self) -> dict[str, object]:
        return {
            "allow_operational": self.allow_operational,
            "intent": self.intent.value,
            "reason": self.reason,
            "next_action": self.next_action,
        }


def evaluate_workflow_gate(decision: ConversationDecision) -> WorkflowGateResult:
    if decision.interaction_type == "hard_safety":
        return WorkflowGateResult(True, Intent.ROADSIDE_OR_ACCIDENT, "hard_safety_override", "safety_check")
    if decision.interaction_type == "hard_financial_dispute":
        return WorkflowGateResult(True, Intent.COMPLAINT_OR_DISPUTE, "hard_financial_override", "collect_case_details")
    if decision.interaction_type == "service_feedback":
        return WorkflowGateResult(True, Intent.SERVICE_EXPERIENCE_FEEDBACK, "service_feedback", "collect_feedback_details")
    if decision.interaction_type in {"faq_or_info", "social_with_service_hint"} and decision.workflow_candidate in {Intent.GENERAL_FAQ, Intent.CARD_DEPOSIT_POLICY}:
        return WorkflowGateResult(True, decision.workflow_candidate, "service_hint_kb_lookup", "lookup_general_faq")
    if decision.interaction_type == "workflow_ready" and decision.workflow_candidate is not None:
        return WorkflowGateResult(True, decision.workflow_candidate, "workflow_ready", "route_operational")
    if decision.interaction_type == "workflow_candidate" and decision.workflow_readiness == "ready" and decision.workflow_candidate is not None:
        return WorkflowGateResult(True, decision.workflow_candidate, "workflow_candidate_ready", "route_operational")
    if decision.interaction_type == "workflow_candidate":
        return WorkflowGateResult(False, Intent.POTENTIALLY_RELEVANT_UNCLEAR, "workflow_candidate_needs_clarification", "ask_clarification")
    if decision.interaction_type == "unclear" or decision.needs_clarification:
        return WorkflowGateResult(False, Intent.POTENTIALLY_RELEVANT_UNCLEAR, "decision_needs_clarification", "ask_clarification")
    return WorkflowGateResult(False, Intent.FALLBACK_UNKNOWN, "no_operational_workflow", "fallback_with_service_options")
