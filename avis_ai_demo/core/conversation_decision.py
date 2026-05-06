from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, TYPE_CHECKING

from avis_ai_demo.core.conversation_history import compact_history_for_ai
from avis_ai_demo.core.schemas import parse_conversation_decision
from avis_ai_demo.core.types import Intent

if TYPE_CHECKING:
    from avis_ai_demo.services.openai_service import OpenAIService


DECISION_CONFIDENCE_THRESHOLD = 0.62


@dataclass(frozen=True)
class ConversationDecision:
    interaction_type: str
    service_domain: str
    workflow_candidate: Intent | None
    workflow_readiness: str
    confidence: float
    customer_mood: str
    needs_clarification: bool
    suggested_dialogue_act: str
    source: str = "deterministic"

    def to_trace(self) -> dict[str, Any]:
        data = asdict(self)
        data["workflow_candidate"] = self.workflow_candidate.value if self.workflow_candidate else None
        return data


def decide_conversation(
    message: str,
    service: OpenAIService | None = None,
    state: dict[str, Any] | None = None,
) -> ConversationDecision:
    text = " ".join(message.strip().lower().split())
    hard = _hard_decision(text)
    if hard:
        return hard

    deterministic = _deterministic_hint_decision(text)
    if deterministic and deterministic.interaction_type in {"faq_or_info", "social_with_service_hint"}:
        return deterministic

    if service is not None:
        try:
            parsed = parse_conversation_decision(service.decide_conversation(message, state))
            decision = _from_payload(parsed)
            if decision.confidence >= DECISION_CONFIDENCE_THRESHOLD:
                return decision
        except Exception:
            pass

    return deterministic or ConversationDecision(
        interaction_type="unclear",
        service_domain="none",
        workflow_candidate=None,
        workflow_readiness="not_applicable",
        confidence=0.0,
        customer_mood="unclear",
        needs_clarification=True,
        suggested_dialogue_act="ask_clarification",
    )


def decision_context(state: dict[str, Any] | None) -> dict[str, Any]:
    state = state or {}
    return {
        "active_language": state.get("language"),
        "current_intent": str(state.get("intent") or ""),
        "current_phase": state.get("phase"),
        "next_action": state.get("next_action"),
        "conversation_memory": compact_history_for_ai(state),
        "note": "Use for conversation decision only. Do not infer prices, availability, payment, booking, branch, or policy facts from memory.",
    }


def _from_payload(data: dict[str, Any]) -> ConversationDecision:
    candidate = Intent(data["workflow_candidate"]) if data["workflow_candidate"] else None
    return ConversationDecision(
        interaction_type=data["interaction_type"],
        service_domain=data["service_domain"],
        workflow_candidate=candidate,
        workflow_readiness=data["workflow_readiness"],
        confidence=float(data["confidence"]),
        customer_mood=data["customer_mood"],
        needs_clarification=bool(data["needs_clarification"]),
        suggested_dialogue_act=data["suggested_dialogue_act"],
        source="gpt",
    )


def _hard_decision(text: str) -> ConversationDecision | None:
    if _looks_like_safety_or_drive_advice(text):
        return ConversationDecision("hard_safety", "roadside", Intent.ROADSIDE_ASSISTANCE, "ready", 1.0, "worried", False, "safety_check")
    if _looks_like_international_case(text):
        return ConversationDecision("hard_financial_dispute", "complaint", Intent.COMPLAINT_OR_FINANCIAL_DISPUTE, "ready", 1.0, "frustrated", False, "collect_case_details")
    if any(token in text for token in [
        "انخصم",
        "خصمتوا من",
        "خصمتو من",
        "مبلغ زيادة",
        "الوديعة ما رجعت",
        "وديعة ما رجعت",
        "دفعت وما",
        "دفعت ولا",
        "wrong charge",
        "double charge",
        "deposit not released",
        "payment complaint",
        "refund",
    ]):
        return ConversationDecision("hard_financial_dispute", "complaint", Intent.COMPLAINT_OR_FINANCIAL_DISPUTE, "ready", 1.0, "frustrated", False, "collect_case_details")
    if _looks_like_service_feedback(text):
        return ConversationDecision("service_feedback", "service_feedback", Intent.SERVICE_EXPERIENCE_FEEDBACK, "ready", 0.92, "upset", False, "collect_feedback_details")
    return None


def _deterministic_hint_decision(text: str) -> ConversationDecision | None:
    if _is_card_policy_query(text):
        return ConversationDecision("faq_or_info", "deposit_policy", Intent.CARD_DEPOSIT_POLICY, "not_applicable", 0.94, "curious", False, "answer_from_kb")
    if _is_discount_or_offer_query(text):
        mood = "playful_positive" if any(token in text for token in ["بحب", "أحب", "احب", "هههه"]) else "curious"
        return ConversationDecision("social_with_service_hint", "offers", Intent.GENERAL_FAQ, "not_applicable", 0.9, mood, False, "acknowledge_then_answer")
    if _looks_like_ready_daily_booking(text):
        return ConversationDecision("workflow_ready", "booking", Intent.DAILY_RENTAL, "ready", 0.82, "task_focused", False, "route_operational")
    if _looks_like_monthly_pricing(text):
        return ConversationDecision("workflow_ready", "pricing", Intent.MONTHLY_RENTAL, "ready", 0.78, "task_focused", False, "route_operational")
    if any(token in text for token in ["شركة ايش", "انتو شركة", "ايش افيس", "وش أفيس", "من انتم", "ليش أختار", "ليش اختار", "مميزات أفيس", "مميزات افيس", "about avis", "what is avis", "why choose avis"]):
        return ConversationDecision("faq_or_info", "company_info", Intent.GENERAL_FAQ, "not_applicable", 0.9, "curious", False, "answer_from_kb")
    if any(token in text for token in ["ودي أرتب", "ودي ارتب", "أبغى أرتب", "ابغى ارتب"]):
        return ConversationDecision("workflow_candidate", "booking", Intent.DAILY_RENTAL, "not_ready", 0.74, "task_focused", True, "ask_clarification")
    return None


def _is_discount_or_offer_query(text: str) -> bool:
    if any(token in text for token in ["انخصم", "بطاقة خصم", "خصمتوا من", "خصمتو من"]) or _is_card_policy_query(text):
        return False
    return any(token in text for token in ["خصم", "عروض", "عرض", "برومو", "كود خصم", "discount", "offer", "promo"])


def _is_card_policy_query(text: str) -> bool:
    card_markers = ["بطاقة", "card", "ائتمان", "credit", "debit", "مدى", "mada"]
    debit_markers = ["بطاقة خصم", "debit card", "مو ائتمان", "not credit", "مدى", "mada"]
    deposit_markers = ["وديعة", "deposit", "pre-authorization", "تفويض"]
    return (
        any(marker in text for marker in debit_markers)
        or (any(marker in text for marker in card_markers) and any(marker in text for marker in ["تنفع", "ينفع", "أستأجر", "استأجر", "rent", "require"]))
        or any(marker in text for marker in deposit_markers)
    )


def _looks_like_service_feedback(text: str) -> bool:
    service_context = ["فرع", "الموظف", "موظف", "زحمة", "تأخر", "تاخر", "زيارة", "رحت", "ريحتها", "دخان", "السيارة كانت", "visited", "branch", "staff", "queue", "delay", "smoke smell", "dirty car"]
    upset = ["متضايق", "زعلان", "مو عاجب", "تجربة سيئة", "خدمة سيئة", "رفعت ضغطي", "تأخر", "تاخر", "زحمة", "دخان", "قال عادي", "upset", "bad service", "not happy"]
    financial = ["انخصم", "وديعة ما رجعت", "خصمتوا", "خصمتو", "refund", "wrong charge", "double charge"]
    return any(marker in text for marker in service_context) and any(marker in text for marker in upset) and not any(marker in text for marker in financial)


def _looks_like_safety_or_drive_advice(text: str) -> bool:
    return any(token in text for token in [
        "حادث",
        "إصابة",
        "اصابة",
        "السيارة ما تتحرك",
        "accident",
        "injury",
        "not drivable",
        "ترجف",
        "تطفي",
        "تطفى",
        "أمشي ولا أوقف",
        "امشي ولا اوقف",
        "أكمل عليها",
        "اكمل عليها",
        "لازم أوقف",
        "safe to drive",
        "continue driving",
    ])


def _looks_like_international_case(text: str) -> bool:
    international = any(token in text for token in ["دبي", "خارج السعودية", "محطة خارجية", "outside saudi", "dubai", "international rental", "foreign station"])
    issue = any(token in text for token in ["وديعة", "deposit", "refund", "معلقة", "معلق", "العقد", "contract"])
    return international and issue


def _looks_like_ready_daily_booking(text: str) -> bool:
    has_vehicle = any(token in text for token in ["يارس", "كامري", "yaris", "camry", "سيارة"])
    has_city = any(token in text for token in ["الرياض", "جدة", "الدمام", "riyadh", "jeddah", "dammam"])
    has_duration = any(token in text for token in ["يوم", "يومين", "ايام", "أيام", "days"])
    has_time_or_date = any(token in text for token in ["2026-", "الساعة", "بكرة", "tomorrow"])
    return has_vehicle and has_city and has_duration and has_time_or_date


def _looks_like_monthly_pricing(text: str) -> bool:
    has_monthly = any(token in text for token in ["monthly", "شهري", "شهر"])
    has_vehicle = any(token in text for token in ["يارس", "كامري", "yaris", "camry"])
    return has_monthly and has_vehicle
