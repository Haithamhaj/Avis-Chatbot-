from __future__ import annotations

from dataclasses import dataclass

from avis_ai_demo.core.schemas import parse_conversation_classification
from avis_ai_demo.core.types import Intent
from avis_ai_demo.services.openai_service import OpenAIService


SEMANTIC_CONFIDENCE_THRESHOLD = 0.55


@dataclass(frozen=True)
class ConversationClassification:
    intent: Intent
    phase: str
    tone_mode: str
    customer_mood: str
    next_action: str
    confidence: float = 1.0


def classify_conversation(
    message: str,
    service: OpenAIService | None = None,
    state: dict | None = None,
) -> ConversationClassification:
    text = " ".join(message.strip().lower().split())
    if not text:
        return _fallback()

    raw_previous_intent = (state or {}).get("intent") or ""
    previous_intent = raw_previous_intent.value if isinstance(raw_previous_intent, Intent) else str(raw_previous_intent)
    if previous_intent == Intent.POTENTIALLY_RELEVANT_UNCLEAR.value and _is_unclear_negative_followup(text):
        return ConversationClassification(
            Intent.POTENTIALLY_RELEVANT_UNCLEAR,
            "conversation_management",
            "serious_supportive",
            "frustrated",
            "ask_open_clarification",
        )

    deterministic = _deterministic_conversation_classification(text)
    if deterministic.intent != Intent.FALLBACK_UNKNOWN:
        return deterministic

    if service is not None:
        try:
            parsed = parse_conversation_classification(service.classify_conversation(message, state))
            return _from_gpt_classification(parsed, text)
        except Exception:
            pass

    return _fallback()


def _deterministic_conversation_classification(text: str) -> ConversationClassification:
    if _is_greeting(text):
        return ConversationClassification(Intent.GREETING, "conversation_management", "friendly_casual", "friendly", "wait_for_user_request")
    if _is_thanks(text):
        return ConversationClassification(Intent.THANKS_ACKNOWLEDGEMENT, "conversation_management", "friendly_casual", "grateful", "wait_for_user_request")
    if _is_small_talk(text):
        return ConversationClassification(Intent.SMALL_TALK, "conversation_management", "friendly_casual", "casual", "scope_redirect")
    if _is_capability_question(text):
        return ConversationClassification(Intent.CAPABILITY_QUESTION, "conversation_management", "professional_helpful", "curious", "answer_capabilities")
    if _is_company_or_general_faq(text):
        return ConversationClassification(Intent.GENERAL_FAQ, "operational_routing", "professional_helpful", "curious", "lookup_general_faq")
    if _is_off_topic(text):
        return ConversationClassification(Intent.OFF_TOPIC, "conversation_management", "light_deflection", "casual", "scope_redirect")
    if _is_roadside_or_accident(text):
        return ConversationClassification(Intent.ROADSIDE_OR_ACCIDENT, "operational_routing", "safety_first", "worried", "safety_check")
    if _has_explicit_complaint_or_dispute_details(text):
        return ConversationClassification(Intent.COMPLAINT_OR_DISPUTE, "operational_routing", "serious_supportive", "frustrated", "collect_case_details")
    if _is_discount_or_offer_query(text):
        return ConversationClassification(Intent.GENERAL_FAQ, "operational_routing", "friendly_casual", "playful_positive", "lookup_general_faq")
    if _is_negative_sentiment_unclear(text):
        return ConversationClassification(
            Intent.POTENTIALLY_RELEVANT_UNCLEAR,
            "conversation_management",
            "serious_supportive",
            "frustrated",
            "ask_clarification",
        )
    if _is_potentially_relevant_unclear(text):
        return ConversationClassification(
            Intent.POTENTIALLY_RELEVANT_UNCLEAR,
            "conversation_management",
            "professional_helpful",
            "unclear",
            "ask_clarification",
        )
    if _is_operational_request(text):
        return ConversationClassification(Intent.OPERATIONAL_REQUEST, "operational_routing", "professional_helpful", "task_focused", "route_operational")
    return _fallback()


def _from_gpt_classification(data: dict, text: str) -> ConversationClassification:
    intent = Intent(data["intent"])
    confidence = float(data["confidence"])
    if intent == Intent.COMPLAINT_OR_DISPUTE and not _has_explicit_complaint_or_dispute_details(text):
        return ConversationClassification(
            Intent.POTENTIALLY_RELEVANT_UNCLEAR,
            "conversation_management",
            "serious_supportive",
            data["customer_mood"] if data["customer_mood"] != "neutral" else "frustrated",
            "ask_clarification",
            confidence,
        )
    if confidence < SEMANTIC_CONFIDENCE_THRESHOLD and intent not in {
        Intent.OPERATIONAL_REQUEST,
        Intent.COMPLAINT_OR_DISPUTE,
        Intent.ROADSIDE_OR_ACCIDENT,
    }:
        return ConversationClassification(
            Intent.POTENTIALLY_RELEVANT_UNCLEAR,
            "conversation_management",
            "professional_helpful",
            data["customer_mood"],
            "ask_clarification",
            confidence,
        )
    phase = "operational_routing" if intent in {
        Intent.OPERATIONAL_REQUEST,
        Intent.COMPLAINT_OR_DISPUTE,
        Intent.ROADSIDE_OR_ACCIDENT,
    } else "conversation_management"
    return ConversationClassification(
        intent=intent,
        phase=phase,
        tone_mode=data["tone_mode"],
        customer_mood=data["customer_mood"],
        next_action=_normalise_next_action(intent, data["next_action"]),
        confidence=confidence,
    )


def is_conversation_only(intent: Intent) -> bool:
    return intent in {
        Intent.GREETING,
        Intent.SMALL_TALK,
        Intent.OFF_TOPIC,
        Intent.THANKS_ACKNOWLEDGEMENT,
        Intent.CAPABILITY_QUESTION,
        Intent.POTENTIALLY_RELEVANT_UNCLEAR,
    }


def _is_greeting(text: str) -> bool:
    return text in {"مرحبا", "مرحباً", "هلا", "السلام عليكم", "اهلا", "أهلا", "hello", "hi", "hey"}


def _is_thanks(text: str) -> bool:
    return text in {"شكرا", "شكرًا", "يعطيك العافية", "مشكور", "thanks", "thank you", "thx"}


def _is_small_talk(text: str) -> bool:
    markers = [
        "كيفك",
        "شلونك",
        "عامل إيه",
        "عامل ايه",
        "أخبارك",
        "اخبارك",
        "وش علومك",
        "ايش الاخبار",
        "إيش الأخبار",
        "وش الاخبار",
        "وش الأخبار",
        "علومك",
        "يا حلو",
        "يا بطل",
        "يا غالي",
        "بحكيلك",
        "بحكي لك",
        "معك يا",
        "how are you",
        "how's it going",
        "how are things",
    ]
    return any(marker in text for marker in markers)


def _is_capability_question(text: str) -> bool:
    markers = ["إيش تقدر تسوي", "ايش تقدر تسوي", "وش تقدر تسوي", "كيف تقدر تساعدني", "what can you do", "how can you help"]
    return any(marker in text for marker in markers)


def _is_company_or_general_faq(text: str) -> bool:
    markers = [
        "شركة ايش",
        "شركة إيش",
        "انتم شركة ايش",
        "انتو شركة ايش",
        "ايش افيس",
        "إيش أفيس",
        "وش افيس",
        "وش أفيس",
        "من انتم",
        "مين انتم",
        "لوجو افيس",
        "لوجو أفيس",
        "what is avis",
        "who are you",
        "about avis",
        "حلول للشركات",
        "تأجير للشركات",
        "عقد شركات",
        "أسطول شركة",
        "corporate rental",
        "business account",
        "operational leasing",
        "لماذا آيفس",
        "لماذا أفيس",
        "ليش افيس",
        "ليش أفيس",
        "ليش آيفس",
        "ليش اختار افيس",
        "ليش أختار أفيس",
        "ليش أختار آيفس",
        "مميزات افيس",
        "مميزات أفيس",
        "خبرة افيس",
        "خبرة أفيس",
        "اسطول افيس",
        "أسطول أفيس",
        "جوائز افيس",
        "جوائز أفيس",
        "why choose avis",
        "avis advantages",
        "avis awards",
    ]
    return any(marker in text for marker in markers)


def _is_off_topic(text: str) -> bool:
    markers = ["الهلال", "النصر", "مباراة", "football", "soccer", "سياسة", "طبخة", "movie", "weather"]
    return any(marker in text for marker in markers)


def _is_roadside_or_accident(text: str) -> bool:
    markers = ["حادث", "إصابة", "اصابة", "صار لي حادث", "تعطلت", "سطحة", "البطارية", "battery", "accident", "injury", "broke down", "towing"]
    return any(marker in text for marker in markers)


def _has_explicit_complaint_or_dispute_details(text: str) -> bool:
    markers = [
        "وديعة ما رجعت",
        "الوديعة ما رجعت",
        "ما رجعت الوديعة",
        "رجعوا الوديعة",
        "انخصم",
        "خصمتوا",
        "خصمتو",
        "خصمتوا من",
        "خصمتو من",
        "مبلغ زيادة",
        "مشكلة في الدفع",
        "مشكلة بالدفع",
        "دفعت",
        "دفعت وما",
        "دفعت ولا",
        "دفعت ولم",
        "دفع وما",
        "اعتراض",
        "ارفع شكوى",
        "أرفع شكوى",
        "ابي اشتكي",
        "أبي أشتكي",
        "ابغى اشتكي",
        "أبغى أشتكي",
        "شكوى",
        "refund",
        "wrong charge",
        "double charge",
        "deposit not released",
        "payment complaint",
        "payment problem",
    ]
    return any(marker in text for marker in markers)


def _is_negative_sentiment_unclear(text: str) -> bool:
    markers = [
        "ما احب افيس",
        "ما أحب أفيس",
        "ما احب ايفيس",
        "ما أحب إيفيس",
        "ما احبكم",
        "ما أحبكم",
        "نصابين",
        "احسكم نصابين",
        "أحسكم نصابين",
        "مو عاجبني",
        "ما عجبني",
        "تجربة سيئة",
        "خدمة سيئة",
        "خدمتكم سيئة",
        "زعلان منكم",
        "متضايق منكم",
        "bad service",
        "i do not like avis",
        "i don't like avis",
        "not happy with avis",
    ]
    return any(marker in text for marker in markers)


def _is_unclear_negative_followup(text: str) -> bool:
    markers = [
        "بدون سبب",
        "كذا",
        "لا اعرف",
        "لا أعرف",
        "ما ادري",
        "ما أدري",
        "مدري",
        "مش عارف",
        "مو عارف",
        "مجرد احساس",
        "مجرد إحساس",
        "احساس",
        "إحساس",
        "no reason",
        "not sure why",
        "just a feeling",
    ]
    return any(marker in text for marker in markers)


def _is_discount_or_offer_query(text: str) -> bool:
    if any(token in text for token in ["انخصم", "بطاقة خصم", "خصمتوا من", "خصمتو من"]):
        return False
    return any(token in text for token in ["خصم", "عروض", "عرض", "برومو", "كود خصم", "discount", "offer", "promo"])


def _is_potentially_relevant_unclear(text: str) -> bool:
    markers = ["أبغى أرتب", "ابغى ارتب", "أبغى أرتب الموضوع", "ابغى ارتب الموضوع", "help me arrange", "i need help with this"]
    return any(marker.lower() in text for marker in markers)


def _is_operational_request(text: str) -> bool:
    markers = ["سيارة", "حجز", "سعر", "فرع", "وديعة", "بطاقة", "rent", "booking", "price", "branch", "deposit", "monthly", "daily", "yaris", "camry", "يارس", "كامري"]
    return any(marker in text for marker in markers)


def _fallback() -> ConversationClassification:
    return ConversationClassification(Intent.FALLBACK_UNKNOWN, "conversation_management", "professional_helpful", "unclear", "fallback_with_service_options", 0.0)


def _normalise_next_action(intent: Intent, raw_next_action: str) -> str:
    allowed = {
        "wait_for_user_request",
        "scope_redirect",
        "answer_capabilities",
        "ask_clarification",
        "route_operational",
        "collect_case_details",
        "safety_check",
        "fallback_with_service_options",
    }
    if raw_next_action in allowed:
        return raw_next_action
    defaults = {
        Intent.GREETING: "wait_for_user_request",
        Intent.THANKS_ACKNOWLEDGEMENT: "wait_for_user_request",
        Intent.SMALL_TALK: "scope_redirect",
        Intent.OFF_TOPIC: "scope_redirect",
        Intent.CAPABILITY_QUESTION: "answer_capabilities",
        Intent.GENERAL_FAQ: "lookup_general_faq",
        Intent.POTENTIALLY_RELEVANT_UNCLEAR: "ask_clarification",
        Intent.OPERATIONAL_REQUEST: "route_operational",
        Intent.COMPLAINT_OR_DISPUTE: "collect_case_details",
        Intent.ROADSIDE_OR_ACCIDENT: "safety_check",
    }
    return defaults.get(intent, "fallback_with_service_options")
