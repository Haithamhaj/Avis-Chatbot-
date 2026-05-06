from __future__ import annotations

import re

from avis_ai_demo.core.conversation_manager import classify_conversation
from avis_ai_demo.core.data_lookup import match_escalation
from avis_ai_demo.core.types import Intent


def route_intent(message: str, gpt_intent: str | None = None) -> Intent:
    text = message.lower()
    conversation_intent = route_conversation_intent(message)
    if conversation_intent == Intent.COMPLAINT_OR_DISPUTE:
        return Intent.COMPLAINT_OR_FINANCIAL_DISPUTE
    if conversation_intent == Intent.ROADSIDE_OR_ACCIDENT:
        return Intent.ROADSIDE_ASSISTANCE
    if conversation_intent is not None and conversation_intent not in {Intent.OPERATIONAL_REQUEST, Intent.FALLBACK_UNKNOWN}:
        return conversation_intent
    if match_escalation(message):
        return Intent.COMPLAINT_OR_FINANCIAL_DISPUTE
    if any(token in text for token in ["accident", "injury", "not drivable", "حادث", "إصابة", "السيارة ما تتحرك"]):
        return Intent.ROADSIDE_ASSISTANCE
    if any(token in text for token in ["broke down", "towing", "battery", "تعطلت", "سطحة", "البطارية"]):
        return Intent.ROADSIDE_ASSISTANCE
    if _is_discount_or_offer_query(text):
        return Intent.GENERAL_FAQ
    if gpt_intent and gpt_intent != Intent.FALLBACK_UNKNOWN.value:
        try:
            return Intent(gpt_intent)
        except ValueError:
            pass
    if any(token in text for token in ["شهري", "monthly", "mini lease", "شهر"]):
        return Intent.MONTHLY_RENTAL
    if any(token in text for token in ["branch", "فرع", "airport", "السليمانية"]):
        return Intent.BRANCH_LOOKUP
    if any(token in text for token in ["credit card", "deposit", "وديعة", "بطاقة"]):
        return Intent.CARD_DEPOSIT_POLICY
    if any(token in text for token in ["شركات", "corporate", "leasing", "أسطول"]):
        return Intent.GENERAL_FAQ
    if _is_price_query(text):
        return Intent.FLEET_PRICING
    if any(token in text for token in ["rent", "استأجر", "أستأجر", "أبغى", "ابغى", "سيارة", "yaris", "يارس", "camry", "كامري"]):
        return Intent.DAILY_RENTAL
    return Intent.FALLBACK_UNKNOWN


def _is_price_query(text: str) -> bool:
    if any(token in text for token in ["price", "how much", "سعر"]):
        return True
    return bool(re.search(r"(^|\s)كم(\s|$)", text))


def _is_discount_or_offer_query(text: str) -> bool:
    if any(token in text for token in ["انخصم", "wrong charge", "double charge", "refund"]):
        return False
    return any(token in text for token in ["خصم", "عروض", "عرض", "برومو", "discount", "offer", "promo"])


def route_conversation_intent(message: str) -> Intent | None:
    classification = classify_conversation(message)
    return None if classification.intent == Intent.FALLBACK_UNKNOWN else classification.intent
