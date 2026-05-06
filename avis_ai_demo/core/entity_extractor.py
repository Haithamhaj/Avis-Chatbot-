from __future__ import annotations

import re
from typing import Any

from avis_ai_demo.core.schemas import SchemaValidationError, parse_gpt_extraction
from avis_ai_demo.core.types import Intent
from avis_ai_demo.services.openai_service import OpenAIService


ARABIC_RE = re.compile(r"[\u0600-\u06ff]")


def deterministic_parse(message: str, previous_language: str = "ar") -> dict[str, Any]:
    text = message.lower()
    language = "ar" if ARABIC_RE.search(message) else previous_language if text.strip() in {"ok", "yes", "no", "thanks"} else "en"
    intent = Intent.FALLBACK_UNKNOWN.value
    if any(token in text for token in ["وديعة", "refund", "double charge", "انخصم", "مشكلة في الدفع", "payment problem", "payment complaint"]):
        intent = Intent.COMPLAINT_OR_FINANCIAL_DISPUTE.value
    elif any(token in text for token in ["تعطلت", "سطحة", "battery", "towing", "broke down"]):
        intent = Intent.ROADSIDE_ASSISTANCE.value
    elif any(token in text for token in ["شهري", "monthly", "mini lease"]):
        intent = Intent.MONTHLY_RENTAL.value
    elif any(token in text for token in ["فرع", "branch", "airport", "السليمانية"]):
        intent = Intent.BRANCH_LOOKUP.value
    elif any(token in text for token in ["شركات", "corporate", "leasing", "أسطول"]):
        intent = Intent.GENERAL_FAQ.value
    elif _is_price_query(text):
        intent = Intent.FLEET_PRICING.value
    elif any(token in text for token in ["rent", "استأجر", "أستأجر", "ابغى", "أبغى", "سيارة", "يارس", "yaris"]):
        intent = Intent.DAILY_RENTAL.value

    entities: dict[str, Any] = {}
    city_aliases = [
        ("Riyadh", ["riyadh", "الرياض", "للرياض", "إلى الرياض", "الى الرياض"]),
        ("Jeddah", ["jeddah", "جدة", "جده", "لجدة", "لجده", "إلى جدة", "الى جدة"]),
        ("Dammam", ["dammam", "الدمام", "للدمام", "إلى الدمام", "الى الدمام"]),
        ("Jubail", ["jubail", "الجبيل", "للجبيل", "إلى الجبيل", "الى الجبيل"]),
        ("Tabuk", ["tabuk", "تبوك", "لتبوك", "إلى تبوك", "الى تبوك"]),
    ]
    found_cities = []
    for canonical, aliases in city_aliases:
        if any(alias in text or alias in message for alias in aliases):
            found_cities.append(canonical)
    if found_cities:
        entities["pickup_city"] = found_cities[0]
    if len(found_cities) > 1:
        entities["dropoff_city"] = found_cities[1]
    if "يارس" in message or "yaris" in text:
        entities["vehicle_query"] = "Yaris"
    if "كامري" in message or "camry" in text:
        entities["vehicle_query"] = "Camry"
    if "يومين" in message or "two days" in text:
        entities["rental_days"] = 2
    else:
        days_match = re.search(r"\b([1-9][0-9]?)\s*(?:days?|أيام|ايام|يوم)\b", text)
        if days_match:
            entities["rental_days"] = int(days_match.group(1))
    if any(token in text for token in ["عندي رخصة", "رخصة سارية", "valid license", "i have a license"]):
        entities["has_valid_license"] = True
    age_match = re.search(r"\b([1-9][0-9])\b", text)
    if age_match:
        entities["age"] = int(age_match.group(1))
    name_match = re.search(r"(?:اسمي|my name is)\s+([\w\u0600-\u06ff]+)", message, flags=re.IGNORECASE)
    if name_match:
        entities["customer_name"] = name_match.group(1)
    date_match = re.search(r"\b(20[0-9]{2}-[0-9]{2}-[0-9]{2})\b", text)
    if date_match:
        entities["pickup_date"] = date_match.group(1)
    time_match = re.search(r"(?:الساعة|at)\s*([0-9]{1,2}(?::[0-9]{2})?)", message, flags=re.IGNORECASE)
    if time_match:
        entities["pickup_time"] = time_match.group(1)

    return {
        "intent": intent,
        "language": language,
        "entities": entities,
        "ambiguities": ["gpt_unavailable_or_invalid"],
        "clarification_question": "Can you clarify the missing details?" if language == "en" else "ممكن توضح التفاصيل الناقصة؟",
        "confidence": 0.35,
    }


def _is_price_query(text: str) -> bool:
    if any(token in text for token in ["سعر", "price", "how much"]):
        return True
    return bool(re.search(r"(^|\s)كم(\s|$)", text))


def extract_with_gpt_or_fallback(
    message: str,
    state: dict[str, Any] | None = None,
    service: OpenAIService | None = None,
) -> tuple[dict[str, Any], bool, str | None]:
    service = service or OpenAIService()
    previous_language = (state or {}).get("language", "ar")
    try:
        raw = service.extract(message, state)
        return parse_gpt_extraction(raw), False, None
    except Exception as exc:
        fallback = deterministic_parse(message, previous_language=previous_language)
        return fallback, True, str(exc)
