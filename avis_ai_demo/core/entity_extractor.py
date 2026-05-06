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
    if _is_international_case(text):
        intent = Intent.COMPLAINT_OR_FINANCIAL_DISPUTE.value
    elif _is_financial_dispute(text):
        intent = Intent.COMPLAINT_OR_FINANCIAL_DISPUTE.value
    elif _is_card_policy_query(text):
        intent = Intent.CARD_DEPOSIT_POLICY.value
    elif _is_roadside_signal(text):
        intent = Intent.ROADSIDE_ASSISTANCE.value
    elif any(token in text for token in ["شهري", "monthly", "mini lease"]):
        intent = Intent.MONTHLY_RENTAL.value
    elif any(token in text for token in ["فرع", "branch", "airport", "المطار", "السليمانية"]):
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
    directional = _extract_directional_cities(text)
    found_cities = []
    for canonical, aliases in city_aliases:
        if any(alias in text or alias in message for alias in aliases):
            found_cities.append(canonical)
    if directional:
        entities["pickup_city"] = directional[0]
        entities["dropoff_city"] = directional[1]
    elif found_cities:
        entities["pickup_city"] = found_cities[0]
    if not directional and len(found_cities) > 1:
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
    age_match = re.search(r"(?:عمري|age|i am)\s*([1-9][0-9])\b", text)
    if age_match:
        entities["age"] = int(age_match.group(1))
    name_match = re.search(r"(?:اسمي|my name is)\s+([\w\u0600-\u06ff]+)", message, flags=re.IGNORECASE)
    if name_match:
        entities["customer_name"] = name_match.group(1)
    date_match = re.search(r"\b(20[0-9]{2}-[0-9]{2}-[0-9]{2})\b", text)
    if date_match:
        entities["pickup_date"] = date_match.group(1)
    elif "بكرة" in message or "tomorrow" in text:
        entities["pickup_date"] = "tomorrow"
    time_match = re.search(r"(?:الساعة|at)\s*([0-9]{1,2}(?::[0-9]{2})?)", message, flags=re.IGNORECASE)
    if not time_match:
        time_match = re.search(r"(?:بكرة|tomorrow)\s+([0-9]{1,2})(?:\b|،|,)", message, flags=re.IGNORECASE)
    if time_match:
        entities["pickup_time"] = time_match.group(1)
    mobile_match = re.search(r"(05[0-9]{8})", text)
    if mobile_match:
        entities["mobile"] = mobile_match.group(1)
    contract_match = re.search(r"\b(?:av|ra)[a-z0-9-]*\d+\b", text, flags=re.IGNORECASE)
    if contract_match:
        entities["plate_or_contract"] = contract_match.group(0).upper()
        entities["booking_or_contract"] = contract_match.group(0).upper()
    if "أمس" in message or "امس" in message or "yesterday" in text:
        entities["transaction_date"] = "yesterday"
    if any(token in text for token in ["لا قصدي", "اقصد", "أقصد", "صحح", "بدل", "غير"]):
        entities["is_correction"] = True
    if _is_roadside_signal(text):
        entities.setdefault("issue_type", "breakdown")
        if any(token in text for token in ["ما أدري أمشي", "ما ادري امشي", "أمشي ولا أوقف", "امشي ولا اوقف", "أكمل عليها", "اكمل عليها", "ترجف"]):
            entities.setdefault("drivable_status", "uncertain")
        if any(token in text for token in ["مافي إصابات", "ما فيه إصابات", "لا إصابات", "no injuries"]):
            entities.setdefault("safety_status", "safe")

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


def _is_financial_dispute(text: str) -> bool:
    return any(token in text for token in ["refund", "double charge", "wrong charge", "انخصم", "خصمتوا", "خصمتو", "مشكلة في الدفع", "payment problem", "payment complaint", "وديعة ما رجعت", "ما رجعت الوديعة"])


def _is_card_policy_query(text: str) -> bool:
    return any(token in text for token in ["بطاقة خصم", "debit card", "مو ائتمان", "not credit", "مدى", "mada", "credit card", "بطاقة ائتمان", "وديعة", "deposit"])


def _is_roadside_signal(text: str) -> bool:
    return any(token in text for token in ["تعطلت", "سطحة", "battery", "towing", "broke down", "ترجف", "تطفي", "تطفى", "أمشي ولا أوقف", "امشي ولا اوقف", "أكمل عليها", "اكمل عليها", "لازم أوقف", "safe to drive", "continue driving"])


def _is_international_case(text: str) -> bool:
    international = any(token in text for token in ["دبي", "خارج السعودية", "محطة خارجية", "outside saudi", "dubai", "international rental", "foreign station"])
    issue = any(token in text for token in ["وديعة", "deposit", "refund", "معلقة", "معلق", "العقد", "contract"])
    return international and issue


def _extract_directional_cities(text: str) -> tuple[str, str] | None:
    text = (
        text.replace("للرياض", "ل الرياض")
        .replace("لجدة", "ل جدة")
        .replace("لجده", "ل جده")
        .replace("للدمام", "ل الدمام")
        .replace("للجبيل", "ل الجبيل")
        .replace("لتبوك", "ل تبوك")
    )
    cities = {
        "الرياض": "Riyadh",
        "riyadh": "Riyadh",
        "جدة": "Jeddah",
        "جده": "Jeddah",
        "jeddah": "Jeddah",
        "الدمام": "Dammam",
        "dammam": "Dammam",
        "الجبيل": "Jubail",
        "jubail": "Jubail",
        "تبوك": "Tabuk",
        "tabuk": "Tabuk",
    }
    names = "|".join(re.escape(name) for name in cities)
    patterns = [
        rf"من\s+({names})\s+(?:إلى|الى|ل)\s*({names})",
        rf"from\s+({names})\s+to\s+({names})",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return cities[match.group(1).lower() if match.group(1).isascii() else match.group(1)], cities[match.group(2).lower() if match.group(2).isascii() else match.group(2)]
    return None


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
