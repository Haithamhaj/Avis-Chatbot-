from __future__ import annotations

import re


def detect_ambiguities(message: str, extracted: dict | None = None) -> list[str]:
    text = message.lower()
    ambiguities: list[str] = []
    if any(token in text for token in ["سعر", "price", "كم"]) and not any(
        token in text for token in ["يومي", "daily", "شهري", "monthly", "شهر"]
    ):
        ambiguities.append("rental_type")
    if re.search(r"(الخميس|الجمعة|السبت|الأحد|الاثنين|الثلاثاء|الأربعاء).*(\b[0-9]{1,2}\b)", message):
        if not any(token in message for token in ["صباح", "مساء", "ظهر", "am", "pm", "AM", "PM"]):
            ambiguities.append("date_time")
    if extracted and extracted.get("ambiguities"):
        for item in extracted["ambiguities"]:
            if item not in ambiguities:
                ambiguities.append(item)
    return ambiguities


def clarification_for_ambiguity(ambiguities: list[str], language: str) -> str | None:
    if not ambiguities:
        return None
    first = ambiguities[0]
    if first == "rental_type":
        return "Do you mean daily or monthly rental?" if language == "en" else "تقصد السعر اليومي أم الشهري؟"
    if first == "date_time":
        return (
            "Do you mean the coming weekday, and is the time AM or PM?"
            if language == "en"
            else "تقصد اليوم القادم؟ والوقت صباحًا أم مساءً؟"
        )
    return "Please clarify the missing details." if language == "en" else "فضلاً وضّح التفاصيل الناقصة."

