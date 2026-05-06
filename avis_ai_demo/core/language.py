from __future__ import annotations

import re


ARABIC_RE = re.compile(r"[\u0600-\u06ff]")
SHORT_TOKENS = {"ok", "okay", "thanks", "thank you", "yes", "no", "تمام", "نعم", "لا"}


def detect_language(message: str, previous_language: str = "ar") -> str:
    text = " ".join(message.strip().lower().split())
    if text in SHORT_TOKENS:
        return previous_language
    if ARABIC_RE.search(message):
        return "ar"
    return "en"


def is_arabic(message: str) -> bool:
    return bool(ARABIC_RE.search(message))

