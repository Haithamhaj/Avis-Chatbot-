from __future__ import annotations

from functools import lru_cache
from typing import Any

from avis_ai_demo.core.data_lookup import load_kb
from avis_ai_demo.core.types import AnswerContext, Intent


BLOCKED_TONE_INTENTS = {
    Intent.COMPLAINT_OR_DISPUTE,
    Intent.COMPLAINT_OR_FINANCIAL_DISPUTE,
    Intent.ROADSIDE_OR_ACCIDENT,
    Intent.ROADSIDE_ASSISTANCE,
}


def candidate_tone_phrases(context: AnswerContext, *, auto_only: bool = False) -> list[str]:
    if context.language != "ar" or _tone_is_blocked(context):
        return []
    phrases = []
    for record in _tone_records():
        if auto_only and not record.get("auto_use"):
            continue
        if context.intent.value not in set(record.get("intents") or []):
            continue
        if context.tone_mode not in set(record.get("tone_modes") or []):
            continue
        phrases.append(record["phrase"])
    return phrases


def apply_optional_tone_phrase(response: str, context: AnswerContext) -> str:
    response = _dedupe_tone_phrases(response)
    if "\n" in response and context.intent not in {Intent.GENERAL_FAQ}:
        return response
    phrases = candidate_tone_phrases(context, auto_only=True)
    if not phrases or any(response.startswith(phrase) for phrase in phrases):
        return response
    phrase = _select_phrase(phrases, context)
    if not phrase:
        return response
    if _already_has_opening_tone(response, phrase):
        return response
    return f"{phrase}، {response}"


@lru_cache(maxsize=1)
def _tone_records() -> tuple[dict[str, Any], ...]:
    return tuple(load_kb("kb17-tone-lexicon.json"))


def _tone_is_blocked(context: AnswerContext) -> bool:
    if context.intent in BLOCKED_TONE_INTENTS:
        return True
    if context.escalation_required or context.risk_level == "high":
        return True
    if context.computed_totals is not None:
        return True
    if context.phase in {"payment_processed", "booking_completed", "awaiting_payment_confirmation"}:
        return True
    return False


def _select_phrase(phrases: list[str], context: AnswerContext) -> str:
    if context.intent in {Intent.GREETING, Intent.SMALL_TALK} and "هلا وارحب" in phrases:
        return "هلا وارحب"
    if context.intent in {Intent.DAILY_RENTAL, Intent.MONTHLY_RENTAL, Intent.BRANCH_LOOKUP, Intent.FLEET_PRICING} and "أبشر" in phrases:
        return "أبشر"
    if context.intent == Intent.GENERAL_FAQ and context.user_message and "أبشر" in phrases:
        return "أبشر"
    return phrases[0] if phrases else ""


def _already_has_opening_tone(response: str, phrase: str) -> bool:
    opening = response.strip()[:80]
    greeting_markers = [
        "مرحبا",
        "مرحبًا",
        "هلا",
        "أهلًا",
        "اهلا",
        "حياك",
        "يا هلا",
        "تشرفنا",
    ]
    service_markers = ["أكيد", "تمام", "حاضر", "أقدر", "يسعدني"]
    if phrase == "هلا وارحب":
        return any(marker in opening for marker in greeting_markers)
    if phrase == "أبشر":
        return any(marker in opening for marker in service_markers)
    return False


def _dedupe_tone_phrases(response: str) -> str:
    text = response.strip()
    tone_phrases = [record["phrase"] for record in _tone_records()]
    found = [phrase for phrase in tone_phrases if phrase in text[:100]]
    if len(found) <= 1:
        return text

    keep = found[0]
    for phrase in found[1:]:
        text = text.replace(f"، {phrase}", "", 1)
        text = text.replace(f"{phrase}، ", "", 1)
        text = text.replace(phrase, "", 1)
    text = " ".join(text.split())
    if keep not in text[:100]:
        text = f"{keep}، {text}"
    return text
