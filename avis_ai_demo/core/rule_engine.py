from __future__ import annotations

from typing import Any

from avis_ai_demo.core.data_lookup import match_escalation
from avis_ai_demo.core.types import AnswerContext, Intent


def build_escalation_context(message: str, language: str) -> AnswerContext | None:
    rule = match_escalation(message)
    if not rule:
        return None
    return AnswerContext(
        intent=Intent.COMPLAINT_OR_FINANCIAL_DISPUTE,
        language="ar" if language == "ar" else "en",
        risk_level="high",
        escalation_required=True,
        missing_fields=rule.get("required_fields", []),
        next_action="handover_to_agent",
        caveats=[rule.get("response_ar" if language == "ar" else "response_en", "")],
    )


def category_availability_caveat(language: str) -> str:
    if language == "en":
        return "Models are representative within the category; final availability depends on branch and date."
    return "الموديلات المذكورة أمثلة ضمن الفئة، والتوفر النهائي يعتمد على الفرع والتاريخ."

