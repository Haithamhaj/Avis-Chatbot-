from __future__ import annotations

import re
from dataclasses import dataclass, field

from avis_ai_demo.core.types import AnswerContext


FORBIDDEN_DISCLOSURE_PATTERNS = [
    "demo",
    "simulation",
    "mock",
    "not connected",
    "api absence",
    "no api",
    "تجريبي",
    "محاكاة",
    "غير متصل",
]
PAYMENT_SUCCESS_PATTERNS = ["payment completed successfully", "تم الدفع بنجاح", "paid successfully"]
INTERNAL_LEAK_PATTERNS = [
    r"\bKB\d{2}\b",
    "internal trace",
    "retrieved_records",
    "guard_failures",
    "workflow_state",
    "awaiting_payment_confirmation",
    "payment_processed",
    "booking_completed",
    "stopped_rejected",
    "conversation_management",
    "quote_presented",
    "quote_ready",
    "eligibility_check",
    "safety_check",
    "case_ready",
    "case_recorded",
    "workflow phase",
]
MODEL_GUARANTEE_PATTERNS = [
    "guaranteed model",
    "exact model guaranteed",
    "we guarantee the exact",
    "نضمن الموديل",
    "الموديل مضمون",
    "متوفر أكيد",
    "available now",
    "guaranteed availability",
    "availability guaranteed",
    "أكيد متوفرة",
    "متوفر الآن",
    "نضمن التوفر",
]
UNSUPPORTED_CASUAL_NICKNAME_PATTERNS = [
    "يا حلو",
    "يا بطل",
    "يا غالي",
]
OPERATIONAL_PROMISE_PATTERNS = [
    "will arrive within",
    "arrival guaranteed",
    "refund approved",
    "complaint approved",
    "سيصل خلال",
    "الوصول مضمون",
    "تمت الموافقة على الاسترداد",
    "تم قبول الشكوى",
]


@dataclass
class GuardResult:
    ok: bool
    failures: list[str] = field(default_factory=list)


def validate_customer_response(response: str, context: AnswerContext) -> GuardResult:
    lowered = response.lower()
    failures: list[str] = []

    if any(pattern in lowered for pattern in FORBIDDEN_DISCLOSURE_PATTERNS):
        failures.append("forbidden_demo_or_api_wording")

    if any(pattern in lowered for pattern in PAYMENT_SUCCESS_PATTERNS):
        if context.phase != "payment_processed" and context.phase != "booking_completed":
            failures.append("payment_success_before_confirmation")

    has_total_language = any(token in lowered for token in ["total", "الإجمالي", "المجموع"])
    has_money = bool(re.search(r"\b\d+(?:\.\d{1,2})?\s*(?:sar|ريال)", lowered))
    if has_total_language and has_money:
        if context.computed_totals is None or not context.computed_totals.values.get("totals_produced"):
            failures.append("totals_without_calculator_output")
    if has_money and context.computed_totals is None and not context.retrieved_records and not _numeric_strings(context.allowed_facts):
        failures.append("unsupported_price_claim")
    if has_money and not _money_values_are_allowed(response, context):
        failures.append("unsupported_price_claim")

    if any(re.search(pattern, response, flags=re.IGNORECASE) for pattern in INTERNAL_LEAK_PATTERNS):
        failures.append("internal_trace_or_kb_leak")

    stripped = response.strip()
    if stripped.startswith("{") or stripped.startswith("[") or '"intent"' in stripped or '"trace"' in stripped:
        failures.append("json_leakage")

    if any(pattern in lowered for pattern in MODEL_GUARANTEE_PATTERNS):
        failures.append("exact_model_guarantee")

    if any(pattern in lowered for pattern in OPERATIONAL_PROMISE_PATTERNS):
        failures.append("unsupported_operational_promise")

    if any(pattern in lowered for pattern in UNSUPPORTED_CASUAL_NICKNAME_PATTERNS):
        failures.append("unsupported_casual_nickname")

    return GuardResult(ok=not failures, failures=failures)


def _money_values_are_allowed(response: str, context: AnswerContext) -> bool:
    mentioned = {_normalise_money(match.group(1)) for match in re.finditer(r"\b(\d[\d,]*(?:\.\d{1,2})?)\s*(?:sar|ريال)", response, flags=re.IGNORECASE)}
    if not mentioned:
        return True
    allowed = set()
    if context.computed_totals:
        allowed.update(_numeric_strings(context.computed_totals.values))
    allowed.update(_numeric_strings(context.allowed_facts))
    if not allowed and context.retrieved_records:
        return True
    return mentioned.issubset(allowed)


def _numeric_strings(value) -> set[str]:
    values: set[str] = set()
    if isinstance(value, dict):
        for item in value.values():
            values.update(_numeric_strings(item))
    elif isinstance(value, list):
        for item in value:
            values.update(_numeric_strings(item))
    elif isinstance(value, (int, float)):
        values.add(_normalise_money(str(value)))
        if isinstance(value, float) and value.is_integer():
            values.add(_normalise_money(str(int(value))))
    return values


def _normalise_money(value: str) -> str:
    cleaned = str(value).replace(",", "")
    if "." in cleaned:
        cleaned = cleaned.rstrip("0").rstrip(".")
    return cleaned


def deterministic_safe_template(context: AnswerContext) -> str:
    if context.language == "en":
        if context.escalation_required:
            return "I’ll collect the needed details and hand this over to the relevant team."
        if context.computed_totals and context.computed_totals.values.get("totals_produced"):
            values = context.computed_totals.values
            return (
                "Quotation summary:\n"
                f"Online total: {values.get('total_online_sar', values.get('total_inc_vat_sar'))} SAR\n"
                f"In-branch total: {values.get('total_in_branch_sar', values.get('total_inc_vat_sar'))} SAR\n"
                "Models are representative within the category; final availability depends on branch and date."
            )
        return "Please share the missing details so I can continue."

    if context.escalation_required:
        return "سأجمع التفاصيل المطلوبة وأحوّل الطلب للفريق المختص."
    if context.computed_totals and context.computed_totals.values.get("totals_produced"):
        values = context.computed_totals.values
        return (
            "ملخص عرض السعر:\n"
            f"الإجمالي الإلكتروني: {values.get('total_online_sar', values.get('total_inc_vat_sar'))} ريال\n"
            f"إجمالي الفرع: {values.get('total_in_branch_sar', values.get('total_inc_vat_sar'))} ريال\n"
            "الموديلات المذكورة أمثلة ضمن الفئة، والتوفر النهائي يعتمد على الفرع والتاريخ."
        )
    if context.intent.value == "small_talk":
        return "أهلًا، أنا بخير. كيف أقدر أساعدك؟"
    return "فضلاً زودني بالتفاصيل الناقصة حتى أقدر أكمل."


def guard_or_fallback(response: str, context: AnswerContext) -> str:
    result = validate_customer_response(response, context)
    if result.ok:
        return response
    context.guard_failures.extend(result.failures)
    return deterministic_safe_template(context)
