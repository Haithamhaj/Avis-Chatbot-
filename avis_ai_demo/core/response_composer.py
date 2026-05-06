from __future__ import annotations

from typing import Any

from avis_ai_demo.core.persona_response_planner import build_persona_response_plan
from avis_ai_demo.core.rule_engine import category_availability_caveat
from avis_ai_demo.core.response_guard import guard_or_fallback
from avis_ai_demo.core.tone_lexicon import apply_optional_tone_phrase
from avis_ai_demo.core.types import AnswerContext, Intent, RetrievedRecord
from avis_ai_demo.services.openai_service import OpenAIService


def compose_response(
    context: AnswerContext,
    records: dict[str, Any] | None = None,
    service: OpenAIService | None = None,
) -> str:
    records = records or {}
    price_record = records.get("daily_price") or records.get("monthly_price")
    if price_record and not context.retrieved_records:
        record_id = price_record.get("price_id") or price_record.get("monthly_price_id") or price_record.get("category_id", "price_record")
        context.retrieved_records.append(RetrievedRecord(kb_id="pricing", record_id=record_id))
    if context.intent in {
        Intent.GREETING,
        Intent.SMALL_TALK,
        Intent.OFF_TOPIC,
        Intent.THANKS_ACKNOWLEDGEMENT,
        Intent.CAPABILITY_QUESTION,
        Intent.CLARIFICATION_REQUEST,
        Intent.POTENTIALLY_RELEVANT_UNCLEAR,
        Intent.SCOPE_REDIRECT,
        Intent.FALLBACK_UNKNOWN,
    }:
        base = _compose_conversation_management(context)
    elif context.intent in {Intent.COMPLAINT_OR_FINANCIAL_DISPUTE, Intent.COMPLAINT_OR_DISPUTE} or context.escalation_required:
        base = _compose_escalation(context)
    elif context.intent in {Intent.ROADSIDE_ASSISTANCE, Intent.ROADSIDE_OR_ACCIDENT}:
        base = _compose_roadside(context)
    elif context.computed_totals:
        base = _compose_quote(context)
    elif context.intent == Intent.BRANCH_LOOKUP:
        base = _compose_branch(context, records.get("branch"))
    elif context.intent == Intent.GENERAL_FAQ:
        base = _compose_general_faq(context, records.get("general_faq"))
    elif context.intent in {Intent.FLEET_PRICING, Intent.DAILY_RENTAL, Intent.MONTHLY_RENTAL}:
        base = _compose_fleet_or_price(context, records)
    else:
        base = "Please share more details." if context.language == "en" else "فضلاً زودني بتفاصيل أكثر."
    phrased = _phrase_with_khalid(base, context, records, service)
    phrased = apply_optional_tone_phrase(phrased, context)
    phrased = _ensure_contextual_opener(phrased, context)
    return guard_or_fallback(phrased, context)


def _phrase_with_khalid(
    base: str,
    context: AnswerContext,
    records: dict[str, Any],
    service: OpenAIService | None,
) -> str:
    if service is None:
        return base
    plan = build_persona_response_plan(context, records, deterministic_fallback=base)
    context.allowed_facts.update(plan.allowed_facts)
    try:
        phrased = service.phrase_khalid_response(plan.to_payload())
    except Exception as exc:
        context.guard_failures.append(f"persona_phrasing_failed:{type(exc).__name__}")
        return base
    return phrased.strip() if phrased and phrased.strip() else base


def _service_options(language: str) -> str:
    if language == "en":
        return "I can help with booking a car, daily or monthly prices, branches and hours, rental requirements and deposit policy, or roadside assistance."
    return "أقدر أساعدك في حجز سيارة، معرفة الأسعار اليومية أو الشهرية، معرفة الفروع وساعات العمل، شروط التأجير والوديعة، أو طلب مساعدة على الطريق."


def _compose_conversation_management(context: AnswerContext) -> str:
    if context.intent == Intent.GREETING:
        if context.language == "en":
            return "Hello, how can I help you?"
        return "مرحبًا، كيف أقدر أساعدك؟"

    if context.intent == Intent.SMALL_TALK:
        if context.language == "en":
            return "I’m doing well, thanks. How can I help you with Avis?"
        return "أهلًا، أنا بخير. كيف أقدر أساعدك؟"

    if context.intent == Intent.THANKS_ACKNOWLEDGEMENT:
        if context.language == "en":
            return "You’re welcome. If you need a booking, price, branch, or roadside assistance, I’m here."
        return "على الرحب والسعة. إذا احتجت حجز سيارة، سعر، فرع، أو مساعدة على الطريق أنا حاضر."

    if context.intent == Intent.OFF_TOPIC:
        if context.language == "en":
            return "That could turn into a long debate, so I’ll stay in my Avis lane. How can I help you with Avis?"
        return "سؤال يفتح باب طويل. الفريقين كبار، لكن خليني أبقى في منطقتي قبل لا أزعل أحد. كيف أقدر أساعدك مع أفيس؟"

    if context.intent == Intent.CAPABILITY_QUESTION:
        if context.language == "en":
            return _service_options("en")
        return _service_options("ar")

    if context.intent in {Intent.CLARIFICATION_REQUEST, Intent.POTENTIALLY_RELEVANT_UNCLEAR}:
        if context.allowed_facts.get("clarification_turn") == "followup":
            if context.language == "en":
                return "I get you. What made you feel that way? Tell me what happened, and I’ll point you to the right Avis support path."
            return "فهمتك. إيش اللي خلاك تاخذ هذا الانطباع؟ قلّي اللي صار، وأنا أوجّهك للمسار الأنسب في أفيس."
        if context.tone_mode == "serious_supportive" or context.customer_mood in {"frustrated", "angry", "upset"}:
            if context.language == "en":
                return "I hear you. To help properly, is this about a booking, payment, deposit, branch, or vehicle experience?"
            return "أفهم عليك. عشان أساعدك صح، هل الموضوع بخصوص حجز، مبلغ أو دفع، وديعة، فرع، أو تجربة سيارة؟"
        if context.language == "en":
            return "Sure. Are you looking to book a car, check a price, find a branch, or get roadside assistance?"
        return "أكيد. تقصد حجز سيارة، معرفة سعر، فرع، أو مساعدة على الطريق؟"

    if context.intent == Intent.SCOPE_REDIRECT:
        return _service_options(context.language)

    if context.language == "en":
        return f"I’m not sure what you mean yet. {_service_options('en')}"
    return f"ما فهمت طلبك بشكل كافي. {_service_options('ar')}"


def _compose_quote(context: AnswerContext) -> str:
    values = context.computed_totals.values
    if context.computed_totals.rental_type == "monthly":
        if context.language == "en":
            return (
                "Monthly quotation summary\n"
                f"Vehicle category: {values.get('category_id')}\n"
                f"Representative model: {values.get('model')}\n"
                f"Online monthly price: {values.get('online_monthly_price_sar')} SAR\n"
                f"With CDW: {values.get('with_cdw_sar')} SAR\n"
                f"Total including VAT: {values.get('total_inc_vat_sar')} SAR\n"
                "Monthly rentals require 48-hour advance booking and same-branch return."
            )
        return (
            "ملخص عرض السعر الشهري\n"
            f"فئة السيارة: {values.get('category_id')}\n"
            f"الموديل الممثل: {values.get('model')}\n"
            f"السعر الشهري الإلكتروني: {values.get('online_monthly_price_sar')} ريال\n"
            f"مع التأمين: {values.get('with_cdw_sar')} ريال\n"
            f"الإجمالي شامل الضريبة: {values.get('total_inc_vat_sar')} ريال\n"
            "التأجير الشهري يتطلب حجزًا قبل 48 ساعة وإرجاع السيارة لنفس الفرع."
        )
    models = ", ".join(values.get("models", []) or [values.get("model", "")]).strip(", ")
    if context.language == "en":
        if context.phase in {"payment_processed", "booking_completed"}:
            return "Payment completed successfully.\nBooking completed successfully.\nBooking number: AVIS-REQ-2026-001"
        return (
            "Quotation summary\n"
            f"Vehicle category: {values.get('category_id')}\n"
            f"Representative models: {models}\n"
            f"Rental days: {values.get('rental_days')}\n"
            f"Online daily price: {values.get('online_daily_price_sar')} SAR\n"
            f"In-branch daily price: {values.get('in_branch_daily_price_sar')} SAR\n"
            f"Online subtotal: {values.get('online_subtotal_sar')} SAR\n"
            f"In-branch subtotal: {values.get('in_branch_subtotal_sar')} SAR\n"
            f"One-way drop-off fee: {values.get('one_way_dropoff_fee_sar')} SAR\n"
            f"Total online amount: {values.get('total_online_sar')} SAR\n"
            f"Total in-branch amount: {values.get('total_in_branch_sar')} SAR\n"
            f"{category_availability_caveat('en')}\n"
            "Reply confirm or pay to continue."
        )
    if context.phase in {"payment_processed", "booking_completed"}:
        return "تم الدفع بنجاح.\nتم الحجز بنجاح.\nرقم الحجز: AVIS-REQ-2026-001"
    return (
        "ملخص عرض السعر\n"
        f"فئة السيارة: {values.get('category_id')}\n"
        f"الموديلات الممثلة: {models}\n"
        f"عدد الأيام: {values.get('rental_days')}\n"
        f"السعر اليومي الإلكتروني: {values.get('online_daily_price_sar')} ريال\n"
        f"السعر اليومي في الفرع: {values.get('in_branch_daily_price_sar')} ريال\n"
        f"المجموع الإلكتروني: {values.get('online_subtotal_sar')} ريال\n"
        f"مجموع الفرع: {values.get('in_branch_subtotal_sar')} ريال\n"
        f"رسوم التسليم في مدينة أخرى: {values.get('one_way_dropoff_fee_sar')} ريال\n"
        f"الإجمالي الإلكتروني: {values.get('total_online_sar')} ريال\n"
        f"إجمالي الفرع: {values.get('total_in_branch_sar')} ريال\n"
        f"{category_availability_caveat('ar')}\n"
        "اكتب أكد أو ادفع للمتابعة."
    )


def _compose_branch(context: AnswerContext, branch: dict[str, Any] | None) -> str:
    if not branch:
        return "Which branch or city?" if context.language == "en" else "أي فرع أو مدينة تقصد؟"
    if context.language == "en":
        return (
            f"{branch['branch_name_en']}\n"
            f"Hours: {branch['hours']}\n"
            f"Phone: {branch.get('phone', 'Not listed')}\n"
            f"Map: {branch.get('map_url', 'Not listed')}\n"
            f"{branch['caveat_en']}"
        )
    return (
        f"{branch['branch_name_ar']}\n"
        f"ساعات العمل: {branch['hours']}\n"
        f"الهاتف: {branch.get('phone', 'غير متوفر')}\n"
        f"الخريطة: {branch.get('map_url', 'غير متوفر')}\n"
        f"{branch['caveat_ar']}"
    )


def _compose_general_faq(context: AnswerContext, faq: dict[str, Any] | None) -> str:
    if not faq:
        return _service_options(context.language)
    answer = faq["answer_en"] if context.language == "en" else faq["answer_ar"]
    opener = _contextual_general_faq_opener(context)
    return f"{opener}\n\n{answer}" if opener else answer


def _contextual_general_faq_opener(context: AnswerContext) -> str:
    text = context.user_message.lower()
    if context.language == "en":
        if any(token in text for token in ["nice logo", "liked the logo", "love the logo"]):
            return "Glad it caught your eye."
        if any(token in text for token in ["saw your branch", "noticed your branch"]):
            return "Nice, sounds like the branch made an impression."
        return ""

    if any(token in text for token in ["عجبنا اللوغو", "عجبني اللوغو", "حلو اللوغو", "لوغوكم حلو", "الشعار عجبني", "الشعار عجبنا"]):
        return "يسعدنا أن اللوغو عجبكم."
    if any(token in text for token in ["فرعكم عجبنا", "عجبنا فرعكم", "شفنا واحد من فروعكم", "شفت فرعكم", "شفنا فرعكم"]):
        return "جميل أنه لفت انتباهكم."
    return ""


def _ensure_contextual_opener(response: str, context: AnswerContext) -> str:
    if context.intent != Intent.GENERAL_FAQ:
        return response
    opener = _contextual_general_faq_opener(context)
    if not opener or response.startswith(opener):
        return response
    return f"{opener}\n\n{response}"


def _compose_fleet_or_price(context: AnswerContext, records: dict[str, Any]) -> str:
    price = records.get("daily_price") or records.get("monthly_price")
    fleet = records.get("fleet")
    if context.language == "en":
        if price:
            return (
                f"Category: {price.get('classification')}\n"
                f"Online price: {price.get('online_total_vat_sar', price.get('online_price_sar'))} SAR\n"
                f"In-branch price: {price.get('in_branch_total_vat_sar', price.get('total_inc_vat_sar'))} SAR\n"
                f"{category_availability_caveat('en')}"
            )
        if fleet:
            return f"{fleet['customer_friendly_name_en']}. {fleet['model_guarantee_policy_en']}"
        if context.intent == Intent.DAILY_RENTAL and context.missing_fields:
            return "Sure. Which Riyadh branch do you prefer for pickup, how long do you need the car, and which vehicle category?"
        return "Which vehicle category or model do you mean?"
    if price:
        return (
            f"الفئة: {price.get('classification')}\n"
            f"السعر الإلكتروني: {price.get('online_total_vat_sar', price.get('online_price_sar'))} ريال\n"
            f"سعر الفرع: {price.get('in_branch_total_vat_sar', price.get('total_inc_vat_sar'))} ريال\n"
            f"{category_availability_caveat('ar')}"
        )
    if fleet:
        return f"{fleet['customer_friendly_name_ar']}. {fleet['model_guarantee_policy_ar']}"
    if context.intent == Intent.DAILY_RENTAL and context.missing_fields:
        return "أكيد. من أي فرع في الرياض تفضل الاستلام؟ وكم مدة الإيجار؟ وأي فئة سيارة تناسبك؟"
    return "أي فئة أو موديل تقصد؟"


def _compose_escalation(context: AnswerContext) -> str:
    if context.language == "en":
        return "I understand. I’ll prepare this correctly and hand it over to the relevant team. Please share the booking or rental agreement number, mobile number, and transaction date."
    return "أفهم عليك. خليني أجهز لك الطلب بالشكل الصحيح وأحوّله إلى الفريق المختص. أحتاج رقم الحجز أو العقد، رقم الجوال، وتاريخ العملية."


def _compose_roadside(context: AnswerContext) -> str:
    if context.language == "en":
        return "Your safety comes first. Are there any injuries or immediate danger? If there is danger, contact emergency services now. If it is safe, I can help record the assistance request."
    return "سلامتك أولًا. هل يوجد إصابات أو خطر مباشر؟ إذا فيه خطر، تواصل مع الطوارئ فورًا. إذا الوضع آمن، أقدر أساعدك بتسجيل طلب المساعدة."
