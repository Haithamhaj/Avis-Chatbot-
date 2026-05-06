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
    elif context.allowed_facts.get("quote_inclusion_answer"):
        base = _compose_quote_inclusion(context)
    elif context.allowed_facts.get("availability_caveat_only"):
        base = _compose_availability_caveat(context)
    elif context.allowed_facts.get("response_sections"):
        base = _compose_response_sections(context)
    elif context.intent == Intent.SERVICE_EXPERIENCE_FEEDBACK:
        base = _compose_service_feedback(context)
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
    elif context.intent == Intent.CARD_DEPOSIT_POLICY:
        base = _compose_card_deposit_policy(context, records.get("card_policy"))
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
        return _compose_greeting(context)

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
        if context.allowed_facts.get("ambiguity_type") == "daily_vs_monthly_or_comparison":
            if context.language == "en":
                return "To avoid mixing prices, do you want a daily quote or a monthly quote? I can compare only after we confirm the rental type and required details."
            return "عشان ما نخلط الأسعار، تقصد عرض يومي ولا شهري؟ أقدر أقارن لك بعد ما نحدد نوع التأجير والتفاصيل المطلوبة."
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


def _compose_greeting(context: AnswerContext) -> str:
    text = context.user_message.strip().lower()
    if context.language == "en":
        if text in {"hi", "hey"}:
            return "Hi, Khalid from Avis here. How can I help?"
        return "Hello, Khalid from Avis here. How can I help?"
    if "السلام" in text:
        return "وعليكم السلام، معك خالد من أفيس. كيف أقدر أساعدك اليوم؟"
    if text in {"هلا", "يا هلا"}:
        return "هلا فيك، معك خالد من أفيس. وش أقدر أرتب لك؟"
    if text in {"اهلا", "أهلا"}:
        return "أهلًا وسهلًا، معك خالد من أفيس. كيف أقدر أساعدك؟"
    return "مرحبًا، معك خالد من أفيس. كيف أقدر أساعدك؟"


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


def _compose_card_deposit_policy(context: AnswerContext, policy: dict[str, Any] | None) -> str:
    if not policy:
        return "Which card or deposit question do you mean?" if context.language == "en" else "تقصد سؤال عن البطاقة أو الوديعة؟"
    answer = policy["answer_en"] if context.language == "en" else policy["answer_ar"]
    if context.language == "en":
        return f"{answer}\n\nIf you were in the middle of a booking, I can continue from where we stopped."
    return f"{answer}\n\nوإذا كنا في وسط الحجز، أقدر أكمل معك من نفس النقطة."


def _compose_service_feedback(context: AnswerContext) -> str:
    case = context.allowed_facts.get("support_case") or {}
    has_mobile = bool(case.get("mobile"))
    has_branch = bool(case.get("branch") or case.get("branch_or_city"))
    if context.language == "en":
        if has_mobile and has_branch:
            return "I understand, and thank you for the details. I’ll pass the branch experience feedback to the right team. If you have a booking or rental agreement number, share it too so they can review faster."
        return "I understand. To route this branch experience feedback properly, please share the branch or city, approximate visit time, mobile number, and a short description of what happened. Booking or agreement number is optional if you have it."
    if has_mobile and has_branch:
        return "أفهم عليك، وشكرًا على التفاصيل. سأوصل ملاحظة تجربة الفرع للفريق المناسب. إذا عندك رقم حجز أو عقد أرسله أيضًا عشان تكون المراجعة أسرع."
    return "أفهم عليك، وتجربتك تهمنا. عشان أوصل ملاحظة الفرع للفريق الصحيح، أحتاج اسم الفرع أو المدينة، وقت الزيارة تقريبًا، رقم جوالك، ووصف مختصر للي صار. رقم الحجز أو العقد اختياري إذا متوفر."


def _compose_response_sections(context: AnswerContext) -> str:
    sections = context.allowed_facts.get("response_sections") or []
    if not sections:
        return _service_options(context.language)
    lines: list[str] = []
    for section in sections:
        if section.get("type") == "branch_info":
            branch = section.get("branch")
            if branch:
                if context.language == "en":
                    lines.append(f"Branch: {branch['branch_name_en']}\\nHours: {branch['hours']}\\nPhone: {branch.get('phone', 'Not listed')}")
                else:
                    lines.append(f"نعم، عندكم {branch['branch_name_ar']}.\\nساعات العمل: {branch['hours']}\\nالهاتف: {branch.get('phone', 'غير متوفر')}")
        elif section.get("type") == "price_info":
            price = section.get("daily_price")
            if price:
                if context.language == "en":
                    lines.append(
                        f"Approximate daily category price: online {price.get('online_total_vat_sar', price.get('online_price_sar'))} SAR, in-branch {price.get('in_branch_total_vat_sar')} SAR. {category_availability_caveat('en')}"
                    )
                else:
                    lines.append(
                        f"وبالنسبة للسعر التقريبي للفئة: الإلكتروني {price.get('online_total_vat_sar', price.get('online_price_sar'))} ريال، وسعر الفرع {price.get('in_branch_total_vat_sar')} ريال. {category_availability_caveat('ar')}"
                    )
        elif section.get("type") == "next_step":
            if context.language == "en":
                lines.append("If you want a full quote, share the rental duration, pickup date and time, and return city.")
            else:
                lines.append("إذا تبغى عرض سعر كامل، أرسل مدة الإيجار ووقت الاستلام ومدينة الإرجاع.")
        elif section.get("type") == "card_policy":
            policy = section.get("card_policy")
            if policy:
                lines.append(policy["answer_en"] if context.language == "en" else policy["answer_ar"])
    return "\n\n".join(lines)


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
    if any(token in text for token in ["بحب ايفيس", "بحب أفيس", "احب ايفيس", "أحب أفيس", "هذا الحب", "مقابل هذا الحب"]):
        return "وصلت المحبة، وهذا شيء نعتز فيه."
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
            return _daily_missing_fields_prompt(context, "en")
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
        return _daily_missing_fields_prompt(context, "ar")
    return "أي فئة أو موديل تقصد؟"


def _compose_availability_caveat(context: AnswerContext) -> str:
    if context.language == "en":
        return "I can’t guarantee an exact model, color, or unit from chat. I can price the matching category and note your preference, while final availability is confirmed by branch and date."
    return "ما أقدر أضمن موديل أو لون أو سيارة محددة من المحادثة. أقدر أعطيك سعر الفئة المناسبة وأسجل تفضيلك، والتوفر النهائي يتأكد حسب الفرع والتاريخ."


def _compose_quote_inclusion(context: AnswerContext) -> str:
    if context.language == "en":
        return "The shown quote is based on the calculator values available in this chat. It includes the listed rental price and any listed one-way fee if applicable. Final branch checks and optional extras are confirmed before payment."
    return "عرض السعر المعروض مبني على الحاسبة المتاحة في المحادثة. يشمل سعر الإيجار المذكور وأي رسوم تسليم بين المدن إذا كانت ظاهرة في العرض. أي إضافات اختيارية أو تحقق نهائي من الفرع يتأكد قبل الدفع."


def _daily_missing_fields_prompt(context: AnswerContext, language: str) -> str:
    missing = set(context.missing_fields)
    if "has_valid_license" in missing:
        if language == "en":
            return "Do you have a valid driving license? Also, which pickup branch do you prefer?"
        return "هل لديك رخصة قيادة سارية المفعول؟ ومن أي فرع تفضل الاستلام؟ وكم مدة الإيجار؟"
    questions_ar = {
        "pickup_city": "مدينة الاستلام",
        "dropoff_city": "مدينة الإرجاع",
        "rental_days": "مدة الإيجار",
        "pickup_date": "تاريخ الاستلام",
        "pickup_time": "وقت الاستلام",
        "vehicle_query": "فئة السيارة أو الموديل",
    }
    questions_en = {
        "pickup_city": "pickup city",
        "dropoff_city": "return city",
        "rental_days": "rental duration",
        "pickup_date": "pickup date",
        "pickup_time": "pickup time",
        "vehicle_query": "vehicle category or model",
    }
    labels = [questions_en[field] for field in context.missing_fields if field in questions_en] if language == "en" else [questions_ar[field] for field in context.missing_fields if field in questions_ar]
    if not labels:
        return "Please share the missing booking details." if language == "en" else "أرسل لي تفاصيل الحجز الناقصة."
    joined = ", ".join(labels) if language == "en" else "، ".join(labels)
    if language == "en":
        return f"Sure. To prepare the quote, please share: {joined}."
    return f"أكيد. عشان أجهز عرض السعر، أحتاج: {joined}."


def _compose_escalation(context: AnswerContext) -> str:
    case = context.allowed_facts.get("support_case") or {}
    if case.get("has_minimum_details"):
        if context.language == "en":
            return "I received the booking/agreement details, mobile number, and transaction date. I’ll prepare the case for the relevant team. If you have a bank notification image or transaction reference, share it to speed up the review."
        return "وصلتني تفاصيل الحجز أو العقد ورقم الجوال وتاريخ العملية. سأجهّز الطلب للفريق المختص. إذا عندك صورة الإشعار البنكي أو مرجع العملية أرسلها لتسريع المراجعة."
    if context.language == "en":
        return "I understand. I’ll prepare this correctly and hand it over to the relevant team. Please share the booking or rental agreement number, mobile number, and transaction date."
    return "أفهم عليك. خليني أجهز لك الطلب بالشكل الصحيح وأحوّله إلى الفريق المختص. أحتاج رقم الحجز أو العقد، رقم الجوال، وتاريخ العملية."


def _compose_roadside(context: AnswerContext) -> str:
    if context.language == "en":
        return "Your safety comes first. Are there any injuries or immediate danger? If there is danger, contact emergency services now. If it is safe, I can help record the assistance request."
    if context.allowed_facts.get("drive_advice"):
        return "سلامتك أولًا. إذا السيارة ترجف أو غير مطمئن تمشي فيها، الأفضل توقف في مكان آمن ولا تكمل عليها قبل ما تتواصل مع المساعدة. هل يوجد إصابات أو خطر مباشر؟ وإذا الوضع آمن أرسل رقم الجوال ورقم اللوحة أو العقد وموقعك."
    return "سلامتك أولًا. هل يوجد إصابات أو خطر مباشر؟ إذا فيه خطر، تواصل مع الطوارئ فورًا. إذا الوضع آمن، أقدر أساعدك بتسجيل طلب المساعدة."
