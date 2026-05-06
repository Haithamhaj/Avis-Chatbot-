from __future__ import annotations

from dataclasses import asdict
import re
from typing import Any

from avis_ai_demo.core.calculators import calculate_monthly_quote, daily_price_summary
from avis_ai_demo.core.confirmation import is_confirmation
from avis_ai_demo.core.conversation_decision import decide_conversation
from avis_ai_demo.core.conversation_manager import classify_conversation, is_conversation_only
from avis_ai_demo.core.conversation_history import compact_history_for_ai, record_conversation_turn
from avis_ai_demo.core.data_lookup import (
    find_branch,
    find_card_deposit_policy,
    find_daily_price,
    find_fleet_category,
    find_general_faq,
    find_monthly_price,
    match_escalation,
)
from avis_ai_demo.core.entity_extractor import deterministic_parse, extract_with_gpt_or_fallback
from avis_ai_demo.core.intent_router import route_intent
from avis_ai_demo.core.language import detect_language
from avis_ai_demo.core.response_composer import compose_response
from avis_ai_demo.core.response_guard import guard_or_fallback
from avis_ai_demo.core.semantic_router import semantic_route_candidates, should_use_semantic_candidate, SemanticRouteCandidate
from avis_ai_demo.core.types import AnswerContext, DailyRentalPhase, Intent, RoadsidePhase, WorkflowState
from avis_ai_demo.core.vector_search import semantic_kb_search
from avis_ai_demo.core.workflow_state import (
    advance_daily_workflow,
    advance_roadside_workflow,
    apply_payment_confirmation,
    complete_booking_after_payment,
    mark_quote_presented,
    transition_roadside_phase,
)
from avis_ai_demo.core.workflow_gate import evaluate_workflow_gate
from avis_ai_demo.services.mock_external_services import (
    create_booking_request,
    create_roadside_case,
    handover_to_agent,
    process_payment,
)
from avis_ai_demo.services.openai_service import OpenAIService


def handle_message(
    state: WorkflowState | None,
    message: str,
    service: OpenAIService | None = None,
) -> tuple[str, WorkflowState, dict[str, Any]]:
    state = state or WorkflowState()
    previous_intent = state.intent
    language = detect_language(message, previous_language=state.language)
    service = service or OpenAIService()
    decision = decide_conversation(message, service=service, state=asdict(state))
    gate = evaluate_workflow_gate(decision)
    conversation = classify_conversation(message, service=service, state=asdict(state))
    if is_conversation_only(conversation.intent) and not gate.allow_operational and not (
        is_confirmation(message, state.phase)
        and state.phase == DailyRentalPhase.AWAITING_PAYMENT_CONFIRMATION.value
    ):
        state.language = language
        state.intent = conversation.intent
        state.next_action = conversation.next_action
        state.tone_mode = conversation.tone_mode
        state.customer_mood = conversation.customer_mood
        context = AnswerContext(
            intent=conversation.intent,
            language=language,
            phase=conversation.phase,
            tone_mode=conversation.tone_mode,
            customer_mood=conversation.customer_mood,
            next_action=conversation.next_action,
            user_message=message,
        )
        context.allowed_facts["previous_intent"] = previous_intent.value
        if (
            previous_intent == Intent.POTENTIALLY_RELEVANT_UNCLEAR
            and conversation.intent == Intent.POTENTIALLY_RELEVANT_UNCLEAR
        ):
            context.allowed_facts["clarification_turn"] = "followup"
        _with_memory(context, state)
        response = compose_response(context, service=service)
        trace = {
            "intent": conversation.intent.value,
            "language": language,
            "phase": conversation.phase,
            "tone_mode": conversation.tone_mode,
            "customer_mood": conversation.customer_mood,
            "classifier_confidence": conversation.confidence,
            "entities": dict(state.entities),
            "missing_fields": list(state.missing_fields),
            "kb_modules_used": [],
            "lookup_records": [],
            "computed_totals": state.quote.values if state.quote else None,
            "risk_level": state.risk_level,
            "next_action": state.next_action,
            "escalation_status": False,
            "mock_action_results": [],
            "guard_failures": context.guard_failures,
            "gpt_fallback_used": False,
            "gpt_error": None,
            "conversation_decision": decision.to_trace(),
            "workflow_gate": gate.to_trace(),
            "conversation_history": {
                "summary": state.conversation_summary,
                "turn_count": len(state.conversation_turns),
            },
        }
        record_conversation_turn(state, user_message=message, assistant_response=response, trace=trace)
        trace["conversation_history"] = {
            "summary": state.conversation_summary,
            "turn_count": len(state.conversation_turns),
        }
        return response, state, trace

    extraction, used_fallback, extraction_error = extract_with_gpt_or_fallback(
        message, asdict(state), service=service
    )
    deterministic_extraction = deterministic_parse(message, previous_language=state.language)
    if extraction.get("intent") == Intent.FALLBACK_UNKNOWN.value and deterministic_extraction.get("intent") != Intent.FALLBACK_UNKNOWN.value:
        extraction["intent"] = deterministic_extraction["intent"]
    extraction.setdefault("entities", {})
    for key, value in deterministic_extraction.get("entities", {}).items():
        extraction["entities"].setdefault(key, value)
    if extraction.get("language") in {"ar", "en"} and message.strip().lower() not in {"yes", "no", "ok", "thanks", "نعم", "لا", "تمام"}:
        language = extraction["language"]
    state.language = language
    _apply_contextual_entities(state, message, extraction)

    route_candidates = semantic_route_candidates(message, service) if _should_check_semantic_routes(message, extraction, conversation.intent) else []
    selected_route = route_candidates[0] if route_candidates else None
    semantic_route_used = False

    if _is_rental_type_ambiguous(message, state):
        intent = Intent.POTENTIALLY_RELEVANT_UNCLEAR
    elif _is_quote_inclusion_question(message, state):
        intent = Intent.FLEET_PRICING
    elif _is_availability_guarantee_question(message, state):
        intent = Intent.FLEET_PRICING
    elif is_confirmation(message, state.phase) and state.phase == DailyRentalPhase.AWAITING_PAYMENT_CONFIRMATION.value:
        intent = Intent.DAILY_RENTAL
    elif gate.allow_operational and gate.intent in {Intent.GENERAL_FAQ, Intent.CARD_DEPOSIT_POLICY, Intent.COMPLAINT_OR_DISPUTE, Intent.SERVICE_EXPERIENCE_FEEDBACK, Intent.ROADSIDE_OR_ACCIDENT}:
        intent = gate.intent
    elif conversation.intent in {Intent.COMPLAINT_OR_DISPUTE, Intent.SERVICE_EXPERIENCE_FEEDBACK, Intent.ROADSIDE_OR_ACCIDENT}:
        intent = conversation.intent
    else:
        intent = route_intent(message, extraction.get("intent"))
        if (
            not gate.allow_operational
            and decision.confidence >= 0.62
            and intent in {Intent.DAILY_RENTAL, Intent.MONTHLY_RENTAL, Intent.ROADSIDE_ASSISTANCE, Intent.COMPLAINT_OR_FINANCIAL_DISPUTE}
        ):
            intent = gate.intent
        semantic_route_used = should_use_semantic_candidate(intent, selected_route)
        if semantic_route_used:
            intent = selected_route.target_intent
    state.intent = intent
    state.tone_mode = _tone_for_intent(intent, conversation.tone_mode)
    state.customer_mood = conversation.customer_mood

    trace: dict[str, Any] = {
        "intent": intent.value,
        "language": language,
        "phase": state.phase,
        "tone_mode": state.tone_mode,
        "customer_mood": state.customer_mood,
        "classifier_confidence": conversation.confidence,
        "entities": dict(state.entities),
        "missing_fields": [],
        "kb_modules_used": [],
        "lookup_records": [],
        "computed_totals": None,
        "risk_level": state.risk_level,
        "next_action": state.next_action,
        "escalation_status": False,
        "mock_action_results": [],
        "guard_failures": [],
        "gpt_fallback_used": used_fallback,
        "gpt_error": extraction_error,
        "semantic_route_candidates": [candidate.to_trace() for candidate in route_candidates],
        "semantic_route_selected": selected_route.to_trace() if semantic_route_used else None,
        "conversation_decision": decision.to_trace(),
        "workflow_gate": gate.to_trace(),
        "final_route": {},
        "corrections_applied": [],
    }

    if intent == Intent.DAILY_RENTAL:
        response = _handle_daily(state, message, extraction, trace, service)
    elif intent == Intent.MONTHLY_RENTAL:
        response = _handle_monthly(state, extraction, trace, service)
    elif intent == Intent.BRANCH_LOOKUP:
        if _looks_like_mixed_branch_price(message):
            response = _handle_mixed_branch_price(state, extraction, message, trace, service)
            trace["intent"] = Intent.FLEET_PRICING.value
            state.intent = Intent.FLEET_PRICING
        else:
            response = _handle_branch(state, extraction, message, trace, service)
    elif intent == Intent.GENERAL_FAQ:
        response = _handle_general_faq(state, message, trace, service, route_candidate=selected_route)
    elif intent == Intent.CARD_DEPOSIT_POLICY:
        if _looks_like_mixed_branch_card_quote(message, state):
            response = _handle_mixed_card_quote(state, extraction, message, trace, service)
            trace["intent"] = Intent.CARD_DEPOSIT_POLICY.value
        else:
            response = _handle_card_deposit_policy(state, message, trace, service)
    elif intent == Intent.FLEET_PRICING:
        if _is_quote_inclusion_question(message, state):
            response = _handle_quote_inclusion(state, message, trace, service)
        elif _is_availability_guarantee_question(message, state):
            response = _handle_availability_caveat(state, message, trace, service)
        elif _looks_like_mixed_branch_price(message):
            response = _handle_mixed_branch_price(state, extraction, message, trace, service)
        else:
            response = _handle_fleet_price(state, extraction, message, trace, service)
    elif intent in {Intent.ROADSIDE_ASSISTANCE, Intent.ROADSIDE_OR_ACCIDENT}:
        response = _handle_roadside(state, extraction, trace, service)
    elif intent in {Intent.COMPLAINT_OR_FINANCIAL_DISPUTE, Intent.COMPLAINT_OR_DISPUTE}:
        response = _handle_escalation(state, message, trace, service)
    elif intent == Intent.SERVICE_EXPERIENCE_FEEDBACK:
        response = _handle_service_feedback(state, message, trace, service)
    else:
        semantic_results = []
        if intent == Intent.FALLBACK_UNKNOWN and _looks_like_avis_adjacent(message):
            semantic_results = semantic_kb_search(message, service)
        if semantic_results:
            state.intent = Intent.GENERAL_FAQ
            state.tone_mode = "professional_helpful"
            response = _handle_general_faq(state, message, trace, service, semantic_results[0])
            trace["intent"] = Intent.GENERAL_FAQ.value
        else:
            if intent == Intent.OFF_TOPIC:
                state.next_action = Intent.SCOPE_REDIRECT.value
            elif intent == Intent.CLARIFICATION_REQUEST:
                state.next_action = "clarify"
            elif intent == Intent.POTENTIALLY_RELEVANT_UNCLEAR and _is_rental_type_ambiguous(message, state):
                state.next_action = "clarify_rental_type"
            elif intent in {Intent.GREETING, Intent.THANKS_ACKNOWLEDGEMENT, Intent.SMALL_TALK}:
                state.next_action = "conversation_acknowledgement"
            elif intent == Intent.FALLBACK_UNKNOWN:
                state.next_action = "fallback_with_service_options"
            context = AnswerContext(
                intent=intent,
                language=language,
                phase=state.phase,
                tone_mode=state.tone_mode,
                customer_mood=state.customer_mood,
                next_action=state.next_action,
                user_message=message,
            )
            if intent == Intent.POTENTIALLY_RELEVANT_UNCLEAR and _is_rental_type_ambiguous(message, state):
                context.allowed_facts["ambiguity_type"] = "daily_vs_monthly_or_comparison"
            _with_memory(context, state)
            response = compose_response(context, service=service)
            trace["guard_failures"] = context.guard_failures

    trace.update(
        {
            "phase": state.phase,
            "tone_mode": state.tone_mode,
            "customer_mood": state.customer_mood,
            "entities": dict(state.entities),
            "missing_fields": list(state.missing_fields),
            "computed_totals": state.quote.values if state.quote else trace.get("computed_totals"),
            "risk_level": state.risk_level,
            "next_action": state.next_action,
            "guard_failures": list(state.guard_failures or trace.get("guard_failures", [])),
            "conversation_history": {
                "summary": state.conversation_summary,
                "turn_count": len(state.conversation_turns),
            },
            "final_route": trace.get("final_route") or {
                "intent": state.intent.value,
                "source": "deterministic_router",
                "reason": "resolved_after_conversation_decision",
            },
        }
    )
    record_conversation_turn(state, user_message=message, assistant_response=response, trace=trace)
    trace["conversation_history"] = {
        "summary": state.conversation_summary,
        "turn_count": len(state.conversation_turns),
    }
    return response, state, trace


def _tone_for_intent(intent: Intent, default: str = "professional_helpful") -> str:
    if intent in {Intent.COMPLAINT_OR_DISPUTE, Intent.COMPLAINT_OR_FINANCIAL_DISPUTE, Intent.CARD_DEPOSIT_POLICY, Intent.SERVICE_EXPERIENCE_FEEDBACK}:
        return "serious_supportive"
    if intent in {Intent.ROADSIDE_OR_ACCIDENT, Intent.ROADSIDE_ASSISTANCE}:
        return "safety_first"
    if intent == Intent.OFF_TOPIC:
        return "light_deflection"
    if intent in {Intent.GREETING, Intent.SMALL_TALK, Intent.THANKS_ACKNOWLEDGEMENT}:
        return "friendly_casual"
    return default or "professional_helpful"


def _with_memory(context: AnswerContext, state: WorkflowState) -> AnswerContext:
    context.allowed_facts["conversation_memory"] = compact_history_for_ai(state)
    if state.support_case:
        context.allowed_facts["support_case"] = dict(state.support_case)
    return context


def _apply_contextual_entities(state: WorkflowState, message: str, extraction: dict[str, Any]) -> None:
    entities = extraction.setdefault("entities", {})
    text = message.lower()
    if _is_safety_followup(text, state):
        extraction["intent"] = Intent.ROADSIDE_ASSISTANCE.value
        state.last_topic = "roadside_safety"
    if _is_international_support_case(text):
        extraction["intent"] = Intent.COMPLAINT_OR_FINANCIAL_DISPUTE.value
        state.last_topic = "international_support"
    if state.last_topic == "international_support" and any(token in text for token in ["العقد", "رقم العقد", "جواب نهائي", "قرار نهائي", "final answer", "contract"]):
        extraction["intent"] = Intent.COMPLAINT_OR_FINANCIAL_DISPUTE.value
    if _is_service_vehicle_feedback(text):
        extraction["intent"] = Intent.SERVICE_EXPERIENCE_FEEDBACK.value
    if _is_correction_message(text):
        entities["is_correction"] = True
    if entities.get("is_correction"):
        state.entities.update({key: value for key, value in entities.items() if value not in (None, "") and key != "is_correction"})
    if state.intent in {Intent.DAILY_RENTAL, Intent.FLEET_PRICING} and _is_topic_shift_policy_question(text):
        state.suspended_workflow = state.intent.value
        state.last_topic = "card_deposit_policy"
        extraction["intent"] = Intent.CARD_DEPOSIT_POLICY.value
    if state.intent == Intent.CARD_DEPOSIT_POLICY and any(token in text for token in ["ما ينفع", "ماينفع", "طريقة ثانية", "بديل", "another way", "alternative"]):
        extraction["intent"] = Intent.CARD_DEPOSIT_POLICY.value
    if state.intent in {Intent.COMPLAINT_OR_FINANCIAL_DISPUTE, Intent.COMPLAINT_OR_DISPUTE} and any(key in entities for key in ["mobile", "booking_or_contract", "plate_or_contract", "transaction_date"]):
        extraction["intent"] = Intent.COMPLAINT_OR_FINANCIAL_DISPUTE.value
    if state.intent in {Intent.COMPLAINT_OR_FINANCIAL_DISPUTE, Intent.COMPLAINT_OR_DISPUTE} and any(token in text for token in ["جواب نهائي", "قرار نهائي", "رقم العقد", "العقد"]):
        extraction["intent"] = Intent.COMPLAINT_OR_FINANCIAL_DISPUTE.value
    if state.intent == Intent.SERVICE_EXPERIENCE_FEEDBACK and any(token in text for token in ["فرع", "الموظف", "زحمة", "تأخر", "تاخر", "جوال", "زيارة"]):
        extraction["intent"] = Intent.SERVICE_EXPERIENCE_FEEDBACK.value
    if state.intent == Intent.FLEET_PRICING and any(key in entities for key in ["pickup_city", "pickup_date", "rental_days"]):
        previous_vehicle = state.entities.get("vehicle_query")
        if previous_vehicle and "vehicle_query" not in entities:
            entities["vehicle_query"] = previous_vehicle
        extraction["intent"] = Intent.DAILY_RENTAL.value
    if _is_rental_type_ambiguous(message, state):
        state.last_topic = "rental_type_ambiguity"
    if _has_daily_quote_candidate(entities) and not _is_rental_type_ambiguous(message, state):
        extraction["intent"] = Intent.DAILY_RENTAL.value


def _is_topic_shift_policy_question(text: str) -> bool:
    shift = any(token in text for token in ["قبلها", "طيب سؤال", "بس سؤال", "خلني اسأل", "خلني أسأل", "before that", "quick question"])
    policy = any(token in text for token in ["وديعة", "deposit", "بطاقة", "card", "ائتمان", "credit"])
    return policy and (shift or not any(token in text for token in ["احجز", "book it", "أكد", "confirm"]))


def _is_correction_message(text: str) -> bool:
    return any(token in text for token in ["لا قصدي", "اقصد", "أقصد", "صحح", "خلها", "بدل", "غير"])


def _is_safety_followup(text: str, state: WorkflowState) -> bool:
    signal = any(token in text for token in ["ترجف", "تطفي", "تطفى", "أمشي ولا أوقف", "امشي ولا اوقف", "أكمل عليها", "اكمل عليها", "لازم أوقف", "safe to drive", "continue driving"])
    sticky = state.intent in {Intent.ROADSIDE_ASSISTANCE, Intent.ROADSIDE_OR_ACCIDENT} or state.last_topic == "roadside_safety"
    return signal or (sticky and any(token in text for token in ["الفرع", "تمشي", "أوقف", "اوقف", "أكمل", "اكمل"]))


def _is_international_support_case(text: str) -> bool:
    international = any(token in text for token in ["دبي", "خارج السعودية", "محطة خارجية", "outside saudi", "dubai", "international rental", "foreign station"])
    issue = any(token in text for token in ["وديعة", "deposit", "refund", "معلقة", "معلق", "العقد", "contract"])
    return international and issue


def _is_service_vehicle_feedback(text: str) -> bool:
    vehicle_or_service = any(token in text for token in ["ريحتها دخان", "ريحة دخان", "دخان", "السيارة كانت", "الفرع قال", "قال عادي", "رفعت ضغطي", "خدمتكم", "تجربة"])
    nonfinancial = not any(token in text for token in ["انخصم", "وديعة ما رجعت", "refund", "wrong charge", "double charge"])
    return vehicle_or_service and nonfinancial and any(token in text for token in ["دخان", "رفع", "ضغطي", "عادي", "سيئة", "مو فلوس"])


def _is_rental_type_ambiguous(message: str, state: WorkflowState | None = None) -> bool:
    text = message.lower()
    monthly = any(token in text for token in ["شهر", "شهري", "monthly"])
    daily = any(token in text for token in ["يوم", "يومين", "daily", "days"])
    compare = any(token in text for token in ["حسب السعر", "أرخص", "ارخص", "قارن", "compare", "cheaper"])
    sticky = state is not None and state.last_topic == "rental_type_ambiguity" and monthly and compare
    return (monthly and daily and compare) or sticky


def _is_availability_guarantee_question(message: str, state: WorkflowState) -> bool:
    text = message.lower()
    guarantee = any(token in text for token in ["تضمن", "ضمان", "بالضبط", "أكيد موجود", "متوفرة أكيد", "guarantee", "exact", "available for sure"])
    context = state.intent == Intent.FLEET_PRICING or bool(state.entities.get("vehicle_query"))
    return guarantee and context


def _is_quote_inclusion_question(message: str, state: WorkflowState) -> bool:
    text = message.lower()
    asks_inclusion = any(token in text for token in ["يشمل", "شامل", "قبل أدفع", "قبل ادفع", "include", "included", "before i pay"])
    return asks_inclusion and state.quote is not None


def _has_daily_quote_candidate(entities: dict[str, Any]) -> bool:
    return all(entities.get(field) for field in ["vehicle_query", "pickup_city", "dropoff_city", "rental_days", "pickup_date", "pickup_time"]) and entities.get("has_valid_license") is True


def _looks_like_avis_adjacent(message: str) -> bool:
    text = message.lower()
    markers = [
        "avis",
        "افيس",
        "أفيس",
        "تأجير",
        "ايجار",
        "إيجار",
        "سيارات",
        "سيارة",
        "موتر",
        "سائق",
        "سواق",
        "شركة",
        "شركات",
        "حجز",
    ]
    return any(marker.lower() in text for marker in markers)


def _looks_like_mixed_branch_price(message: str) -> bool:
    text = message.lower()
    branch = any(token in text for token in ["فرع", "branch", "المطار", "airport", "هناك", "قريب"])
    price_or_vehicle = any(token in text for token in ["سعر", "كم", "price", "كامري", "camry", "يارس", "yaris"])
    return branch and price_or_vehicle


def _looks_like_mixed_branch_card_quote(message: str, state: WorkflowState) -> bool:
    text = message.lower()
    has_card = any(token in text for token in ["بطاقة", "مدى", "mada", "debit", "credit"])
    has_return_or_duration = any(token in text for token in ["أرجعها", "ارجعها", "إرجاع", "ارجاع", "بعد يومين", "يومين", "return"])
    has_vehicle_context = bool(state.entities.get("vehicle_query")) or any(token in text for token in ["كامري", "camry", "يارس", "yaris"])
    return has_card and has_return_or_duration and has_vehicle_context


def _merge_support_case_details(state: WorkflowState, message: str) -> None:
    text = message.lower()
    mobile = re.search(r"(05[0-9]{8})", text)
    if mobile:
        state.support_case["mobile"] = mobile.group(1)
    reference = re.search(r"\b(?:av|ra)[a-z0-9-]*\d+\b", text, flags=re.IGNORECASE)
    if reference:
        state.support_case["booking_or_contract"] = reference.group(0).upper()
    if any(token in text for token in ["أمس", "امس", "yesterday"]):
        state.support_case["transaction_date"] = "yesterday"
    date = re.search(r"\b(20[0-9]{2}-[0-9]{2}-[0-9]{2})\b", text)
    if date:
        state.support_case["transaction_date"] = date.group(1)
    if any(token in text for token in ["فرع المطار", "المطار", "airport"]):
        state.support_case["branch_or_city"] = "airport branch"
    elif "فرع" in text:
        state.support_case["branch_or_city"] = "branch mentioned"
    if message.strip():
        state.support_case["summary"] = message.strip()
    state.support_case["has_minimum_details"] = all(
        state.support_case.get(field)
        for field in ["booking_or_contract", "mobile", "transaction_date"]
    )


def _should_check_semantic_routes(message: str, extraction: dict[str, Any], conversation_intent: Intent) -> bool:
    if conversation_intent in {Intent.COMPLAINT_OR_DISPUTE, Intent.ROADSIDE_OR_ACCIDENT}:
        return False
    if extraction.get("intent") == Intent.FALLBACK_UNKNOWN.value:
        return True
    return _looks_like_avis_adjacent(message)


def _handle_daily(
    state: WorkflowState,
    message: str,
    extraction: dict[str, Any],
    trace: dict[str, Any],
    service: OpenAIService | None = None,
) -> str:
    if is_confirmation(message, state.phase):
        state = apply_payment_confirmation(state, message)
        payment_result = process_payment(state.quote, confirmed=state.payment_confirmed)
        booking_result = create_booking_request(state.quote, state.entities)
        complete_booking_after_payment(state)
        trace["mock_action_results"].extend([asdict(payment_result), asdict(booking_result)])
        context = AnswerContext(
            intent=Intent.DAILY_RENTAL,
            language=state.language,
            computed_totals=state.quote,
            phase=state.phase,
            tone_mode=state.tone_mode,
            customer_mood=state.customer_mood,
        )
        _with_memory(context, state)
        response = compose_response(context, service=service)
        state.guard_failures.extend(context.guard_failures)
        return response

    entities = extraction.get("entities", {})
    if entities.get("is_correction"):
        trace["corrections_applied"] = [key for key in ("pickup_city", "dropoff_city", "pickup_date", "pickup_time", "rental_days", "vehicle_query") if key in entities]
    advance_daily_workflow(state, entities)
    state.last_route_entities = {key: state.entities.get(key) for key in ("pickup_city", "dropoff_city", "pickup_date", "pickup_time", "rental_days", "vehicle_query") if state.entities.get(key)}
    trace["kb_modules_used"].extend(["KB03", "KB05"])
    if state.quote:
        trace["lookup_records"].append(state.quote.values["price_id"])
        mark_quote_presented(state)
    context = AnswerContext(
        intent=Intent.DAILY_RENTAL,
        language=state.language,
        computed_totals=state.quote,
        missing_fields=state.missing_fields,
        phase=state.phase,
        tone_mode=state.tone_mode,
        customer_mood=state.customer_mood,
        next_action=state.next_action,
        user_message=message,
    )
    _with_memory(context, state)
    response = compose_response(context, service=service)
    if trace.get("corrections_applied"):
        if state.language == "en":
            response = f"Got it, I updated the route to pickup from {state.entities.get('pickup_city')} and return in {state.entities.get('dropoff_city')}. {response}"
        else:
            response = f"تمام، عدلتها: الاستلام من {state.entities.get('pickup_city')} والإرجاع في {state.entities.get('dropoff_city')}. {response}"
    state.guard_failures.extend(context.guard_failures)
    return response


def _handle_monthly(
    state: WorkflowState,
    extraction: dict[str, Any],
    trace: dict[str, Any],
    service: OpenAIService | None = None,
) -> str:
    query = extraction.get("entities", {}).get("vehicle_query") or ""
    price = find_monthly_price(query)
    trace["kb_modules_used"].append("KB04")
    if price:
        quote = calculate_monthly_quote(price)
        state.quote = quote
        trace["lookup_records"].append(price["monthly_price_id"])
        context = AnswerContext(
            intent=Intent.MONTHLY_RENTAL,
            language=state.language,
            computed_totals=quote,
            phase=state.phase,
            tone_mode=state.tone_mode,
            customer_mood=state.customer_mood,
        )
        _with_memory(context, state)
        response = compose_response(context, service=service)
        state.guard_failures.extend(context.guard_failures)
        return response
    context = AnswerContext(
        intent=Intent.MONTHLY_RENTAL,
        language=state.language,
        missing_fields=["vehicle_query"],
        tone_mode=state.tone_mode,
        customer_mood=state.customer_mood,
    )
    _with_memory(context, state)
    return compose_response(context, service=service)


def _handle_branch(
    state: WorkflowState,
    extraction: dict[str, Any],
    message: str,
    trace: dict[str, Any],
    service: OpenAIService | None = None,
) -> str:
    query = extraction.get("entities", {}).get("branch_or_city_query") or message
    branches = find_branch(query)
    trace["kb_modules_used"].append("KB01")
    if branches:
        trace["lookup_records"].append(branches[0]["id"])
    context = AnswerContext(
        intent=Intent.BRANCH_LOOKUP,
        language=state.language,
        phase=state.phase,
        tone_mode=state.tone_mode,
        customer_mood=state.customer_mood,
        user_message=message,
    )
    _with_memory(context, state)
    response = compose_response(context, {"branch": branches[0] if branches else None}, service=service)
    state.guard_failures.extend(context.guard_failures)
    return response


def _handle_general_faq(
    state: WorkflowState,
    message: str,
    trace: dict[str, Any],
    service: OpenAIService | None = None,
    semantic_result: Any | None = None,
    route_candidate: SemanticRouteCandidate | None = None,
) -> str:
    faq = semantic_result.as_general_faq_record() if semantic_result else find_general_faq(message)
    if faq is None:
        source_kbs = route_candidate.target_modules if route_candidate else None
        semantic_results = semantic_kb_search(
            message,
            service,
            source_kbs=source_kbs,
            threshold=0.3 if source_kbs else 0.5,
        )
        if semantic_results:
            semantic_result = semantic_results[0]
            faq = semantic_result.as_general_faq_record()
    trace["kb_modules_used"].append("KB14")
    if faq:
        trace["lookup_records"].append(faq["faq_id"])
    if semantic_result:
        trace["semantic_search"] = {
            "provider": semantic_result.retrieval_method,
            "source_kb": semantic_result.source_kb,
            "record_id": semantic_result.record_id,
            "score": semantic_result.score,
        }
        if semantic_result.source_kb not in trace["kb_modules_used"]:
            trace["kb_modules_used"].append(semantic_result.source_kb)
    context = AnswerContext(
        intent=Intent.GENERAL_FAQ,
        language=state.language,
        phase=state.phase,
        tone_mode=state.tone_mode,
        customer_mood=state.customer_mood,
        user_message=message,
        retrieved_records=[],
    )
    _with_memory(context, state)
    response = compose_response(context, {"general_faq": faq}, service=service)
    state.guard_failures.extend(context.guard_failures)
    return response


def _handle_card_deposit_policy(
    state: WorkflowState,
    message: str,
    trace: dict[str, Any],
    service: OpenAIService | None = None,
) -> str:
    policy = find_card_deposit_policy(message)
    if policy is None and state.intent == Intent.CARD_DEPOSIT_POLICY:
        policy = find_card_deposit_policy("بطاقة خصم")
    state.next_action = "answer_policy"
    trace["kb_modules_used"].append("KB08")
    if policy:
        trace["lookup_records"].append(policy["policy_id"])
    context = AnswerContext(
        intent=Intent.CARD_DEPOSIT_POLICY,
        language=state.language,
        phase=state.phase,
        tone_mode=state.tone_mode,
        customer_mood=state.customer_mood,
        user_message=message,
        next_action=state.next_action,
    )
    if state.suspended_workflow:
        context.allowed_facts["suspended_workflow"] = state.suspended_workflow
    _with_memory(context, state)
    response = compose_response(context, {"card_policy": policy}, service=service)
    state.guard_failures.extend(context.guard_failures)
    return response


def _handle_fleet_price(
    state: WorkflowState,
    extraction: dict[str, Any],
    message: str,
    trace: dict[str, Any],
    service: OpenAIService | None = None,
) -> str:
    query = extraction.get("entities", {}).get("vehicle_query") or message
    daily = find_daily_price(query)
    fleet = find_fleet_category(query)
    trace["kb_modules_used"].extend(["KB02", "KB03"])
    records: dict[str, Any] = {"fleet": fleet, "daily_price": daily}
    if daily:
        trace["lookup_records"].append(daily["price_id"])
        state.quote = daily_price_summary(daily)
        state.entities["vehicle_query"] = query
    context = AnswerContext(
        intent=Intent.FLEET_PRICING,
        language=state.language,
        computed_totals=None,
        phase=state.phase,
        tone_mode=state.tone_mode,
        customer_mood=state.customer_mood,
        user_message=message,
    )
    _with_memory(context, state)
    response = compose_response(context, records, service=service)
    state.guard_failures.extend(context.guard_failures)
    return response


def _handle_mixed_branch_price(
    state: WorkflowState,
    extraction: dict[str, Any],
    message: str,
    trace: dict[str, Any],
    service: OpenAIService | None = None,
) -> str:
    entities = extraction.get("entities", {})
    branch_query = entities.get("branch_or_city_query") or entities.get("pickup_city") or state.entities.get("pickup_city") or message
    vehicle_query = entities.get("vehicle_query") or state.entities.get("vehicle_query") or message
    branches = find_branch(str(branch_query))
    daily = find_daily_price(str(vehicle_query))
    trace["kb_modules_used"].extend(["KB01", "KB02", "KB03"])
    if branches:
        trace["lookup_records"].append(branches[0]["id"])
    if daily:
        trace["lookup_records"].append(daily["price_id"])
        state.entities["vehicle_query"] = vehicle_query
    state.next_action = "answer_mixed_branch_price"
    context = AnswerContext(
        intent=Intent.FLEET_PRICING,
        language=state.language,
        phase=state.phase,
        tone_mode=state.tone_mode,
        customer_mood=state.customer_mood,
        user_message=message,
        next_action=state.next_action,
    )
    sections = []
    if branches:
        sections.append({"type": "branch_info", "branch": branches[0]})
    if daily:
        sections.append({"type": "price_info", "daily_price": daily})
    sections.append({"type": "next_step"})
    context.allowed_facts["response_sections"] = sections
    _with_memory(context, state)
    response = compose_response(context, service=service)
    state.guard_failures.extend(context.guard_failures)
    return response


def _handle_mixed_card_quote(
    state: WorkflowState,
    extraction: dict[str, Any],
    message: str,
    trace: dict[str, Any],
    service: OpenAIService | None = None,
) -> str:
    policy = find_card_deposit_policy(message) or find_card_deposit_policy("بطاقة خصم")
    daily = find_daily_price(str(state.entities.get("vehicle_query") or extraction.get("entities", {}).get("vehicle_query") or message))
    trace["kb_modules_used"].extend(["KB08", "KB03"])
    if policy:
        trace["lookup_records"].append(policy["policy_id"])
    if daily:
        trace["lookup_records"].append(daily["price_id"])
    state.next_action = "answer_card_policy_and_quote_next_step"
    context = AnswerContext(
        intent=Intent.CARD_DEPOSIT_POLICY,
        language=state.language,
        phase=state.phase,
        tone_mode=state.tone_mode,
        customer_mood=state.customer_mood,
        user_message=message,
        next_action=state.next_action,
    )
    sections: list[dict[str, Any]] = []
    if policy:
        sections.append({"type": "card_policy", "card_policy": policy})
    if daily:
        sections.append({"type": "price_info", "daily_price": daily})
    sections.append({"type": "next_step"})
    context.allowed_facts["response_sections"] = sections
    _with_memory(context, state)
    response = compose_response(context, service=service)
    state.guard_failures.extend(context.guard_failures)
    return response


def _handle_availability_caveat(
    state: WorkflowState,
    message: str,
    trace: dict[str, Any],
    service: OpenAIService | None = None,
) -> str:
    trace["kb_modules_used"].extend(["KB02"])
    state.next_action = "answer_availability_caveat"
    context = AnswerContext(
        intent=Intent.FLEET_PRICING,
        language=state.language,
        phase=state.phase,
        tone_mode=state.tone_mode,
        customer_mood=state.customer_mood,
        user_message=message,
        next_action=state.next_action,
    )
    context.allowed_facts["availability_caveat_only"] = True
    _with_memory(context, state)
    response = compose_response(context, service=service)
    state.guard_failures.extend(context.guard_failures)
    return response


def _handle_quote_inclusion(
    state: WorkflowState,
    message: str,
    trace: dict[str, Any],
    service: OpenAIService | None = None,
) -> str:
    trace["kb_modules_used"].append("calculator_quote")
    state.next_action = "answer_quote_inclusion"
    context = AnswerContext(
        intent=Intent.FLEET_PRICING,
        language=state.language,
        computed_totals=state.quote,
        phase=state.phase,
        tone_mode=state.tone_mode,
        customer_mood=state.customer_mood,
        user_message=message,
        next_action=state.next_action,
    )
    context.allowed_facts["quote_inclusion_answer"] = True
    _with_memory(context, state)
    response = compose_response(context, service=service)
    state.guard_failures.extend(context.guard_failures)
    return response


def _handle_roadside(
    state: WorkflowState,
    extraction: dict[str, Any],
    trace: dict[str, Any],
    service: OpenAIService | None = None,
) -> str:
    if state.phase not in {phase.value for phase in RoadsidePhase}:
        state.phase = RoadsidePhase.SAFETY_CHECK.value
    advance_roadside_workflow(state, extraction.get("entities", {}))
    trace["kb_modules_used"].extend(["KB10"])
    if state.phase == RoadsidePhase.CASE_READY.value:
        result = create_roadside_case(state.entities)
        transition_roadside_phase(state, RoadsidePhase.CASE_RECORDED)
        trace["mock_action_results"].append(asdict(result))
        context = AnswerContext(
            intent=Intent.ROADSIDE_ASSISTANCE,
            language=state.language,
            phase=state.phase,
            tone_mode=state.tone_mode,
            customer_mood=state.customer_mood,
        )
        _with_memory(context, state)
        response = guard_or_fallback(result.payload[f"customer_message_{state.language}"], context)
        state.guard_failures.extend(context.guard_failures)
        return response
    context = AnswerContext(
        intent=Intent.ROADSIDE_ASSISTANCE,
        language=state.language,
        missing_fields=state.missing_fields,
        phase=state.phase,
        tone_mode=state.tone_mode,
        customer_mood=state.customer_mood,
    )
    if any(token in str(state.entities.get("drivable_status", "")) for token in ["uncertain"]) or state.last_topic == "roadside_safety":
        context.allowed_facts["drive_advice"] = True
    _with_memory(context, state)
    response = compose_response(context, service=service)
    state.guard_failures.extend(context.guard_failures)
    return response


def _handle_escalation(
    state: WorkflowState,
    message: str,
    trace: dict[str, Any],
    service: OpenAIService | None = None,
) -> str:
    rule = match_escalation(message)
    _merge_support_case_details(state, message)
    state.risk_level = "high"
    state.next_action = "handover_to_agent"
    trace["kb_modules_used"].append("KB13")
    trace["escalation_status"] = True
    if rule:
        trace["lookup_records"].append(rule["case_type"])
        result = handover_to_agent(rule["case_type"], state.entities)
        trace["mock_action_results"].append(asdict(result))
    context = AnswerContext(
        intent=Intent.COMPLAINT_OR_FINANCIAL_DISPUTE,
        language=state.language,
        risk_level="high",
        escalation_required=True,
        phase=state.phase,
        tone_mode=state.tone_mode,
        customer_mood=state.customer_mood,
        next_action=state.next_action,
        user_message=message,
    )
    context.allowed_facts["support_case"] = dict(state.support_case)
    _with_memory(context, state)
    response = compose_response(context, service=service)
    state.guard_failures.extend(context.guard_failures)
    return response


def _handle_service_feedback(
    state: WorkflowState,
    message: str,
    trace: dict[str, Any],
    service: OpenAIService | None = None,
) -> str:
    _merge_support_case_details(state, message)
    state.risk_level = "medium"
    state.next_action = "collect_feedback_details"
    trace["kb_modules_used"].append("KB13")
    trace["escalation_status"] = False
    context = AnswerContext(
        intent=Intent.SERVICE_EXPERIENCE_FEEDBACK,
        language=state.language,
        risk_level="medium",
        phase=state.phase,
        tone_mode="serious_supportive",
        customer_mood="upset",
        next_action=state.next_action,
        user_message=message,
    )
    context.allowed_facts["support_case"] = dict(state.support_case)
    _with_memory(context, state)
    response = compose_response(context, service=service)
    state.guard_failures.extend(context.guard_failures)
    return response
