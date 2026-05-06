from __future__ import annotations

import os
import json
from typing import Any, Protocol

from avis_ai_demo.core.conversation_history import compact_history_for_ai
from avis_ai_demo.core.conversation_decision import decision_context


class ChatClient(Protocol):
    def extract(self, message: str, state: dict[str, Any] | None = None) -> str | dict[str, Any]:
        ...


class OpenAIService:
    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        embedding_model: str | None = None,
        client: ChatClient | None = None,
    ) -> None:
        self.api_key = api_key if api_key is not None else os.environ.get("OPENAI_API_KEY")
        self.model = model if model is not None else os.environ.get("OPENAI_MODEL", "gpt-5.4")
        self.embedding_model = embedding_model if embedding_model is not None else os.environ.get("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
        self.client = client

    @property
    def available(self) -> bool:
        return bool(self.api_key) or self.client is not None

    def extract(self, message: str, state: dict[str, Any] | None = None) -> str | dict[str, Any]:
        if self.client is not None:
            return self.client.extract(message, state)
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY is not configured.")

        from openai import OpenAI

        context = _state_context_for_ai(state)
        client = OpenAI(api_key=self.api_key)
        response = client.responses.create(
            model=self.model,
            input=[
                {
                    "role": "system",
                    "content": (
                        "Extract operational intent and entities for Avis Saudi. Return only strict JSON "
                        "with keys: intent, language, entities, ambiguities, clarification_question, confidence. "
                        "Allowed intents include daily_rental, monthly_rental, branch_lookup, fleet_pricing, "
                        "card_deposit_policy, roadside_assistance, complaint_or_financial_dispute, general_faq, fallback_unknown. "
                        "Entities may include pickup_city, dropoff_city, pickup_branch, dropoff_branch, "
                        "pickup_date, pickup_time, rental_days, vehicle_query, has_valid_license, age, mobile, "
                        "plate_or_contract, current_location, issue_type, drivable_status, safety_status. "
                        "Do not calculate prices or totals. Do not answer the user."
                    ),
                },
                {"role": "user", "content": json.dumps({"conversation_context": context, "message": message}, ensure_ascii=False)},
            ],
        )
        return response.output_text

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if self.client is not None and hasattr(self.client, "embed_texts"):
            return self.client.embed_texts(texts)
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY is not configured.")

        from openai import OpenAI

        client = OpenAI(api_key=self.api_key)
        response = client.embeddings.create(model=self.embedding_model, input=texts)
        return [item.embedding for item in response.data]

    def classify_conversation(
        self,
        message: str,
        state: dict[str, Any] | None = None,
    ) -> str | dict[str, Any]:
        if self.client is not None and hasattr(self.client, "classify_conversation"):
            return self.client.classify_conversation(message, state)
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY is not configured.")

        from openai import OpenAI

        context = _state_context_for_ai(state)
        client = OpenAI(api_key=self.api_key)
        response = client.responses.create(
            model=self.model,
            input=[
                {
                    "role": "system",
                    "content": (
                        "Classify the customer message for Avis Saudi conversation management. "
                        "Return only strict JSON with keys: intent, tone_mode, customer_mood, next_action, confidence. "
                        "Allowed intents: greeting, small_talk, off_topic, thanks_acknowledgement, "
                        "capability_question, potentially_relevant_unclear, operational_request, "
                        "complaint_or_dispute, roadside_or_accident, fallback_unknown. "
                        "Allowed tone_mode: friendly_casual, professional_helpful, serious_supportive, "
                        "safety_first, light_deflection. Allowed next_action: wait_for_user_request, "
                        "scope_redirect, answer_capabilities, ask_clarification, route_operational, "
                        "collect_case_details, safety_check, fallback_with_service_options. "
                        "Do not answer the user. Do not include facts, "
                        "prices, calculations, branch data, policy decisions, or JSON beyond the object."
                    ),
                },
                {"role": "user", "content": json.dumps({"conversation_context": context, "message": message}, ensure_ascii=False)},
            ],
        )
        return response.output_text

    def decide_conversation(
        self,
        message: str,
        state: dict[str, Any] | None = None,
    ) -> str | dict[str, Any]:
        if self.client is not None and hasattr(self.client, "decide_conversation"):
            return self.client.decide_conversation(message, state)
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY is not configured.")

        from openai import OpenAI

        context = decision_context(state)
        client = OpenAI(api_key=self.api_key)
        response = client.responses.create(
            model=self.model,
            input=[
                {
                    "role": "system",
                    "content": (
                        "You are the Conversation Decision Layer for Avis Saudi. Return only strict JSON "
                        "with keys: interaction_type, service_domain, workflow_candidate, workflow_readiness, "
                        "confidence, customer_mood, needs_clarification, suggested_dialogue_act. "
                        "Classify the message as conversation first. Use hard_safety only for accidents, "
                        "injuries, danger, or non-drivable vehicle. Use hard_financial_dispute only for explicit "
                        "charges, refunds, deposits not returned, payment complaints, or paid-with-no-booking cases. "
                        "Treat discount/offer requests as faq_or_info or social_with_service_hint, not financial disputes. "
                        "Treat playful praise with a service hint as social_with_service_hint. Mark workflow_candidate "
                        "not_ready when the customer may want a workflow but missing intent/details need clarification. "
                        "Do not answer the user. Do not calculate, invent facts, or decide payment/booking status."
                    ),
                },
                {"role": "user", "content": json.dumps({"conversation_context": context, "message": message}, ensure_ascii=False)},
            ],
        )
        return response.output_text

    def generate_clarification(self, language: str, missing_fields: list[str]) -> str:
        if language == "en":
            return f"Please clarify: {', '.join(missing_fields)}."
        return f"فضلاً وضّح: {', '.join(missing_fields)}."

    def phrase_response(self, deterministic_response: str, _context: dict[str, Any] | None = None) -> str:
        if self.client is not None and hasattr(self.client, "phrase_response"):
            return self.client.phrase_response(deterministic_response, _context)
        if self.client is not None:
            return deterministic_response
        if self.api_key and _context is not None:
            from openai import OpenAI

            client = OpenAI(api_key=self.api_key)
            response = client.responses.create(
                model=self.model,
                input=[
                    {
                        "role": "system",
                        "content": (
                            "You are Khalid, Avis Saudi's virtual assistant. Rephrase the provided "
                            "draft into a lively but safe customer-facing message. Use only the allowed "
                            "facts in the context. Do not add prices, availability, booking status, "
                        "payment status, branch facts, policies, JSON, internal trace, or workflow names."
                        "Do not invent casual nicknames or address the customer with يا حلو, يا بطل, or يا غالي."
                    ),
                    },
                    {"role": "user", "content": f"Context: {_context}\nDraft: {deterministic_response}"},
                ],
            )
            return response.output_text
        return deterministic_response

    def phrase_khalid_response(self, plan: dict[str, Any]) -> str:
        if self.client is not None and hasattr(self.client, "phrase_khalid_response"):
            return self.client.phrase_khalid_response(plan)
        if self.client is not None:
            return str(plan.get("deterministic_fallback", ""))
        if not self.api_key:
            return str(plan.get("deterministic_fallback", ""))

        from openai import OpenAI

        client = OpenAI(api_key=self.api_key)
        response = client.responses.create(
            model=self.model,
            input=[
                {
                    "role": "system",
                    "content": (
                        "You are Khalid, Avis Saudi's virtual assistant. Write only the final "
                        "customer-facing reply. You control tone and wording only. Use only the "
                        "allowed_facts, lookup_records, computed_totals, missing_fields, and "
                        "deterministic_fallback in the plan. Do not calculate, invent prices, "
                        "invent availability, invent booking or payment status, invent branch or "
                        "policy facts, expose JSON, expose KB IDs, expose workflow names, or mention "
                        "demo/mock/API limitations. Do not invent casual nicknames or address the customer "
                        "with يا حلو, يا بطل, or يا غالي. If tone_phrases are provided, you may use at most "
                        "one and only when it naturally fits the tone. Do not stack greetings or compliments, "
                        "such as هلا وارحب، مرحبًا، تشرفنا. Use conversation_memory only for continuity, "
                        "tone, and avoiding repeated wording; never as a source for operational facts. "
                        "If the customer is following up on an unclear issue, do not repeat the same "
                        "clarification options. Keep the reply concise and match the requested language."
                    ),
                },
                {"role": "user", "content": json.dumps(plan, ensure_ascii=False)},
            ],
        )
        return response.output_text


def _state_context_for_ai(state: dict[str, Any] | None) -> dict[str, Any]:
    state = state or {}
    return {
        "active_language": state.get("language"),
        "current_intent": str(state.get("intent") or ""),
        "current_phase": state.get("phase"),
        "next_action": state.get("next_action"),
        "conversation_memory": compact_history_for_ai(state),
        "note": "Use this as conversational context only. It is not a source for prices, availability, payment status, booking status, branch data, or policy rules.",
    }
