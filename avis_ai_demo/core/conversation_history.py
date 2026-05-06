from __future__ import annotations

from typing import Any

from avis_ai_demo.core.types import WorkflowState


MAX_HISTORY_TURNS = 8
MAX_TURN_TEXT = 280


def record_conversation_turn(
    state: WorkflowState,
    *,
    user_message: str,
    assistant_response: str,
    trace: dict[str, Any],
) -> WorkflowState:
    user_turn = {
        "role": "user",
        "text": _compact_text(user_message),
        "intent": trace.get("intent"),
        "tone_mode": trace.get("tone_mode"),
        "customer_mood": trace.get("customer_mood"),
    }
    assistant_turn = {
        "role": "assistant",
        "text": _compact_text(assistant_response),
        "intent": trace.get("intent"),
        "next_action": trace.get("next_action"),
    }
    state.conversation_turns.extend([user_turn, assistant_turn])
    state.conversation_turns = state.conversation_turns[-MAX_HISTORY_TURNS:]
    state.conversation_summary = build_conversation_summary(state)
    return state


def build_conversation_summary(state: WorkflowState) -> str:
    if not state.conversation_turns:
        return ""
    last_user = next(
        (turn for turn in reversed(state.conversation_turns) if turn.get("role") == "user"),
        {},
    )
    pieces = [
        f"active_language={state.language}",
        f"last_intent={state.intent.value}",
        f"phase={state.phase}",
        f"tone={state.tone_mode}",
    ]
    if state.customer_mood and state.customer_mood != "neutral":
        pieces.append(f"customer_mood={state.customer_mood}")
    if last_user.get("text"):
        pieces.append(f"last_user={last_user['text']}")
    return "; ".join(pieces)


def compact_history_for_ai(state: dict[str, Any] | WorkflowState | None) -> dict[str, Any]:
    if state is None:
        return {"summary": "", "recent_turns": []}
    if isinstance(state, WorkflowState):
        summary = state.conversation_summary
        turns = state.conversation_turns
    else:
        summary = str(state.get("conversation_summary") or "")
        turns = list(state.get("conversation_turns") or [])
    return {
        "summary": _compact_text(summary, limit=360),
        "recent_turns": [
            {
                "role": turn.get("role"),
                "text": _compact_text(str(turn.get("text") or "")),
                "intent": turn.get("intent"),
            }
            for turn in turns[-MAX_HISTORY_TURNS:]
            if turn.get("role") in {"user", "assistant"} and turn.get("text")
        ],
    }


def _compact_text(text: str, *, limit: int = MAX_TURN_TEXT) -> str:
    compacted = " ".join(str(text).split())
    if len(compacted) <= limit:
        return compacted
    return compacted[: limit - 1].rstrip() + "…"
