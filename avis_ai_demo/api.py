from __future__ import annotations

from typing import Any

from fastapi import FastAPI
from pydantic import BaseModel, Field

from avis_ai_demo.core.orchestrator import handle_message
from avis_ai_demo.core.types import WorkflowState
from avis_ai_demo.services.openai_service import OpenAIService


app = FastAPI(title="Avis Saudi Chatbot API", version="0.1.0")

_SESSIONS: dict[str, WorkflowState] = {}


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1)
    session_id: str = Field(default="default", min_length=1)


class ChatResponse(BaseModel):
    session_id: str
    response: str
    trace: dict[str, Any]


@app.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    state = _SESSIONS.get(request.session_id)
    response, updated_state, trace = handle_message(
        state,
        request.message,
        service=OpenAIService(),
    )
    _SESSIONS[request.session_id] = updated_state
    return ChatResponse(session_id=request.session_id, response=response, trace=trace)


@app.delete("/sessions/{session_id}")
def reset_session(session_id: str) -> dict[str, str]:
    _SESSIONS.pop(session_id, None)
    return {"status": "reset", "session_id": session_id}

