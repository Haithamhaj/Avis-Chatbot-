from __future__ import annotations

from dataclasses import asdict

from avis_ai_demo.core.orchestrator import handle_message
from avis_ai_demo.core.types import WorkflowState
from avis_ai_demo.services.openai_service import OpenAIService


WELCOME_MESSAGE = (
    "مرحبًا بك في أفيس السعودية. أقدر أساعدك في حجز سيارة، معرفة الأسعار اليومية "
    "أو الشهرية، الفروع وساعات العمل، شروط التأجير والوديعة، أو طلب مساعدة على الطريق."
)


def render_app() -> None:
    import streamlit as st

    st.set_page_config(page_title="Avis Saudi Chatbot", layout="wide")
    st.title("Avis Saudi Chatbot")

    if "workflow_state" not in st.session_state:
        st.session_state.workflow_state = WorkflowState()
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": WELCOME_MESSAGE}]
    if "trace" not in st.session_state:
        st.session_state.trace = {}

    with st.sidebar:
        with st.expander("Internal Trace", expanded=False):
            st.json(st.session_state.trace or _state_trace(st.session_state.workflow_state))

    for item in st.session_state.messages:
        with st.chat_message(item["role"]):
            st.markdown(item["content"])

    prompt = st.chat_input("اكتب رسالتك / Type your message")
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        response, state, trace = handle_message(
            st.session_state.workflow_state,
            prompt,
            service=OpenAIService(),
        )
        st.session_state.workflow_state = state
        st.session_state.trace = trace
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.rerun()


def _state_trace(state: WorkflowState) -> dict:
    data = asdict(state)
    if data.get("quote"):
        data["quote"] = data["quote"]["values"]
    return data
