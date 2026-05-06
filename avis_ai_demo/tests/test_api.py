from fastapi.testclient import TestClient

from avis_ai_demo.api import app


def test_healthz():
    client = TestClient(app)
    response = client.get("/healthz")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_chat_endpoint_returns_agent_response_and_trace():
    client = TestClient(app)
    response = client.post(
        "/chat",
        json={"session_id": "test-session", "message": "مرحبا"},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["session_id"] == "test-session"
    assert "مرحبًا" in data["response"]
    assert data["trace"]["intent"] == "greeting"


def test_chat_endpoint_preserves_session_state_for_payment_confirmation():
    client = TestClient(app)
    session_id = "quote-session"
    client.delete(f"/sessions/{session_id}")

    first = client.post(
        "/chat",
        json={
            "session_id": session_id,
            "message": "أبغى يارس من الرياض للدمام يومين، اسمي هيثم عمري 30 وعندي رخصة، الاستلام 2026-05-10 الساعة 14:00",
        },
    )
    assert first.status_code == 200
    assert first.json()["trace"]["phase"] == "awaiting_payment_confirmation"

    second = client.post("/chat", json={"session_id": session_id, "message": "نعم"})
    assert second.status_code == 200
    assert second.json()["trace"]["phase"] == "booking_completed"

