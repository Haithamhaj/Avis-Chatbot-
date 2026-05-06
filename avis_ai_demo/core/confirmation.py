from __future__ import annotations

from avis_ai_demo.core.types import DailyRentalPhase


AR_CONFIRMATIONS = ["أكد", "ادفع", "موافق", "تمام كمل", "احجز"]
EN_CONFIRMATIONS = ["confirm", "pay", "book it", "go ahead", "proceed"]
SHORT_CONFIRMATIONS = {"yes", "نعم"}


def is_confirmation(message: str, current_phase: str | None = None) -> bool:
    text = " ".join(message.strip().lower().split())
    if not text:
        return False
    if text in SHORT_CONFIRMATIONS:
        return current_phase == DailyRentalPhase.AWAITING_PAYMENT_CONFIRMATION.value
    return any(token in text for token in AR_CONFIRMATIONS) or any(
        token in text for token in EN_CONFIRMATIONS
    )

