from avis_ai_demo.core.confirmation import is_confirmation
from avis_ai_demo.core.types import DailyRentalPhase


def test_confirmation_examples_are_recognized():
    assert is_confirmation("أكد")
    assert is_confirmation("ادفع")
    assert is_confirmation("موافق")
    assert is_confirmation("تمام كمل")
    assert is_confirmation("احجز")
    assert is_confirmation("confirm")
    assert is_confirmation("pay")
    assert is_confirmation("book it")
    assert is_confirmation("go ahead")
    assert is_confirmation("proceed")


def test_short_confirmations_are_phase_aware():
    assert not is_confirmation("yes", current_phase=DailyRentalPhase.QUOTE_PRESENTED.value)
    assert not is_confirmation("نعم", current_phase=DailyRentalPhase.INTAKE.value)
    assert is_confirmation("yes", current_phase=DailyRentalPhase.AWAITING_PAYMENT_CONFIRMATION.value)
    assert is_confirmation("نعم", current_phase=DailyRentalPhase.AWAITING_PAYMENT_CONFIRMATION.value)

