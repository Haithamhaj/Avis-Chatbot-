from avis_ai_demo.core.ambiguity import clarification_for_ambiguity, detect_ambiguities
from avis_ai_demo.core.intent_router import route_intent
from avis_ai_demo.core.language import detect_language
from avis_ai_demo.core.types import Intent


def test_language_detection_follows_meaningful_message_and_ignores_short_tokens():
    assert detect_language("أبغى سيارة", previous_language="en") == "ar"
    assert detect_language("I need a car", previous_language="ar") == "en"
    assert detect_language("OK", previous_language="ar") == "ar"
    assert detect_language("تمام", previous_language="en") == "en"


def test_routing_examples_from_kb15():
    assert route_intent("أحتاج سيارة من الرياض بكرة") == Intent.DAILY_RENTAL
    assert route_intent("monthly rental") == Intent.MONTHLY_RENTAL
    assert route_intent("where is Sulaymaniyah branch") == Intent.BRANCH_LOOKUP
    assert route_intent("how much is Yaris") == Intent.FLEET_PRICING
    assert route_intent("do I need credit card") == Intent.CARD_DEPOSIT_POLICY
    assert route_intent("battery is dead") == Intent.ROADSIDE_ASSISTANCE


def test_escalation_overrides_normal_deposit_policy():
    assert route_intent("وديعة ما رجعت بعد حجز دولي") == Intent.COMPLAINT_OR_FINANCIAL_DISPUTE
    assert route_intent("I need refund for wrong charge") == Intent.COMPLAINT_OR_FINANCIAL_DISPUTE


def test_ambiguity_detection_and_clarification():
    ambiguities = detect_ambiguities("كم سعر يارس")
    assert "rental_type" in ambiguities
    assert clarification_for_ambiguity(ambiguities, "ar") == "تقصد السعر اليومي أم الشهري؟"

    date_ambiguities = detect_ambiguities("استلام الخميس الساعة 2")
    assert "date_time" in date_ambiguities
    assert "صباح" in clarification_for_ambiguity(date_ambiguities, "ar")

