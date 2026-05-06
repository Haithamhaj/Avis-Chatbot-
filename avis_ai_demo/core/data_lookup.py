from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any


DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def normalize(text: str | None) -> str:
    return " ".join(str(text or "").strip().lower().split())


@lru_cache(maxsize=None)
def load_kb(filename: str) -> Any:
    return json.loads((DATA_DIR / filename).read_text(encoding="utf-8"))


def _matches_query(query: str, values: list[str]) -> bool:
    q = normalize(query)
    if not q:
        return False
    normalized_values = [normalize(value) for value in values if value]
    return any(q in value or value in q for value in normalized_values)


def _match_score(query: str, values: list[str]) -> tuple[int, int]:
    q = normalize(query)
    best = (0, 0)
    for value in (normalize(item) for item in values if item):
        if not value:
            continue
        if q == value:
            best = max(best, (3, len(value)))
        elif value in q:
            best = max(best, (2, len(value)))
        elif q in value:
            best = max(best, (1, len(value)))
    return best


def find_branch(query: str) -> list[dict[str, Any]]:
    records = load_kb("kb01-branches.json")
    matches = []
    for record in records:
        values = [
            record.get("city", ""),
            record.get("branch_name_ar", ""),
            record.get("branch_name_en", ""),
            *(record.get("aliases") or []),
        ]
        if _matches_query(query, values):
            matches.append(record)
    return matches


def find_fleet_category(query: str) -> dict[str, Any] | None:
    for record in load_kb("kb02-fleet-categories.json"):
        values = [
            record.get("category_id", ""),
            record.get("classification", ""),
            record.get("customer_friendly_name_ar", ""),
            record.get("customer_friendly_name_en", ""),
            *(record.get("aliases") or []),
            *(record.get("representative_models") or []),
        ]
        if _matches_query(query, values):
            return record
    return None


def find_daily_price(query: str) -> dict[str, Any] | None:
    for record in load_kb("kb03-daily-prices.json"):
        values = [
            record.get("price_id", ""),
            record.get("category_id", ""),
            record.get("classification", ""),
            *(record.get("aliases") or []),
            *(record.get("models_2026") or []),
        ]
        if _matches_query(query, values):
            return record
    return None


def find_monthly_price(query: str) -> dict[str, Any] | None:
    for record in load_kb("kb04-monthly-prices.json"):
        values = [
            record.get("monthly_price_id", ""),
            record.get("category_id", ""),
            record.get("classification", ""),
            record.get("model", ""),
            *(record.get("aliases") or []),
        ]
        if _matches_query(query, values):
            return record
    return None


def find_dropoff_fee(from_city: str, to_city: str) -> int | None:
    origin = _canonical_city(from_city)
    destination = _canonical_city(to_city)
    if not origin or not destination:
        return None
    if origin == destination:
        return 0
    matrix = load_kb("kb05-dropoff-fees.json")
    return matrix.get(origin, {}).get(destination)


def _canonical_city(city: str) -> str | None:
    q = normalize(city)
    aliases = {
        "riyadh": ["riyadh", "الرياض"],
        "jeddah": ["jeddah", "جدة", "جده"],
        "dammam": ["dammam", "الدمام"],
        "jubail": ["jubail", "الجبيل"],
        "makkah": ["makkah", "mecca", "مكة", "مكه"],
        "madinah": ["madinah", "medina", "المدينة", "المدينه"],
        "taif": ["taif", "الطائف"],
        "abha": ["abha", "أبها", "ابها"],
        "tabuk": ["tabuk", "تبوك"],
    }
    for canonical, city_aliases in aliases.items():
        if any(normalize(alias) in q or q in normalize(alias) for alias in city_aliases):
            return canonical.title()
    return None


def find_requirement(customer_type_query: str) -> dict[str, Any] | None:
    for record in load_kb("kb07-rental-requirements.json"):
        values = [
            record.get("customer_type", ""),
            record.get("nationality_group", ""),
            *(record.get("aliases") or []),
        ]
        if _matches_query(customer_type_query, values):
            return record
    return None


def find_card_deposit_policy(query: str) -> dict[str, Any] | None:
    q = normalize(query)
    if any(token in q for token in ["ترجع", "تحرير", "release", "متى"]) and any(token in q for token in ["وديعة", "deposit"]):
        for record in load_kb("kb08-card-deposit-policy.json"):
            if record.get("policy_id") == "deposit_release_timing":
                return record
    if any(token in q for token in ["بطاقة خصم", "debit", "مدى", "mada", "مو ائتمان", "not credit"]):
        for record in load_kb("kb08-card-deposit-policy.json"):
            if record.get("policy_id") == "credit_card_requirement":
                return record
    best_record = None
    best_score = (0, 0)
    for record in load_kb("kb08-card-deposit-policy.json"):
        values = [
            record.get("policy_id", ""),
            record.get("topic", ""),
            *(record.get("aliases") or []),
            *(record.get("escalation_triggers") or []),
        ]
        score = _match_score(query, values)
        if score > best_score:
            best_score = score
            best_record = record
    return best_record


def find_addon(query: str) -> dict[str, Any] | None:
    for record in load_kb("kb09-addons.json"):
        values = [
            record.get("addon_id", ""),
            record.get("name_ar", ""),
            record.get("name_en", ""),
            *(record.get("aliases") or []),
        ]
        if _matches_query(query, values):
            return record
    return None


def find_general_faq(query: str) -> dict[str, Any] | None:
    best_record = None
    best_score = (0, 0)
    for record in load_kb("kb14-general-faq.json"):
        values = [
            record.get("faq_id", ""),
            record.get("topic", ""),
            *(record.get("aliases") or []),
        ]
        score = _match_score(query, values)
        if score > best_score:
            best_score = score
            best_record = record
    return best_record


def get_sos_data() -> dict[str, Any]:
    return load_kb("kb10-roadside-accident.json")


def match_escalation(message: str) -> dict[str, Any] | None:
    for record in load_kb("kb13-escalation-rules.json"):
        values = [
            record.get("case_type", ""),
            *(record.get("aliases") or []),
        ]
        if _matches_query(message, values):
            return record
    return None
