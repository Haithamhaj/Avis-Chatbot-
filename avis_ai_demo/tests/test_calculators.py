import ast
from pathlib import Path

import pytest

from avis_ai_demo.core.calculators import (
    PriceTableMismatch,
    UnknownDropoffRoute,
    calculate_daily_quote,
    calculate_monthly_quote,
    daily_price_summary,
)
from avis_ai_demo.core.data_lookup import find_daily_price, find_monthly_price


def test_daily_quote_calculates_online_and_branch_totals_in_code():
    price = find_daily_price("Yaris")
    quote = calculate_daily_quote(price, rental_days=2, pickup_city="Riyadh", dropoff_city="Dammam")

    assert quote.calculator_source == "calculate_daily_quote"
    assert quote.values["online_daily_price_sar"] == 217.35
    assert quote.values["in_branch_daily_price_sar"] == 233.45
    assert quote.values["online_subtotal_sar"] == 434.7
    assert quote.values["in_branch_subtotal_sar"] == 466.9
    assert quote.values["one_way_dropoff_fee_sar"] == 250.0
    assert quote.values["total_online_sar"] == 684.7
    assert quote.values["total_in_branch_sar"] == 716.9
    assert quote.values["totals_produced"] is True


def test_same_city_daily_quote_has_zero_dropoff_fee():
    price = find_daily_price("Camry")
    quote = calculate_daily_quote(price, rental_days=1, pickup_city="Riyadh", dropoff_city="Riyadh")

    assert quote.values["one_way_dropoff_fee_sar"] == 0.0
    assert quote.values["total_online_sar"] == quote.values["online_subtotal_sar"]


def test_unknown_dropoff_route_blocks_guessing():
    price = find_daily_price("Yaris")
    with pytest.raises(UnknownDropoffRoute):
        calculate_daily_quote(price, rental_days=2, pickup_city="Abha", dropoff_city="Riyadh")


def test_daily_and_monthly_tables_are_never_mixed():
    daily = find_daily_price("Yaris")
    monthly = find_monthly_price("Yaris")

    with pytest.raises(PriceTableMismatch):
        calculate_daily_quote(monthly, rental_days=2, pickup_city="Riyadh", dropoff_city="Dammam")

    with pytest.raises(PriceTableMismatch):
        calculate_monthly_quote(daily)


def test_price_only_answer_without_branch_has_no_totals():
    price = find_daily_price("Yaris")
    summary = daily_price_summary(price)

    assert summary.values["online_daily_price_sar"] == 217.35
    assert summary.values["in_branch_daily_price_sar"] == 233.45
    assert summary.values["totals_produced"] is False
    assert "total_online_sar" not in summary.values


def test_monthly_quote_uses_monthly_table_only():
    monthly = find_monthly_price("Camry")
    quote = calculate_monthly_quote(monthly)

    assert quote.rental_type == "monthly"
    assert quote.calculator_source == "calculate_monthly_quote"
    assert quote.values["monthly_price_id"] == "monthly_camry"
    assert quote.values["total_inc_vat_sar"] == 7661.0


def test_calculator_module_has_no_openai_or_model_dependency():
    path = Path(__file__).resolve().parents[1] / "core" / "calculators.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.append(node.module)
    assert not any(name.startswith("openai") for name in imports)
    assert "avis_ai_demo.services.openai_service" not in imports

