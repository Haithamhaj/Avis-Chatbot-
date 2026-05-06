from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from avis_ai_demo.core.data_lookup import find_dropoff_fee
from avis_ai_demo.core.types import QuoteResult


class CalculationError(ValueError):
    pass


class PriceTableMismatch(CalculationError):
    pass


class UnknownDropoffRoute(CalculationError):
    pass


def _money(value: float | int) -> float:
    return round(float(value), 2)


def _assert_daily_record(record: dict[str, Any]) -> None:
    if "price_id" not in record or "online_total_vat_sar" not in record:
        raise PriceTableMismatch("Expected a daily pricing record.")
    if "monthly_price_id" in record:
        raise PriceTableMismatch("Monthly pricing record cannot be used for daily quote.")


def _assert_monthly_record(record: dict[str, Any]) -> None:
    if "monthly_price_id" not in record or "total_inc_vat_sar" not in record:
        raise PriceTableMismatch("Expected a monthly pricing record.")
    if "price_id" in record:
        raise PriceTableMismatch("Daily pricing record cannot be used for monthly quote.")


def daily_price_summary(daily_price: dict[str, Any]) -> QuoteResult:
    _assert_daily_record(daily_price)
    return QuoteResult(
        rental_type="daily",
        calculator_source="daily_price_summary",
        values={
            "price_id": daily_price["price_id"],
            "category_id": daily_price["category_id"],
            "models": daily_price.get("models_2026", []),
            "online_daily_price_sar": _money(daily_price["online_total_vat_sar"]),
            "in_branch_daily_price_sar": _money(daily_price["in_branch_total_vat_sar"]),
            "currency": daily_price.get("currency", "SAR"),
            "totals_produced": False,
        },
    )


def calculate_daily_quote(
    daily_price: dict[str, Any],
    rental_days: int,
    pickup_city: str,
    dropoff_city: str,
) -> QuoteResult:
    _assert_daily_record(daily_price)
    if rental_days <= 0:
        raise CalculationError("rental_days must be positive.")

    dropoff_fee = find_dropoff_fee(pickup_city, dropoff_city)
    if dropoff_fee is None:
        raise UnknownDropoffRoute(f"No drop-off fee for {pickup_city} to {dropoff_city}.")

    online_daily = _money(daily_price["online_total_vat_sar"])
    branch_daily = _money(daily_price["in_branch_total_vat_sar"])
    online_subtotal = _money(online_daily * rental_days)
    branch_subtotal = _money(branch_daily * rental_days)

    return QuoteResult(
        rental_type="daily",
        calculator_source="calculate_daily_quote",
        values={
            "price_id": daily_price["price_id"],
            "category_id": daily_price["category_id"],
            "models": daily_price.get("models_2026", []),
            "rental_days": rental_days,
            "pickup_city": pickup_city,
            "dropoff_city": dropoff_city,
            "online_daily_price_sar": online_daily,
            "in_branch_daily_price_sar": branch_daily,
            "online_subtotal_sar": online_subtotal,
            "in_branch_subtotal_sar": branch_subtotal,
            "one_way_dropoff_fee_sar": _money(dropoff_fee),
            "total_online_sar": _money(online_subtotal + dropoff_fee),
            "total_in_branch_sar": _money(branch_subtotal + dropoff_fee),
            "currency": daily_price.get("currency", "SAR"),
            "totals_produced": True,
        },
    )


def calculate_monthly_quote(monthly_price: dict[str, Any], months: int = 1) -> QuoteResult:
    _assert_monthly_record(monthly_price)
    if months <= 0:
        raise CalculationError("months must be positive.")

    online_monthly = _money(monthly_price["online_price_sar"])
    with_cdw = _money(monthly_price["with_cdw_sar"])
    total_inc_vat = _money(monthly_price["total_inc_vat_sar"])

    return QuoteResult(
        rental_type="monthly",
        calculator_source="calculate_monthly_quote",
        values={
            "monthly_price_id": monthly_price["monthly_price_id"],
            "category_id": monthly_price["category_id"],
            "model": monthly_price["model"],
            "months": months,
            "online_monthly_price_sar": online_monthly,
            "with_cdw_sar": with_cdw,
            "total_inc_vat_sar": _money(total_inc_vat * months),
            "currency": monthly_price.get("currency", "SAR"),
            "required_rules": monthly_price.get("required_rules", []),
            "totals_produced": True,
        },
    )

