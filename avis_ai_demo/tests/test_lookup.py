from avis_ai_demo.core.data_lookup import (
    find_addon,
    find_branch,
    find_card_deposit_policy,
    find_daily_price,
    find_dropoff_fee,
    find_fleet_category,
    find_monthly_price,
    find_requirement,
    get_sos_data,
    match_escalation,
)


def test_find_branch_by_arabic_alias_and_city():
    sulaymaniyah = find_branch("السليمانية")
    assert len(sulaymaniyah) == 1
    assert sulaymaniyah[0]["id"] == "riyadh_sulaymaniyah"

    riyadh = find_branch("Riyadh")
    assert {branch["id"] for branch in riyadh} >= {"riyadh_t5", "riyadh_sulaymaniyah"}


def test_find_fleet_category_returns_category_not_inventory():
    record = find_fleet_category("يارس")
    assert record is not None
    assert record["category_id"] == "economy_plus_sedan"
    assert "representative_models" in record
    assert "branches" not in record
    assert "inventory" not in record


def test_find_daily_and_monthly_prices_without_branch():
    daily = find_daily_price("Yaris")
    monthly = find_monthly_price("Yaris")

    assert daily is not None
    assert daily["price_id"] == "daily_economy_plus_sedan"
    assert daily["online_total_vat_sar"] == 217.35
    assert "branch" not in daily

    assert monthly is not None
    assert monthly["monthly_price_id"] == "monthly_economy_plus_sedan"
    assert monthly["total_inc_vat_sar"] == 4631.9
    assert "branch" not in monthly


def test_find_dropoff_fee_full_matrix_and_unknown_route():
    assert find_dropoff_fee("Riyadh", "Dammam") == 250
    assert find_dropoff_fee("جدة", "تبوك") == 430
    assert find_dropoff_fee("Riyadh", "Riyadh") == 0
    assert find_dropoff_fee("Abha", "Riyadh") is None


def test_find_requirement_and_policy_and_addon():
    requirement = find_requirement("visitor")
    assert requirement is not None
    assert requirement["customer_type"] == "International Visitor"
    assert requirement["min_age"] == 21

    policy = find_card_deposit_policy("متى ترجع الوديعة")
    assert policy is not None
    assert policy["policy_id"] == "deposit_release_timing"

    addon = find_addon("unlimited km")
    assert addon is not None
    assert addon["addon_id"] == "open_mileage"


def test_sos_data_and_escalation_matching():
    sos = get_sos_data()
    assert len(sos["sos_channels"]) == 3
    assert "arrival time" in sos["do_not_promise"]

    escalation = match_escalation("الوديعة ما رجعت بعد حجز دولي")
    assert escalation is not None
    assert escalation["case_type"] in {"financial_dispute", "international_rental_issue"}

