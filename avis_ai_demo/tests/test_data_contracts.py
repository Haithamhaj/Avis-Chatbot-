import json
from pathlib import Path


DATA_DIR = Path(__file__).resolve().parents[1] / "data"

REQUIRED_FILES = [
    "kb00-source-registry.json",
    "kb01-branches.json",
    "kb02-fleet-categories.json",
    "kb03-daily-prices.json",
    "kb04-monthly-prices.json",
    "kb05-dropoff-fees.json",
    "kb06-booking-channels.json",
    "kb07-rental-requirements.json",
    "kb08-card-deposit-policy.json",
    "kb09-addons.json",
    "kb10-roadside-accident.json",
    "kb11-chauffeur.json",
    "kb12-corporate.json",
    "kb13-escalation-rules.json",
    "kb14-general-faq.json",
    "kb15-intent-examples.json",
    "kb16-semantic-route-metadata.json",
    "kb17-tone-lexicon.json",
]


def load(name: str):
    return json.loads((DATA_DIR / name).read_text(encoding="utf-8"))


def test_all_kb_files_exist_and_parse():
    for filename in REQUIRED_FILES:
        path = DATA_DIR / filename
        assert path.exists(), filename
        assert load(filename) is not None


def test_required_structured_datasets_are_populated():
    assert len(load("kb01-branches.json")) == 7
    assert len(load("kb03-daily-prices.json")) == 5
    assert len(load("kb04-monthly-prices.json")) == 4
    assert len(load("kb07-rental-requirements.json")) == 5
    assert len(load("kb08-card-deposit-policy.json")) == 4
    assert len(load("kb13-escalation-rules.json")) == 3

    dropoff = load("kb05-dropoff-fees.json")
    assert set(dropoff) == {"Jeddah", "Riyadh", "Dammam", "Jubail"}
    assert all(len(routes) == 8 for routes in dropoff.values())

    roadside = load("kb10-roadside-accident.json")
    assert len(roadside["sos_channels"]) == 3
    assert "notes" in roadside["required_intake_fields"]


def test_no_raw_combined_knowledge_bank_file_exists():
    names = {path.name.lower() for path in DATA_DIR.iterdir()}
    assert "knowledge_bank.json" not in names
    assert "raw_knowledge_bank.txt" not in names


def test_key_fields_present_on_pricing_and_branches():
    for branch in load("kb01-branches.json"):
        assert {"id", "city", "branch_name_ar", "branch_name_en", "aliases", "hours"} <= set(branch)

    for price in load("kb03-daily-prices.json"):
        assert {
            "price_id",
            "category_id",
            "aliases",
            "in_branch_total_vat_sar",
            "online_total_vat_sar",
            "currency",
        } <= set(price)


def test_semantic_route_metadata_is_candidate_only():
    records = load("kb16-semantic-route-metadata.json")
    assert len(records) >= 8
    for record in records:
        assert {
            "route_id",
            "type",
            "target_intent",
            "target_modules",
            "examples_ar",
            "examples_en",
            "allowed_next_step",
            "notes",
        } <= set(record)
        assert "Candidate only" in record["notes"] or record["type"] == "kb_lookup"

    for price in load("kb04-monthly-prices.json"):
        assert {
            "monthly_price_id",
            "category_id",
            "aliases",
            "online_price_sar",
            "total_inc_vat_sar",
            "required_rules",
        } <= set(price)


def test_tone_lexicon_is_non_operational():
    records = load("kb17-tone-lexicon.json")
    phrases = {record["phrase"] for record in records}
    assert {"هلا وارحب", "أبشر", "تامر امر", "غالي و الطلب رخيص", "على هالخشم"} <= phrases
    for record in records:
        assert {"phrase", "tone_modes", "intents", "auto_use", "notes"} <= set(record)
        assert "roadside_assistance" not in record["intents"]
        assert "complaint_or_dispute" not in record["intents"]
