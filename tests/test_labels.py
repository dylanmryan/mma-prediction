from mma.labels import (
    decision_subtype,
    map_method,
    parse_scheduled_rounds,
    parse_weight_class,
)


def test_method_ko_tko():
    assert map_method("KO/TKO") == "ko_tko"
    assert map_method("TKO - Doctor's Stoppage") == "ko_tko"


def test_method_submission():
    assert map_method("Submission") == "submission"


def test_method_decision():
    assert map_method("Decision - Unanimous") == "decision"
    assert map_method("Decision - Split") == "decision"
    assert map_method("Decision - Majority") == "decision"


def test_method_excluded():
    assert map_method("DQ") is None
    assert map_method("Overturned") is None
    assert map_method("Could Not Continue") is None
    assert map_method(None) is None


def test_decision_subtype():
    assert decision_subtype("Decision - Unanimous") == "unanimous"
    assert decision_subtype("Decision - Split") == "split"
    assert decision_subtype("Decision - Majority") == "majority"
    assert decision_subtype("KO/TKO") is None


def test_scheduled_rounds():
    assert parse_scheduled_rounds("3 Rnd (5-5-5)") == 3
    assert parse_scheduled_rounds("5 Rnd (5-5-5-5-5)") == 5
    assert parse_scheduled_rounds("1 Rnd + OT (15-3)") == 1
    assert parse_scheduled_rounds("No Time Limit") is None
    assert parse_scheduled_rounds(None) is None


def test_weight_class_ordering_pitfalls():
    # 'Light Heavyweight' must not match 'Heavyweight'
    assert parse_weight_class("UFC Light Heavyweight Title Bout") == "Light Heavyweight"
    assert parse_weight_class("Heavyweight Bout") == "Heavyweight"
    # Women's divisions must not match the men's substring
    assert parse_weight_class("Women's Strawweight Bout") == "Women's Strawweight"
    assert parse_weight_class("Lightweight Bout") == "Lightweight"
    assert parse_weight_class("Catch Weight Bout") == "Catch Weight"
    assert parse_weight_class("Some Unknown Bout") is None
