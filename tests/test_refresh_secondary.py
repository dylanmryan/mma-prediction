"""Unit tests for the secondary (daily ufcstats scrape) adapter and merge.

No network: every test either exercises a pure function or monkeypatches
`fetch_source`. The fixtures below are raw-shaped -- ufcstats strings like
"45 of 118", "4:20" and fighter/fight URLs -- because that is what the
upstream CSVs actually contain and the adaptation is the thing under test.
"""
from __future__ import annotations

import io

import pandas as pd
import pytest

from scripts.refresh_secondary import (
    SecondaryRefused,
    UnknownFighterError,
    adapt,
    apply_secondary,
    build_name_index,
    build_provenance,
    build_secondary_fighters,
    complete_round_fights,
    merge_into,
    merge_secondary,
)

FIGHTS_COLUMNS = [
    "fight_id", "date", "fighter_a_id", "fighter_b_id", "winner",
    "method", "method_raw", "decision_subtype", "finish_round",
    "scheduled_rounds", "weight_class", "title_fight", "duration_sec",
    "referee", "event_id", "location",
]

EVENT_ONE = "aaaa000000000001"
EVENT_TWO = "aaaa000000000002"
FIGHT_ONE = "ffff000000000001"
FIGHT_TWO = "ffff000000000002"
FIGHT_THREE = "ffff000000000003"
ALICE = "1111111111111111"
BOB = "2222222222222222"
CARA = "3333333333333333"
DANA = "4444444444444444"
ELI = "5555555555555555"
FINN = "6666666666666666"


def _csv(text: str) -> pd.DataFrame:
    return pd.read_csv(io.StringIO(text.strip() + "\n"))


@pytest.fixture
def source() -> dict[str, pd.DataFrame]:
    """A miniature copy of the upstream CSV set, raw strings and all."""
    events = _csv(
        f"""
EVENT,URL,DATE,LOCATION
Test Event One,http://ufcstats.com/event-details/{EVENT_ONE},"September 05, 2026","Paris, France"
Test Event Two,http://ufcstats.com/event-details/{EVENT_TWO},"August 29, 2026","Shanghai, China"
Noche Test Event Two,http://ufcstats.com/event-details/{EVENT_TWO},"August 29, 2026","Shanghai, China"
"""
    )
    # Fight three is listed twice, once under each of the event's two names --
    # upstream really does this for the co-branded cards (Noche UFC).
    results = _csv(
        f"""
EVENT,BOUT,OUTCOME,WEIGHTCLASS,METHOD,ROUND,TIME,TIME FORMAT,REFEREE,DETAILS,URL
Test Event One ,Alice Ace vs. Bob Blue,W/L,Lightweight Bout,KO/TKO ,2,2:35,3 Rnd (5-5-5),Ref One,Punch,http://ufcstats.com/fight-details/{FIGHT_ONE}
Test Event One ,Cara Cole vs. Dana Dee,L/W,UFC Women's Flyweight Title Bout,Decision - Unanimous ,5,5:00,5 Rnd (5-5-5-5-5),Ref Two,Cards,http://ufcstats.com/fight-details/{FIGHT_TWO}
Test Event Two ,Eli East vs. Finn Frost,D/D,Welterweight Bout,Decision - Majority ,3,5:00,3 Rnd (5-5-5),Ref Three,Cards,http://ufcstats.com/fight-details/{FIGHT_THREE}
Noche Test Event Two ,Eli East vs. Finn Frost,D/D,Welterweight Bout,Decision - Majority ,3,5:00,3 Rnd (5-5-5),Ref Three,Cards,http://ufcstats.com/fight-details/{FIGHT_THREE}
"""
    )
    stats = _csv(
        f"""
EVENT,BOUT,ROUND,FIGHTER,KD,SIG.STR.,SIG.STR. %,TOTAL STR.,TD,TD %,SUB.ATT,REV.,CTRL,HEAD,BODY,LEG,DISTANCE,CLINCH,GROUND
Test Event One,Alice Ace vs. Bob Blue,Round 1,Alice Ace,1,45 of 118,38%,50 of 130,1 of 2,50%,0,0,4:20,20 of 60,15 of 40,10 of 18,40 of 110,3 of 5,2 of 3
Test Event One,Alice Ace vs. Bob Blue,Round 2,Alice Ace,0,5 of 10,50%,6 of 12,0 of 0,---,1,0,0:40,3 of 6,1 of 2,1 of 2,5 of 10,0 of 0,0 of 0
Test Event One,Alice Ace vs. Bob Blue,Round 1,Bob Blue,0,12 of 30,40%,15 of 35,0 of 1,0%,0,1,0:00,6 of 15,4 of 10,2 of 5,12 of 30,0 of 0,0 of 0
Test Event One,Alice Ace vs. Bob Blue,Round 2,Bob Blue,0,2 of 8,25%,2 of 8,0 of 0,---,0,0,---,1 of 4,1 of 3,0 of 1,2 of 8,0 of 0,0 of 0
Test Event Two,Eli East vs. Finn Frost,Round 1,Eli East,0,7 of 20,35%,9 of 24,0 of 0,---,0,0,1:00,4 of 12,2 of 5,1 of 3,7 of 20,0 of 0,0 of 0
Test Event Two,Eli East vs. Finn Frost,Round 2,Eli East,0,8 of 21,38%,10 of 25,0 of 0,---,0,0,0:30,5 of 13,2 of 5,1 of 3,8 of 21,0 of 0,0 of 0
Test Event Two,Eli East vs. Finn Frost,Round 3,Eli East,0,9 of 22,41%,11 of 26,0 of 0,---,0,0,0:00,6 of 14,2 of 5,1 of 3,9 of 22,0 of 0,0 of 0
Test Event Two,Eli East vs. Finn Frost,Round 1,Finn Frost,0,6 of 19,32%,8 of 23,0 of 0,---,0,0,0:10,3 of 11,2 of 5,1 of 3,6 of 19,0 of 0,0 of 0
Test Event Two,Eli East vs. Finn Frost,Round 2,Finn Frost,0,5 of 18,28%,7 of 22,0 of 0,---,0,0,0:20,2 of 10,2 of 5,1 of 3,5 of 18,0 of 0,0 of 0
Test Event Two,Eli East vs. Finn Frost,Round 3,Finn Frost,0,4 of 17,24%,6 of 21,0 of 0,---,0,0,0:30,1 of 9,2 of 5,1 of 3,4 of 17,0 of 0,0 of 0
"""
    )
    # HEIGHT/REACH are left as "--" on purpose: the adaptation reads only
    # FIGHTER and URL from this file.
    fighter_tott = _csv(
        f"""
FIGHTER,HEIGHT,WEIGHT,REACH,STANCE,DOB,URL
Alice Ace,--,155 lbs.,--,Orthodox,"Jul 13, 1990",http://ufcstats.com/fighter-details/{ALICE}
Bob Blue,--,155 lbs.,--,Southpaw,"Feb 01, 1994",http://ufcstats.com/fighter-details/{BOB}
Cara Cole,--,125 lbs.,--,Orthodox,"Feb 25, 1993",http://ufcstats.com/fighter-details/{CARA}
Dana Dee,--,125 lbs.,--,Orthodox,"Apr 26, 1995",http://ufcstats.com/fighter-details/{DANA}
Eli East,--,170 lbs.,--,Orthodox,"Jan 22, 1992",http://ufcstats.com/fighter-details/{ELI}
Finn Frost,--,170 lbs.,--,Switch,"Mar 03, 1991",http://ufcstats.com/fighter-details/{FINN}
Sam Same,--,145 lbs.,--,Orthodox,"Mar 03, 1991",http://ufcstats.com/fighter-details/7777777777777777
Sam Same,--,145 lbs.,--,Orthodox,"Mar 03, 1980",http://ufcstats.com/fighter-details/8888888888888888
"""
    )
    fighter_details = _csv(
        f"""
FIRST,LAST,NICKNAME,URL
Alice,Ace,,http://ufcstats.com/fighter-details/{ALICE}
Bob,Blue,,http://ufcstats.com/fighter-details/{BOB}
Cara,Cole,,http://ufcstats.com/fighter-details/{CARA}
Dana,Dee,,http://ufcstats.com/fighter-details/{DANA}
Eli,East,,http://ufcstats.com/fighter-details/{ELI}
Finn,Frost,,http://ufcstats.com/fighter-details/{FINN}
"""
    )
    return {
        "events": events,
        "results": results,
        "stats": stats,
        "fighter_tott": fighter_tott,
        "fighter_details": fighter_details,
    }


# --------------------------------------------------------------------------
# schema adaptation
# --------------------------------------------------------------------------


def test_name_index_drops_ambiguous_names(source):
    index = build_name_index(source["fighter_tott"], source["fighter_details"])
    assert index["Alice Ace"] == ALICE
    assert index["Finn Frost"] == FINN
    # Two ufcstats pages share the name "Sam Same"; guessing between them
    # would silently attach one fighter's history to another.
    assert "Sam Same" not in index


def test_adapted_fights_match_our_schema(source):
    adapted = adapt(source)
    fights, report = adapted.fights, adapted.report

    assert list(fights.columns) == FIGHTS_COLUMNS
    assert str(fights["fight_id"].dtype) == "string"
    assert str(fights["scheduled_rounds"].dtype) == "Int64"
    assert str(fights["duration_sec"].dtype) == "Float64"
    assert fights["title_fight"].dtype == bool
    assert pd.api.types.is_datetime64_any_dtype(fights["date"])
    # The duplicated co-branded listing collapses to one fight.
    assert report["n_source_rows"] == 4
    assert report["n_duplicate_fight_ids"] == 1
    assert len(fights) == 3
    assert fights["fight_id"].is_unique


def test_adapted_fight_values(source):
    fights = adapt(source).fights
    by_id = fights.set_index("fight_id")

    one = by_id.loc[FIGHT_ONE]
    assert one["fighter_a_id"] == ALICE and one["fighter_b_id"] == BOB
    assert one["winner"] == "a"
    assert one["method"] == "ko_tko" and one["method_raw"] == "KO/TKO"
    assert one["finish_round"] == 2
    assert one["scheduled_rounds"] == 3
    assert one["weight_class"] == "Lightweight"
    assert bool(one["title_fight"]) is False
    # Round 2 stoppage at 2:35 -> one full round plus 155 seconds.
    assert float(one["duration_sec"]) == pytest.approx(455.0)
    assert one["event_id"] == EVENT_ONE
    assert one["location"] == "Paris, France"
    assert one["date"] == pd.Timestamp("2026-09-05")

    two = by_id.loc[FIGHT_TWO]
    assert two["winner"] == "b"  # "L/W": the second-named corner won
    assert two["method"] == "decision" and two["decision_subtype"] == "unanimous"
    assert pd.isna(two["finish_round"])  # decisions have no finish round
    assert float(two["duration_sec"]) == pytest.approx(1500.0)  # 5 rounds
    assert bool(two["title_fight"]) is True
    assert two["weight_class"] == "Women's Flyweight"

    three = by_id.loc[FIGHT_THREE]
    assert three["winner"] == "draw"
    assert float(three["duration_sec"]) == pytest.approx(900.0)


def test_adapted_fight_stats_parse_raw_strings(source):
    stats = adapt(source).stats

    assert stats["fight_id"].nunique() == 3
    assert len(stats) == 6  # two corners per fight
    alice = stats[(stats["fight_id"] == FIGHT_ONE) & (stats["corner"] == "a")].iloc[0]
    assert alice["fighter_id"] == ALICE
    assert alice["kd"] == 1
    assert alice["sig_landed"] == 50 and alice["sig_attempted"] == 128
    assert alice["total_landed"] == 56 and alice["total_attempted"] == 142
    assert alice["td_landed"] == 1 and alice["td_attempted"] == 2
    assert alice["sub_att"] == 1
    assert alice["ctrl_sec"] == 300  # 4:20 + 0:40
    assert alice["head_landed"] == 23 and alice["head_attempted"] == 66
    assert alice["ground_landed"] == 2 and alice["ground_attempted"] == 3

    bob = stats[(stats["fight_id"] == FIGHT_ONE) & (stats["corner"] == "b")].iloc[0]
    assert bob["fighter_id"] == BOB
    assert bob["rev"] == 1
    assert bob["ctrl_sec"] == 0  # "---" contributes nothing, "0:00" is zero

    # A fight with no per-round record keeps missing stats missing.
    absent = stats[stats["fight_id"] == FIGHT_TWO]
    assert len(absent) == 2
    assert absent["sig_landed"].isna().all()


def test_rows_whose_names_do_not_resolve_are_dropped_and_counted(source):
    source["results"].loc[0, "BOUT"] = "Nobody Known vs. Bob Blue"
    adapted = adapt(source)
    fights, report = adapted.fights, adapted.report

    assert report["n_unresolved_name"] == 1
    assert FIGHT_ONE not in set(fights["fight_id"])


# --------------------------------------------------------------------------
# merge
# --------------------------------------------------------------------------


def _fights_frame(rows: list[dict]) -> pd.DataFrame:
    frame = pd.DataFrame(rows, columns=["fight_id", "date", "fighter_a_id", "fighter_b_id", "winner"])
    frame["date"] = pd.to_datetime(frame["date"])
    for column in ("fight_id", "fighter_a_id", "fighter_b_id", "winner"):
        frame[column] = frame[column].astype("string")
    return frame


PRIMARY = _fights_frame(
    [
        {"fight_id": "f1", "date": "2026-08-01", "fighter_a_id": "A", "fighter_b_id": "B", "winner": "a"},
        {"fight_id": "f2", "date": "2026-08-08", "fighter_a_id": "B", "fighter_b_id": "C", "winner": "b"},
    ]
)
KNOWN = {"A", "B", "C"}


def test_duplicate_fights_are_dropped_and_primary_wins():
    secondary = _fights_frame(
        [{"fight_id": "f2", "date": "2026-08-08", "fighter_a_id": "B", "fighter_b_id": "C", "winner": "a"}]
    )
    merged, report = merge_secondary(PRIMARY, secondary, KNOWN)

    assert report["n_secondary"] == 1
    assert report["n_dropped_duplicate"] == 1
    assert report["n_added"] == 0
    assert len(merged) == 2
    # The primary label survives: the secondary disagreed and lost.
    assert merged.set_index("fight_id").loc["f2", "winner"] == "b"


def test_new_rows_are_appended_and_counted():
    secondary = _fights_frame(
        [
            {"fight_id": "f2", "date": "2026-08-08", "fighter_a_id": "B", "fighter_b_id": "C", "winner": "b"},
            {"fight_id": "f3", "date": "2026-09-05", "fighter_a_id": "A", "fighter_b_id": "C", "winner": "a"},
        ]
    )
    merged, report = merge_secondary(PRIMARY, secondary, KNOWN)

    assert report["n_secondary"] == 2
    assert report["n_dropped_duplicate"] == 1
    assert report["n_added"] == 1
    assert report["n_rejected_unknown_fighter"] == 0
    assert list(merged["fight_id"]) == ["f1", "f2", "f3"]
    assert report["max_date_before"] == pd.Timestamp("2026-08-08")
    assert report["max_date_after"] == pd.Timestamp("2026-09-05")


def test_unknown_fighter_row_is_rejected_and_named():
    secondary = _fights_frame(
        [{"fight_id": "f3", "date": "2026-09-05", "fighter_a_id": "A", "fighter_b_id": "ZZZ", "winner": "a"}]
    )
    merged, report = merge_secondary(PRIMARY, secondary, KNOWN)

    assert report["n_rejected_unknown_fighter"] == 1
    assert report["n_added"] == 0
    assert report["rejected_fighter_ids"] == ["ZZZ"]
    assert len(merged) == 2
    assert report["max_date_after"] == pd.Timestamp("2026-08-08")


def test_strict_mode_raises_naming_the_unknown_fighter():
    secondary = _fights_frame(
        [{"fight_id": "f3", "date": "2026-09-05", "fighter_a_id": "A", "fighter_b_id": "ZZZ", "winner": "a"}]
    )
    with pytest.raises(UnknownFighterError, match="ZZZ"):
        merge_secondary(PRIMARY, secondary, KNOWN, strict=True)


def test_empty_secondary_is_a_clean_noop():
    merged, report = merge_secondary(PRIMARY, PRIMARY.iloc[:0], KNOWN)

    assert report == {
        "n_secondary": 0,
        "n_added": 0,
        "n_dropped_duplicate": 0,
        "n_rejected_unknown_fighter": 0,
        "rejected_fighter_ids": [],
        "max_date_before": pd.Timestamp("2026-08-08"),
        "max_date_after": pd.Timestamp("2026-08-08"),
    }
    pd.testing.assert_frame_equal(merged, PRIMARY)


def test_missing_column_in_secondary_is_a_loud_error():
    secondary = _fights_frame(
        [{"fight_id": "f3", "date": "2026-09-05", "fighter_a_id": "A", "fighter_b_id": "C", "winner": "a"}]
    ).drop(columns=["winner"])
    with pytest.raises(ValueError, match="winner"):
        merge_secondary(PRIMARY, secondary, KNOWN)


# --------------------------------------------------------------------------
# CLI: the default must never write and must never need the network
# --------------------------------------------------------------------------


def test_the_cli_never_writes(monkeypatch, capsys, source, primary):
    """The script reports; `make_dataset.py` is the only thing that writes."""
    import scripts.refresh_secondary as mod

    monkeypatch.setattr(mod, "fetch_source", lambda base=None: source)
    monkeypatch.setattr(mod, "load_primary", lambda: primary)
    written: list = []
    monkeypatch.setattr(mod.pd.DataFrame, "to_parquet", lambda self, *a, **k: written.append(a))

    exit_code = mod.main([])

    assert exit_code == 0
    assert written == []
    out = capsys.readouterr().out
    assert "fights added" in out
    assert "report only" in out.lower()


def test_the_cli_exits_non_zero_when_the_source_contributes_nothing(
    monkeypatch, capsys, primary
):
    import scripts.refresh_secondary as mod

    def boom(base=None):
        raise OSError("connection reset")

    monkeypatch.setattr(mod, "fetch_source", boom)
    monkeypatch.setattr(mod, "load_primary", lambda: primary)

    assert mod.main([]) == 1
    assert "applied: NO" in capsys.readouterr().out


# --------------------------------------------------------------------------
# fighters: the debutant gap
# --------------------------------------------------------------------------


def test_secondary_fighters_adapt_to_our_schema(source):
    fighters = build_secondary_fighters(
        source["fighter_tott"], source["fighter_details"]
    )

    assert list(fighters.columns) == [
        "fighter_id", "name", "height_cm", "reach_cm", "stance", "dob",
    ]
    assert fighters["fighter_id"].is_unique
    # Both ambiguous "Sam Same" ids still get a row: the ambiguity is in the
    # NAME lookup, not in the ids, and a row keyed by id is unambiguous.
    assert set(fighters["fighter_id"]) >= {ALICE, BOB, CARA, DANA, ELI, FINN}
    alice = fighters.set_index("fighter_id").loc[ALICE]
    assert alice["name"] == "Alice Ace"
    assert alice["stance"] == "Orthodox"
    assert alice["dob"] == pd.Timestamp("1990-07-13")
    # "--" is missing, not a value.
    assert pd.isna(alice["height_cm"])
    assert pd.isna(alice["reach_cm"])


def test_secondary_fighters_convert_height_and_reach(source):
    tott = source["fighter_tott"].copy()
    tott.loc[tott["FIGHTER"] == "Alice Ace", "HEIGHT"] = "5' 10\""
    tott.loc[tott["FIGHTER"] == "Alice Ace", "REACH"] = '72"'
    fighters = build_secondary_fighters(tott, source["fighter_details"])
    alice = fighters.set_index("fighter_id").loc[ALICE]

    assert alice["height_cm"] == pytest.approx(70 * 2.54)
    assert alice["reach_cm"] == pytest.approx(72 * 2.54)


def test_secondary_fighters_fall_back_to_the_details_name(source):
    """An id present only in the details file still gets a named row."""
    tott = source["fighter_tott"]
    tott = tott[tott["FIGHTER"] != "Finn Frost"]
    fighters = build_secondary_fighters(tott, source["fighter_details"])
    finn = fighters.set_index("fighter_id").loc[FINN]

    assert finn["name"] == "Finn Frost"
    assert pd.isna(finn["dob"])  # the details file carries no biography


# --------------------------------------------------------------------------
# round_stats
# --------------------------------------------------------------------------


ROUND_COLUMNS = [
    "fight_id", "round_no", "corner", "fighter_id", "kd", "sig_landed",
    "sig_attempted", "total_landed", "total_attempted", "td_landed",
    "td_attempted", "sub_att", "ctrl_sec", "rev", "head_landed",
    "head_attempted", "body_landed", "body_attempted", "leg_landed",
    "leg_attempted", "distance_landed", "distance_attempted",
    "clinch_landed", "clinch_attempted", "ground_landed", "ground_attempted",
]


def test_adapted_round_stats_match_our_schema(source):
    rounds = adapt(source).rounds

    assert list(rounds.columns) == ROUND_COLUMNS
    assert not rounds.duplicated(["fight_id", "round_no", "corner"]).any()
    # Fight two has no per-round record upstream, so it has none here.
    assert set(rounds["fight_id"]) == {FIGHT_ONE, FIGHT_THREE}
    one = rounds[(rounds["fight_id"] == FIGHT_ONE) & (rounds["corner"] == "a")]
    assert list(one["round_no"]) == [1, 2]
    assert list(one["fighter_id"]) == [ALICE, ALICE]
    assert list(one["sig_landed"]) == [45, 5]
    assert list(one["ctrl_sec"]) == [260.0, 40.0]


def test_adapted_round_totals_equal_the_adapted_fight_totals(source):
    adapted = adapt(source)
    summed = adapted.rounds.groupby(["fight_id", "corner"])["sig_landed"].sum()
    totals = adapted.stats.set_index(["fight_id", "corner"])["sig_landed"]
    shared = summed.index.intersection(totals.index)

    assert len(shared) == 4  # two corners each for fights one and three
    assert (summed.loc[shared].values == totals.loc[shared].values).all()


def test_a_round_missing_a_corner_makes_the_fight_incomplete(source):
    stats = source["stats"]
    source["stats"] = stats[
        ~((stats["BOUT"] == "Alice Ace vs. Bob Blue") & (stats["FIGHTER"] == "Bob Blue"))
    ]
    adapted = adapt(source)

    assert FIGHT_ONE not in set(adapted.rounds["fight_id"])
    assert FIGHT_ONE not in complete_round_fights(adapted.rounds)


# --------------------------------------------------------------------------
# the standing stage
# --------------------------------------------------------------------------


@pytest.fixture
def primary(source):
    """Processed tables holding only fight one, and only its two fighters."""
    adapted = adapt(source)
    keep = adapted.fights["fight_id"] == FIGHT_ONE
    fighters = adapted.fighters[adapted.fighters["fighter_id"].isin([ALICE, BOB])]
    return (
        fighters.reset_index(drop=True),
        adapted.fights[keep].reset_index(drop=True),
        adapted.stats[adapted.stats["fight_id"] == FIGHT_ONE].reset_index(drop=True),
        adapted.rounds[adapted.rounds["fight_id"] == FIGHT_ONE].reset_index(drop=True),
    )


def test_apply_secondary_adds_the_fight_and_its_debutants(primary, source):
    result = apply_secondary(*primary, source)

    # Fight three is added; fight two is not, because it has no round record.
    assert set(result.fights["fight_id"]) == {FIGHT_ONE, FIGHT_THREE}
    assert result.secondary_fight_ids == frozenset({FIGHT_THREE})
    assert result.report["n_added"] == 1
    assert result.report["n_rejected_no_round_record"] == 1
    assert result.report["n_rejected_unknown_fighter"] == 0
    # Its two fighters were absent from the primary table and came along.
    assert result.secondary_fighter_ids == frozenset({ELI, FINN})
    assert set(result.fighters["fighter_id"]) == {ALICE, BOB, ELI, FINN}
    # ... and the tables stay consistent with each other.
    assert len(result.stats) == 2 * len(result.fights)
    assert set(result.rounds["fight_id"]) == {FIGHT_ONE, FIGHT_THREE}
    assert result.report["n_overlap"] == 1
    assert result.report["winner_agreement"] == 1.0


def test_apply_secondary_keeps_the_primary_row_on_an_overlapping_fight(primary, source):
    fighters, fights, stats, rounds = primary
    fights = fights.copy()
    fights.loc[0, "winner"] = "b"  # the primary disagrees with the secondary
    # floor=0 so the disagreement is merged rather than refused: this test is
    # about which row survives, not about the reconciliation gate.
    result = apply_secondary(fighters, fights, stats, rounds, source, floor=0.0)

    assert result.fights.set_index("fight_id").loc[FIGHT_ONE, "winner"] == "b"


def test_apply_secondary_is_idempotent(primary, source):
    once = apply_secondary(*primary, source)
    twice = apply_secondary(
        once.fighters, once.fights, once.stats, once.rounds, source
    )

    pd.testing.assert_frame_equal(once.fighters, twice.fighters)
    pd.testing.assert_frame_equal(once.fights, twice.fights)
    pd.testing.assert_frame_equal(once.stats, twice.stats)
    pd.testing.assert_frame_equal(once.rounds, twice.rounds)
    assert twice.report["n_added"] == 0


def test_apply_secondary_preserves_the_primary_dtypes(primary, source):
    fighters, fights, stats, rounds = primary
    result = apply_secondary(fighters, fights, stats, rounds, source)

    for merged, original in (
        (result.fighters, fighters), (result.fights, fights),
        (result.stats, stats), (result.rounds, rounds),
    ):
        assert merged.dtypes.to_dict() == original.dtypes.to_dict()


def test_apply_secondary_refuses_below_the_agreement_floor(primary, source):
    fighters, fights, stats, rounds = primary
    fights = fights.copy()
    fights.loc[0, "winner"] = "b"  # 0/1 agreement on the only overlapping fight
    with pytest.raises(SecondaryRefused, match="agreement"):
        apply_secondary(fighters, fights, stats, rounds, source, floor=0.99)


def test_a_fighter_the_secondary_cannot_describe_keeps_its_fight_out(primary, source):
    fighters, fights, stats, rounds = primary
    thinned = source | {
        "fighter_tott": source["fighter_tott"][
            source["fighter_tott"]["FIGHTER"] != "Finn Frost"
        ],
        "fighter_details": source["fighter_details"][
            source["fighter_details"]["LAST"] != "Frost"
        ],
    }
    result = apply_secondary(fighters, fights, stats, rounds, thinned)

    # With Finn unresolvable the bout name no longer resolves at all, so the
    # fight is dropped in adaptation rather than merged with a hole.
    assert set(result.fights["fight_id"]) == {FIGHT_ONE}
    assert result.report["n_unresolved_name"] == 1


# --------------------------------------------------------------------------
# fail-soft
# --------------------------------------------------------------------------


def test_merge_into_degrades_to_primary_when_the_fetch_fails(primary, capsys):
    def boom(base=None):
        raise OSError("connection reset")

    result = merge_into(*primary, fetch=boom)

    assert result.report["applied"] is False
    assert "connection reset" in result.report["error"]
    for merged, original in zip(
        (result.fighters, result.fights, result.stats, result.rounds), primary
    ):
        pd.testing.assert_frame_equal(merged, original)
    assert "WARNING" in capsys.readouterr().err


def test_merge_into_degrades_when_the_upstream_schema_changes(primary, source, capsys):
    broken = source | {"results": source["results"].drop(columns=["OUTCOME"])}
    result = merge_into(*primary, fetch=lambda base=None: broken)

    assert result.report["applied"] is False
    pd.testing.assert_frame_equal(result.fights, primary[1])
    assert "WARNING" in capsys.readouterr().err


def test_merge_into_degrades_when_the_source_disagrees(primary, source, capsys):
    fighters, fights, stats, rounds = primary
    fights = fights.copy()
    fights.loc[0, "winner"] = "b"
    result = merge_into(
        fighters, fights, stats, rounds, fetch=lambda base=None: source
    )

    assert result.report["applied"] is False
    assert "agreement" in result.report["error"]
    pd.testing.assert_frame_equal(result.fights, fights)
    assert "WARNING" in capsys.readouterr().err


def test_merge_into_applies_when_everything_is_well(primary, source):
    result = merge_into(*primary, fetch=lambda base=None: source)

    assert result.report["applied"] is True
    assert result.report["n_added"] == 1


# --------------------------------------------------------------------------
# provenance
# --------------------------------------------------------------------------


def test_provenance_labels_every_fight_and_fighter(primary, source):
    result = apply_secondary(*primary, source)
    provenance = build_provenance(
        result.fights, result.fighters,
        result.secondary_fight_ids, result.secondary_fighter_ids,
    )

    assert list(provenance.columns) == ["table", "row_id", "source"]
    assert set(provenance["source"]) <= {"primary", "secondary"}
    fights = provenance[provenance["table"] == "fights"]
    fighters = provenance[provenance["table"] == "fighters"]
    assert set(fights["row_id"]) == set(result.fights["fight_id"])
    assert set(fighters["row_id"]) == set(result.fighters["fighter_id"])
    assert set(fights.loc[fights["source"] == "secondary", "row_id"]) == {FIGHT_THREE}
    assert set(fighters.loc[fighters["source"] == "secondary", "row_id"]) == {ELI, FINN}
    # Deterministic order, so a rebuild that changes nothing writes nothing.
    assert provenance.equals(
        provenance.sort_values(["table", "row_id"]).reset_index(drop=True)
    )


def test_provenance_of_a_primary_only_build_is_all_primary(primary):
    fighters, fights, _stats, _rounds = primary
    provenance = build_provenance(fights, fighters, frozenset(), frozenset())

    assert set(provenance["source"]) == {"primary"}
    assert len(provenance) == len(fights) + len(fighters)
