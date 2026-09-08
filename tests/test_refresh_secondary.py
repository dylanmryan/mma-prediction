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
    UnknownFighterError,
    adapt,
    build_name_index,
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
    fights, _stats, report = adapt(source)

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
    fights, _stats, _report = adapt(source)
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
    _fights, stats, _report = adapt(source)

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
    fights, _stats, report = adapt(source)

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


def test_dry_run_writes_nothing(monkeypatch, tmp_path, capsys, source):
    import scripts.refresh_secondary as mod

    monkeypatch.setattr(mod, "fetch_source", lambda base=None: source)
    written: list = []
    monkeypatch.setattr(mod.pd.DataFrame, "to_parquet", lambda self, *a, **k: written.append(a))

    # A "primary" holding only the first of the three fights, in our real
    # processed schema, so the reconciliation guard has a genuine overlap.
    adapted_fights, adapted_stats, _ = mod.adapt(source)
    primary = adapted_fights[adapted_fights["fight_id"] == FIGHT_ONE].reset_index(drop=True)
    primary_stats = adapted_stats[adapted_stats["fight_id"] == FIGHT_ONE].reset_index(drop=True)
    known = {ALICE, BOB, CARA, DANA, ELI, FINN}
    monkeypatch.setattr(mod, "load_primary", lambda: (primary, primary_stats, known))

    exit_code = mod.main([])

    assert exit_code == 0
    assert written == []
    out = capsys.readouterr().out
    assert "n_added" in out
    assert "dry run" in out.lower()
    assert "2" in out  # the two fights the secondary adds
