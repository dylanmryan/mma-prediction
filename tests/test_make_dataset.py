"""The standing secondary stage as `scripts/make_dataset.py` runs it.

The builders themselves are tested in `test_dataset_*.py` and the merge in
`test_refresh_secondary.py`; what is tested here is the seam -- that the
stage can be switched off, that it never needs the network to be switched
off, and that a build always writes a provenance row for every row it
wrote.
"""
from __future__ import annotations

import pandas as pd

from scripts.make_dataset import secondary_stage


def _tables():
    fighters = pd.DataFrame({"fighter_id": pd.array(["A", "B"], dtype="string")})
    fights = pd.DataFrame(
        {
            "fight_id": pd.array(["f1"], dtype="string"),
            "date": pd.to_datetime(["2026-08-01"]),
        }
    )
    stats = pd.DataFrame({"fight_id": pd.array(["f1", "f1"], dtype="string")})
    rounds = pd.DataFrame({"fight_id": pd.array(["f1"], dtype="string")})
    return fighters, fights, stats, rounds


def test_the_stage_is_a_clean_noop_when_switched_off(monkeypatch):
    import scripts.make_dataset as mod

    def unreachable(*args, **kwargs):
        raise AssertionError("--no-secondary must not touch the network")

    monkeypatch.setattr(mod, "merge_into", unreachable)
    tables = _tables()
    result = secondary_stage(*tables, enabled=False)

    assert result.report["applied"] is False
    assert "disabled" in result.report["error"]
    assert result.secondary_fight_ids == frozenset()
    assert result.secondary_fighter_ids == frozenset()
    for merged, original in zip(
        (result.fighters, result.fights, result.stats, result.rounds), tables
    ):
        pd.testing.assert_frame_equal(merged, original)


def test_the_stage_delegates_to_the_fail_soft_merge(monkeypatch):
    import scripts.make_dataset as mod

    seen = {}

    def fake(fighters, fights, stats, rounds, **kwargs):
        seen["called"] = True
        return mod.SecondaryResult(
            fighters, fights, stats, rounds, frozenset({"f2"}), frozenset(),
            {"applied": True, "error": None},
        )

    monkeypatch.setattr(mod, "merge_into", fake)
    result = secondary_stage(*_tables(), enabled=True)

    assert seen["called"] is True
    assert result.secondary_fight_ids == frozenset({"f2"})
