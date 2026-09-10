"""The weekly snapshot-coverage check (scripts/check_snapshot_coverage.py).

The arithmetic is `mma.decay`'s and is tested there. What this pins is the
part that has to hold for the check to be worth wiring into the Action at all:

* that it WARNS -- a check that measures decay and stays quiet about it is the
  rot it exists to prevent;
* that the warning fires on the TRAILING window rather than on the overall
  share, which is the whole point: overall coverage is dominated by a decade
  of well-covered history and would never breach;
* that a missing `external_missing` column is a loud failure rather than a
  silently skipped check, since that column is kept in the table (and out of
  both model matrices) for exactly this purpose.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

import scripts.check_snapshot_coverage as check


def table(rows) -> pd.DataFrame:
    dates, missing = zip(*rows)
    return pd.DataFrame({"date": pd.to_datetime(list(dates)),
                         check.FLAG: list(missing)})


def steady(share: float, *, n: int = 100, year: str = "2026") -> pd.DataFrame:
    """`n` fights spread over one year, `share` of them unmapped."""
    days = pd.date_range(f"{year}-01-01", periods=n, freq="3D")
    missing = [i < round(share * n) for i in range(n)]
    return pd.DataFrame({"date": days, check.FLAG: missing})


def test_a_covered_recent_year_does_not_breach():
    report = check.measure(steady(0.20))
    assert report["trailing"]["share"] == 0.20
    assert report["breached"] is False
    assert check.warning_text(report) is None


def test_a_decayed_recent_year_breaches_and_names_what_to_do():
    report = check.measure(steady(0.55))
    assert report["breached"] is True
    warning = check.warning_text(report)
    assert warning.startswith("WARNING:")
    assert "0.550" in warning
    assert "external-decay-decision" in warning
    assert "build_external.py" in warning


def test_the_threshold_is_exclusive_so_sitting_on_it_is_not_a_breach():
    report = check.measure(steady(0.40))
    assert report["trailing"]["share"] == pytest.approx(check.THRESHOLD)
    assert report["breached"] is False


def test_the_warning_reads_the_trailing_window_not_the_overall_share():
    """A decade of covered history must not mask a decayed present -- this is
    the failure mode the trailing window exists for."""
    history = [(f"20{y:02d}-06-01", False) for y in range(10, 25)] * 20
    recent = [("2026-0%d-01" % m, True) for m in range(1, 9)] * 2
    report = check.measure(table(history + recent))
    assert report["overall"]["share"] < check.THRESHOLD  # the masking share
    assert report["trailing"]["share"] == 1.0
    assert report["breached"] is True


def test_by_year_carries_the_decay_curve():
    report = check.measure(table([
        ("2024-05-01", False), ("2024-06-01", False), ("2024-07-01", True),
        ("2026-05-01", True), ("2026-06-01", True), ("2026-07-01", False),
    ]))
    assert report["by_year"]["2024"]["share"] == 0.3333
    assert report["by_year"]["2026"]["share"] == 0.6667


def test_a_table_without_the_flag_fails_loudly():
    with pytest.raises(SystemExit, match="snapshot coverage cannot be measured"):
        check.measure(pd.DataFrame({"date": pd.to_datetime(["2026-01-01"])}))


def test_the_window_is_anchored_on_the_tables_own_last_date_not_on_today():
    """Otherwise the number drifts between runs that read the same table, and
    a stale table would report a clean window that contains nothing."""
    report = check.measure(steady(1.0, year="2010"))
    assert report["trailing"]["window_end"] == "2010-10-25"
    assert report["trailing"]["n"] == 100


def test_an_empty_trailing_window_is_not_a_breach():
    """No recent rows is nothing to say, not a clean bill of health -- and in
    particular not a NaN share quietly comparing False against the threshold.
    Every one of those rows IS unmapped, so a naive read would breach."""
    report = check.measure(steady(1.0, year="2010"), as_of="2026-01-01")
    assert report["trailing"]["n"] == 0
    assert report["breached"] is False
    assert check.warning_text(report) is None


def test_main_writes_the_artifact_and_exits_zero_even_when_breached(tmp_path, capsys):
    """It is a report, not a gate: the Action must never fail on it."""
    features = tmp_path / "features.parquet"
    steady(0.9).to_parquet(features)
    out = tmp_path / "snapshot_coverage.json"
    check.main(["--features", str(features), "--out", str(out)])
    report = json.loads(out.read_text())
    assert report["breached"] is True
    assert "WARNING:" in capsys.readouterr().err


def test_the_committed_threshold_sits_above_the_2025_fold_it_was_set_from():
    """0.359 is the 2025 fold's external_missing, the worst fold the decay
    decision was measured on; a threshold at or below it would fire on the
    state that decision already accounts for."""
    assert check.THRESHOLD > 0.359
    assert check.THRESHOLD < 0.534  # and below 2026's, so it can still fire
