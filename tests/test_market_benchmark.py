"""Structural checks on the committed model-vs-market benchmark artifact.

These tests only validate the shape and plausibility of the committed
models/market_benchmark.json -- they never recompute anything, never hit
the network, and must never trigger a rerun of scripts/build_odds_benchmark.py.
"""
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_PATH = ROOT / "models" / "market_benchmark.json"

pytestmark = pytest.mark.skipif(
    not BENCHMARK_PATH.exists(),
    reason="market benchmark not built yet",
)


@pytest.fixture(scope="module")
def benchmark():
    return json.loads(BENCHMARK_PATH.read_text())


def test_top_level_keys_present(benchmark):
    assert {
        "computed_once_on", "odds_dataset", "note", "alignment",
        "headline_2021_plus", "all_matched_fights_secondary",
    } <= set(benchmark)


def test_alignment_counts_sane(benchmark):
    alignment = benchmark["alignment"]
    assert alignment["n_aligned_by_id"] > 2000
    assert alignment["n_aligned_by_id"] + alignment["n_aligned_by_name"] > 2000
    assert alignment["n_skipped"] >= 0


def test_headline_n_fights_in_sane_range(benchmark):
    # The model's validation+test era (2021+) has ~2,380 decisive fights in
    # features.parquet; odds coverage is not 100%, but should be substantial.
    n = benchmark["headline_2021_plus"]["n_fights"]
    assert 500 < n < 3000, n


def test_headline_log_losses_plausible(benchmark):
    headline = benchmark["headline_2021_plus"]
    for label in ("model", "market"):
        block = headline[label]
        assert 0.5 < block["log_loss"] < 0.8, (label, block)
        assert 0.0 < block["brier"] < 0.3, (label, block)
        assert 0.3 < block["accuracy"] < 0.85, (label, block)


def test_market_log_loss_is_sharp(benchmark):
    """A liquid closing-line betting market should be well calibrated and
    sharp -- log-loss noticeably better than a coin flip (0.693), landing
    in the ~0.55-0.68 range typical of UFC moneylines. A wildly different
    number is the strongest signal that odds/corner alignment is broken."""
    market_ll = benchmark["headline_2021_plus"]["market"]["log_loss"]
    assert 0.5 < market_ll < 0.72, market_ll


def test_honesty_gate_model_does_not_implausibly_dominate_market(benchmark):
    """Markets are sharp; if the model beats the market by a wide margin on
    log-loss, that's a red flag for leakage or misalignment, not a genuine
    edge. Mirrors the gate documented in scripts/build_odds_benchmark.py."""
    delta = benchmark["headline_2021_plus"]["delta_model_minus_market"]["log_loss"]
    assert delta > -0.02, (
        f"model beats market by {-delta:.4f} log-loss -- investigate "
        "alignment/leakage before trusting this artifact"
    )


def test_calibration_tables_have_ten_bins(benchmark):
    calibration = benchmark["headline_2021_plus"]["calibration"]
    for label in ("model", "market"):
        table = calibration[label]
        assert len(table) == 10
        total_count = sum(row["count"] for row in table)
        assert total_count == benchmark["headline_2021_plus"]["n_fights"]


def test_roi_sweep_has_all_thresholds(benchmark):
    roi = benchmark["headline_2021_plus"]["roi"]
    assert set(roi) == {"0.00", "0.05", "0.10"}
    for block in roi.values():
        for side in ("favorite_edge_on_a", "underdog_edge_on_b"):
            assert "n_bets" in block[side]
            assert "roi_pct" in block[side]
            assert block[side]["n_bets"] >= 0


def test_secondary_cut_covers_at_least_headline(benchmark):
    # all_matched_fights_secondary includes pre-2021 (training-era) fights
    # too, so it should never have FEWER fights than the headline cut.
    assert (
        benchmark["all_matched_fights_secondary"]["n_fights"]
        >= benchmark["headline_2021_plus"]["n_fights"]
    )
