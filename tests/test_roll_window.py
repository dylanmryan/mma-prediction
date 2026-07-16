from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

import scripts.roll_window as roll_window
from scripts.roll_window import (
    current_data_cutoff,
    decide_promotion,
    graded_fights_since,
    promotion_protocol_text,
)


def test_current_data_cutoff_is_max_date():
    features = pd.DataFrame({"date": pd.to_datetime(["2020-01-01", "2023-06-15", "2022-01-01"])})
    assert current_data_cutoff(features) == pd.Timestamp("2023-06-15")


def _event(event_date, fights):
    return {"event_name": "Test", "event_date": event_date, "fights": fights}


def test_graded_fights_since_excludes_events_before_cutoff():
    cutoff = pd.Timestamp("2026-01-01")
    events = [
        _event("2025-06-01", [{"skipped": False, "actual_winner": "x"}]),  # before cutoff
        _event("2026-06-01", [{"skipped": False, "actual_winner": "y"}]),  # after cutoff
    ]
    graded = graded_fights_since(events, cutoff)
    assert len(graded) == 1
    assert graded[0]["actual_winner"] == "y"


def test_graded_fights_since_excludes_skipped_and_ungraded():
    cutoff = pd.Timestamp("2026-01-01")
    events = [
        _event("2026-06-01", [
            {"skipped": True, "reason": "no match"},
            {"skipped": False},  # not yet graded
            {"skipped": False, "actual_winner": "z"},
        ]),
    ]
    graded = graded_fights_since(events, cutoff)
    assert len(graded) == 1
    assert graded[0]["actual_winner"] == "z"


def test_graded_fights_since_empty():
    assert graded_fights_since([], pd.Timestamp("2026-01-01")) == []


def test_promotion_protocol_text_mentions_threshold_and_margin():
    text = promotion_protocol_text(150, pd.Timestamp("2023-01-01"))
    assert "150" in text
    assert "0.002" in text
    assert "2023-01-01" in text


def test_decide_promotion_beats_margin():
    assert decide_promotion(new_log_loss=0.640, incumbent_log_loss=0.650) is True


def test_decide_promotion_within_margin_rejects():
    # 0.001 improvement -- below the 0.002 margin -> reject
    assert decide_promotion(new_log_loss=0.649, incumbent_log_loss=0.650) is False


def test_decide_promotion_worse_rejects():
    assert decide_promotion(new_log_loss=0.660, incumbent_log_loss=0.650) is False


def test_decide_promotion_custom_margin():
    assert decide_promotion(new_log_loss=0.640, incumbent_log_loss=0.650, margin=0.02) is False


# --- ensemble-based promotion gate (_execute) --------------------------------
#
# _execute retrains the FULL torch ensemble into a temp dir and gates on it.
# The real retrain is minutes long and the real evaluation loads torch nets,
# so both are mocked: the candidate retrain (a subprocess) is replaced with a
# stub that writes fake candidate artifacts, and the two ensemble evaluations
# are driven with controlled (incumbent_ll, candidate_ll) values. We never
# touch the real committed models/torch -- MODELS_DIR is monkeypatched to a
# tmp dir holding fake incumbent artifacts with distinctive bytes.

INCUMBENT_FILES = {
    "net_seed0.pt": b"INCUMBENT_NET_0",
    "net_seed1.pt": b"INCUMBENT_NET_1",
    "preprocess.json": b"INCUMBENT_PREP",
    "display_priors.json": b"INCUMBENT_PRIORS",
    "metrics_val.json": b"INCUMBENT_METRICS",
}
CANDIDATE_FILES = {
    "net_seed0.pt": b"CANDIDATE_NET_0",
    "net_seed1.pt": b"CANDIDATE_NET_1",
    "preprocess.json": b"CANDIDATE_PREP",
    "metrics_val.json": b"CANDIDATE_METRICS",
}


def _write_dir(directory: Path, files: dict[str, bytes]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for name, content in files.items():
        (directory / name).write_bytes(content)


def _read_dir(directory: Path) -> dict[str, bytes]:
    return {p.name: p.read_bytes() for p in directory.iterdir() if p.is_file()}


@pytest.fixture
def staged(tmp_path, monkeypatch):
    """A tmp MODELS_DIR with fake incumbent models/torch artifacts."""
    models_dir = tmp_path / "models"
    torch_dir = models_dir / "torch"
    _write_dir(torch_dir, INCUMBENT_FILES)
    monkeypatch.setattr(roll_window, "MODELS_DIR", models_dir)
    return models_dir, torch_dir


def _drive_execute(models_dir, monkeypatch, incumbent_ll, candidate_ll):
    """Run _execute with the retrain + both ensemble evals mocked."""
    candidate_torch_dir = models_dir / roll_window.CANDIDATE_DIR_NAME / "torch"

    def fake_retrain(out_dir, train_end, val_start, val_end):
        # stand in for the minutes-long train_torch.py subprocess
        _write_dir(Path(out_dir), CANDIDATE_FILES)

    def fake_eval(ensemble_dir, features, val_start, val_end):
        # incumbent scored on models/torch, candidate on the temp dir
        if roll_window.CANDIDATE_DIR_NAME in str(ensemble_dir):
            return candidate_ll
        return incumbent_ll

    monkeypatch.setattr(roll_window, "_retrain_candidate", fake_retrain)
    monkeypatch.setattr(roll_window, "_ensemble_val_log_loss", fake_eval)

    features = pd.DataFrame({"date": pd.to_datetime(["2020-01-01", "2026-06-01"])})
    roll_window._execute(features, cutoff=pd.Timestamp("2024-06-01"))
    return candidate_torch_dir


def test_execute_promotes_when_candidate_beats_margin(staged, monkeypatch, capsys):
    models_dir, torch_dir = staged
    candidate_torch_dir = _drive_execute(
        models_dir, monkeypatch, incumbent_ll=0.650, candidate_ll=0.640
    )

    # candidate artifacts staged into models/torch
    staged_now = _read_dir(torch_dir)
    for name, content in CANDIDATE_FILES.items():
        assert staged_now[name] == content

    # incumbent backed up (originals recoverable on disk)
    backup_dir = models_dir / roll_window.BACKUP_DIR_NAME
    assert backup_dir.exists()
    assert _read_dir(backup_dir) == INCUMBENT_FILES

    # temp candidate dir cleaned up
    assert not candidate_torch_dir.parent.exists()

    out = capsys.readouterr().out
    assert "PROMOTED" in out
    assert "STAGED" in out or "staged" in out


def test_execute_rejects_when_candidate_within_margin(staged, monkeypatch, capsys):
    models_dir, torch_dir = staged
    candidate_torch_dir = _drive_execute(
        models_dir, monkeypatch, incumbent_ll=0.650, candidate_ll=0.649
    )

    # incumbent artifacts byte-identical unchanged
    assert _read_dir(torch_dir) == INCUMBENT_FILES

    # temp dir cleaned + no backup created (torch was never touched)
    assert not candidate_torch_dir.parent.exists()
    assert not (models_dir / roll_window.BACKUP_DIR_NAME).exists()

    out = capsys.readouterr().out
    assert "REJECTED" in out


def test_execute_rejects_when_candidate_worse(staged, monkeypatch):
    models_dir, torch_dir = staged
    _drive_execute(models_dir, monkeypatch, incumbent_ll=0.650, candidate_ll=0.700)
    assert _read_dir(torch_dir) == INCUMBENT_FILES


def test_execute_always_cleans_temp_dir(staged, monkeypatch):
    """The candidate temp dir must never linger, promote or reject; and a
    reject leaves no backup dir behind."""
    models_dir, _ = staged
    candidate_root = models_dir / roll_window.CANDIDATE_DIR_NAME

    # promote path
    _drive_execute(models_dir, monkeypatch, incumbent_ll=0.650, candidate_ll=0.600)
    assert not candidate_root.exists()

    # reject path from a clean state (rebuild incumbent since promote
    # overwrote it; clear the promote's retained incumbent backup)
    import shutil as _shutil
    _shutil.rmtree(models_dir / roll_window.BACKUP_DIR_NAME, ignore_errors=True)
    _write_dir(models_dir / "torch", INCUMBENT_FILES)
    _drive_execute(models_dir, monkeypatch, incumbent_ll=0.650, candidate_ll=0.650)
    assert not candidate_root.exists()
    assert not (models_dir / roll_window.BACKUP_DIR_NAME).exists()


def test_execute_aborts_when_no_incumbent_ensemble(tmp_path, monkeypatch, capsys):
    models_dir = tmp_path / "models"
    (models_dir / "torch").mkdir(parents=True)  # empty -- no net_seed*.pt
    monkeypatch.setattr(roll_window, "MODELS_DIR", models_dir)

    called = {"retrain": False}

    def fake_retrain(*a, **k):
        called["retrain"] = True

    monkeypatch.setattr(roll_window, "_retrain_candidate", fake_retrain)
    features = pd.DataFrame({"date": pd.to_datetime(["2020-01-01", "2026-06-01"])})
    roll_window._execute(features, cutoff=pd.Timestamp("2024-06-01"))

    assert called["retrain"] is False  # aborted before any retrain
    assert "No incumbent" in capsys.readouterr().out
