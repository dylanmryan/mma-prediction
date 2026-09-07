import pytest

import scripts.download_data as download_data
from scripts.download_data import REQUIRED_FILES, download, verify_raw_files


def test_all_required_files_present_passes(tmp_path):
    for name in REQUIRED_FILES:
        (tmp_path / name).write_text("x")
    verify_raw_files(tmp_path)  # no raise


def test_missing_file_raises_with_names(tmp_path):
    (tmp_path / "master.csv").write_text("x")
    with pytest.raises(FileNotFoundError) as excinfo:
        verify_raw_files(tmp_path)
    assert "round.csv" in str(excinfo.value)
    assert "master.csv" not in str(excinfo.value).split("missing")[-1]


def test_required_files_are_the_new_layout():
    assert set(REQUIRED_FILES) == {"master.csv", "fighter.csv", "round.csv", "fighter_bonus.csv"}


def test_nested_snapshot_layout_passes(tmp_path):
    sub = tmp_path / "sub"
    sub.mkdir()
    for name in REQUIRED_FILES:
        (sub / name).write_text("x")
    verify_raw_files(tmp_path)  # no raise: required files found via rglob


def test_download_clears_stale_csvs_and_verifies_snapshot(tmp_path, monkeypatch):
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    for name in REQUIRED_FILES:
        (snapshot / name).write_text("x")
    monkeypatch.setattr(
        download_data.kagglehub, "dataset_download", lambda dataset: str(snapshot)
    )

    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    stale = raw_dir / "UFC.csv"
    stale.write_text("stale")

    download(raw_dir)

    assert not stale.exists()
    for name in REQUIRED_FILES:
        assert (raw_dir / name).exists()


def test_download_rejects_incomplete_snapshot(tmp_path, monkeypatch):
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    for name in REQUIRED_FILES:
        if name == "round.csv":
            continue
        (snapshot / name).write_text("x")
    monkeypatch.setattr(
        download_data.kagglehub, "dataset_download", lambda dataset: str(snapshot)
    )

    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    stale = raw_dir / "UFC.csv"
    stale.write_text("stale")

    with pytest.raises(FileNotFoundError, match="round.csv"):
        download(raw_dir)

    # verify-before-touching-raw_dir: an incomplete snapshot must not clear
    # out whatever raw_dir already held.
    assert stale.exists()
