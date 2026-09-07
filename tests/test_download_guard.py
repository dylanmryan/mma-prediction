import pytest

from scripts.download_data import REQUIRED_FILES, verify_raw_files


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
