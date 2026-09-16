"""The repository stays legible, or the suite says so.

Two documents exist purely so a reader can tell what matters: `scripts/README.md`
says which of the thirty-eight scripts actually run, and `README.md` points at
`docs/EXPERIMENTS.md` for the detail it no longer carries itself. Both rot
silently -- a new script simply never gets classified, and nobody notices until
the map is worse than no map.
"""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_every_script_is_classified_in_the_scripts_map():
    """A new script must be filed under a role. If this fails, add it to
    `scripts/README.md` under the heading that describes what it is for --
    weekly pipeline, run by hand, an instrument, or a recorded experiment."""
    listed = set(re.findall(r"`([a-z_0-9]+\.py)`",
                            (ROOT / "scripts" / "README.md").read_text()))
    actual = {p.name for p in (ROOT / "scripts").glob("*.py")}
    unclassified = sorted(actual - listed)
    assert not unclassified, (
        f"not described in scripts/README.md: {unclassified}"
    )


def test_the_scripts_map_names_no_script_that_has_been_deleted():
    listed = set(re.findall(r"`([a-z_0-9]+\.py)`",
                            (ROOT / "scripts" / "README.md").read_text()))
    actual = {p.name for p in (ROOT / "scripts").glob("*.py")}
    assert not sorted(listed - actual), (
        f"scripts/README.md describes files that no longer exist: "
        f"{sorted(listed - actual)}"
    )


def test_the_weekly_pipeline_section_matches_the_workflow():
    """The map's headline claim is which scripts run on a schedule. If the
    workflow gains or drops a step, the claim has to move with it."""
    workflow = (ROOT / ".github" / "workflows" / "refresh-data.yml").read_text()
    # INVOKED, not merely mentioned: the workflow names `roll_window.py` in a
    # comment explaining what replaced it, and a comment is not a step.
    invoked = set(re.findall(r"run:\s*python\s+scripts/([a-z_0-9]+\.py)",
                             workflow))
    pipeline = (ROOT / "scripts" / "README.md").read_text().split(
        "## Run by hand")[0]
    described = set(re.findall(r"`([a-z_0-9]+\.py)", pipeline))
    assert invoked <= described, (
        f"the weekly workflow runs scripts the map does not list under the "
        f"pipeline: {sorted(invoked - described)}"
    )


def test_the_readme_stays_a_front_door():
    """It was 1,481 lines and unreadable; the detail lives in
    docs/EXPERIMENTS.md now. This is a ratchet, not a style rule -- if the
    README needs to grow past this, the material probably belongs in the
    experiments record instead."""
    readme = (ROOT / "README.md").read_text().splitlines()
    assert len(readme) < 400, (
        f"README.md is {len(readme)} lines; move detail into "
        "docs/EXPERIMENTS.md rather than growing the front door"
    )


def test_the_readme_points_at_the_full_record():
    assert "docs/EXPERIMENTS.md" in (ROOT / "README.md").read_text()
    assert (ROOT / "docs" / "EXPERIMENTS.md").exists()


def test_the_readme_still_states_the_losing_result():
    """The market comparison is the honest headline. If a future edit buries
    it, that is exactly the kind of drift worth failing a build over."""
    readme = (ROOT / "README.md").read_text().lower()
    assert "market" in readme
    assert "does not beat the betting market" in readme
