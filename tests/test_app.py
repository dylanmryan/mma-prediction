from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]

pytestmark = pytest.mark.skipif(
    not (ROOT / "models" / "torch" / "metrics_val.json").exists(),
    reason="ensemble artifacts not built",
)


@pytest.mark.skipif(
    pd.__version__.startswith("3."),
    reason=(
        "Streamlit AppTest thread crashes with pandas 3.x string_arrow indexing "
        "(environment issue, app verified via headless boot)"
    ),
)
def test_app_boots_and_predicts():
    from streamlit.testing.v1 import AppTest

    app_test = AppTest.from_file(str(ROOT / "app.py"), default_timeout=120)
    app_test.run()
    assert not app_test.exception
    names = app_test.selectbox[0].options
    assert len(names) > 1000
    app_test.selectbox[0].select(names[0])
    app_test.selectbox[1].select(names[1])
    app_test.run()
    assert not app_test.exception
    assert app_test.subheader[0].value == "Prediction"


def test_the_model_card_quotes_the_deployed_forms_calibration_not_the_harness_form():
    """The card describes the scorer a reader is about to use.

    That scorer applies ONE fixed post-average temperature; the walk-forward
    report it used to quote scores a form that fits a temperature per fold on
    each fold's inner-validation year, and the two have different ECEs (0.0108
    against 0.0124). The harness number stays in the card, labelled as the
    harness's, so the card and the report cannot be read as contradicting each
    other.
    """
    import json
    import sys

    sys.path.insert(0, str(ROOT))
    import app
    from mma.inference import load_blend_config

    deployed = json.loads(app.BLEND_TEMPERATURE.read_text())["deployed"]
    harness = json.loads(app.BLEND_REPORT.read_text())["pooled"]
    config = load_blend_config()
    text = app.model_card_text(config["weight"], config["temperature"])

    assert f"T={config['temperature']:g}" in text
    assert f"ECE {deployed['honest_pooled']['ece']['10']:.4f}" in text
    assert f"{deployed['honest_pooled']['winner_log_loss']:.3f} log-loss" in text
    # the harness figure appears only inside its own label
    assert f"per-fold-fitted form reads {harness['ece']:.4f}" in text
    assert f"ECE {harness['ece']:.4f}" not in text


def _import_app():
    import sys

    sys.path.insert(0, str(ROOT))
    import app

    return app


def test_outcome_rows_show_every_cell_exactly_once_in_reading_order():
    """The outcome table is the user-visible payoff of the simulator, so it has
    to be the WHOLE distribution -- every cell once, nothing double-counted."""
    import numpy as np

    from mma.joint import cells_to_dict, compose_joint_cells
    from mma.models.train_loop import METHOD_CLASSES, ROUND_CLASSES

    app = _import_app()
    cells = compose_joint_cells(
        [0.6], [[0.4, 0.2, 0.4]], [[0.4, 0.3, 0.2, 0.1]], METHOD_CLASSES, ROUND_CLASSES)
    joint = cells_to_dict(cells[0], METHOD_CLASSES, ROUND_CLASSES)
    rows = app.outcome_rows(joint, METHOD_CLASSES, ROUND_CLASSES)

    assert [row[0] for row in rows] == [
        "KO/TKO in round 1", "KO/TKO in round 2", "KO/TKO in round 3",
        "KO/TKO in rounds 4-5",
        "Submission in round 1", "Submission in round 2", "Submission in round 3",
        "Submission in rounds 4-5",
        "Decision",
    ]
    assert sum(row[1] + row[2] for row in rows) == pytest.approx(1.0)
    # each column is that corner's win probability, because they are one
    # distribution rather than three heads multiplied together
    assert sum(row[1] for row in rows) == pytest.approx(0.6)
    assert sum(row[2] for row in rows) == pytest.approx(0.4)
    assert np.isclose([row[1] for row in rows][:4], cells[0][:4]).all()


def test_outcome_rows_drop_rounds_a_three_round_bout_cannot_reach():
    from mma.joint import cells_to_dict, compose_joint_cells
    from mma.models.train_loop import METHOD_CLASSES, ROUND_CLASSES

    app = _import_app()
    cells = compose_joint_cells(
        [0.5], [[0.4, 0.2, 0.4]], [[0.5, 0.3, 0.2, 0.0]], METHOD_CLASSES, ROUND_CLASSES)
    joint = cells_to_dict(cells[0], METHOD_CLASSES, ROUND_CLASSES)
    rows = app.outcome_rows(joint, METHOD_CLASSES, ROUND_CLASSES)
    assert not any("rounds 4-5" in row[0] for row in rows)
    assert sum(row[1] + row[2] for row in rows) == pytest.approx(1.0)


def test_the_model_card_describes_the_hybrid_and_names_the_simulator_evidence():
    """The card must say what the reader is looking at: a win probability from
    the blend and an outcome table from the simulator, with the simulator's own
    evidence quoted rather than folded into winner metrics."""
    from mma.inference import load_blend_config

    app = _import_app()
    config = load_blend_config()
    text = app.model_card_text(config["weight"], config["temperature"])
    assert "hybrid" in text and "simulator" in text
    assert "joint-outcome log-loss of 2.1432" in text
    assert "leaves the win probability itself unchanged" in text
    # 10,000 is the per-pass run count, not the per-matchup one: `simulate_fights`
    # plays each fight from both corners, and `predict_symmetrized` runs the
    # whole thing again in the mirrored corner ordering. Four passes, 40,000.
    assert "40,000" in text
    assert "10,000 simulated fights per matchup" not in text


def test_the_market_section_reads_the_out_of_fold_benchmark():
    """The app's market panel must quote OUT-OF-FOLD numbers.

    The deployed model is refit through the latest event, so it trained on
    every fight in the odds dataset; scoring them with its own predictions
    makes it look like it beats the market. The artifact the app reads is
    therefore the out-of-fold one, and this pins both the file it reads and
    the keys it reads out of it -- a rename that silently fell back to the
    frozen July artifact would put a different model's numbers under a
    caption describing this one.
    """
    import json

    source = (ROOT / "app.py").read_text()
    assert 'models" / "market_benchmark_oof.json"' in source
    assert '"headline_out_of_fold"' in source

    artifact = ROOT / "models" / "market_benchmark_oof.json"
    if not artifact.exists():
        pytest.skip("out-of-fold benchmark not built")
    benchmark = json.loads(artifact.read_text())
    head = benchmark["headline_out_of_fold"]
    assert {"n_fights", "model", "market", "calibration", "roi"} <= set(head)
    assert benchmark["provenance"]["fold_years"]
    for threshold in ("0.00", "0.10"):
        assert "roi_pct" in head["roi"][threshold]["favorite_edge_on_a"]
    july = benchmark["comparison_with_frozen_july_artifact"]
    assert {"direction", "same_fight_set", "july_2026_artifact",
            "out_of_fold_2021_plus"} <= set(july)
