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
