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
