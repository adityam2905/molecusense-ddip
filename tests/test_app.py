"""Smoke test of the Streamlit app with the deployed model (no network needed)."""

import os
import re

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest

from tests.conftest import ROOT

APP = os.path.join(ROOT, "app", "streamlit_app.py")


@pytest.fixture
def app():
    at = AppTest.from_file(APP, default_timeout=180).run()
    assert not at.exception, at.exception
    return at


def _all_text(at):
    return " ".join(str(x.value) for x in [*at.markdown, *at.caption, *at.error, *at.info])


def test_single_pair_prediction_renders(app, no_pubchem):
    [b for b in app.button if "Execute Prediction" in b.label][0].click().run()
    assert not app.exception, app.exception
    text = _all_text(app)
    assert re.search(r"Model score: (High|Medium|Low)", text)
    assert not re.search(r"(HIGH|MEDIUM|LOW) RISK", text)  # old clinical-style badge
    assert "co-administer" not in text and "patient" not in text
    assert len(app.get("imgs")) == 2


def test_misspelled_name_shows_suggestion(app, no_pubchem):
    app.text_input(key="na").set_value("Asprin").run()
    [b for b in app.button if "Execute Prediction" in b.label][0].click().run()
    assert any("Aspirin" in e.value for e in app.error)


def test_system_info_page(app):
    app.sidebar.selectbox[0].select("System Info").run()
    assert not app.exception, app.exception
    labels = [m.label for m in app.metric]
    assert "New pairs of known drugs" in labels
    assert "Drugs never seen in training" in labels
