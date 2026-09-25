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


def _score(at):
    [b for b in at.button if b.label == "Score pair"][0].click().run()
    assert not at.exception, at.exception
    return at


def _all_text(at):
    return " ".join(str(x.value) for x in [*at.markdown, *at.caption, *at.error, *at.info])


def test_single_pair_prediction_renders(app, no_pubchem):
    text = _all_text(_score(app))
    assert re.search(r"Model score: (High|Medium|Low)", text)
    assert not re.search(r"(HIGH|MEDIUM|LOW) RISK", text)  # old clinical-style badge
    assert "co-administer" not in text and "patient" not in text
    assert len(app.get("imgs")) == 2


def test_smiles_input_works(app):
    app.radio[0].set_value("SMILES").run()
    text = _all_text(_score(app))
    assert re.search(r"Model score: (High|Medium|Low)", text)


def test_misspelled_name_shows_suggestion(app, no_pubchem):
    app.text_input(key="na").set_value("Asprin").run()
    assert any("Aspirin" in e.value for e in _score(app).error)


def test_unreliable_input_shows_warning(app, no_pubchem):
    app.text_input(key="nb").set_value("Sodium Chloride").run()
    assert any("inorganic" in w.value for w in _score(app).warning)


def test_other_pages_render(app):
    app.sidebar.radio[0].set_value("Batch scoring").run()
    assert not app.exception, app.exception
    app.sidebar.radio[0].set_value("About the model").run()
    assert not app.exception, app.exception
    labels = [m.label for m in app.metric]
    assert "Drugs seen in training" in labels
    assert "Drugs never seen in training" in labels
