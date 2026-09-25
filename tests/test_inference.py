"""The deployed checkpoint loads and scores pairs; name lookup works offline."""

import pytest

from utils.calibration import RISK_TEXT, risk_from_percentile
from utils.drug_lookup import resolve, suggest
from tests.conftest import ASPIRIN, IBUPROFEN


def test_checkpoint_loads_calibrated(inference):
    assert inference.calibrated
    assert inference.temperature > 0
    assert len(inference.reference_logits) > 100


def test_one_prediction_works(inference):
    res = inference.predict(smiles_a=ASPIRIN, smiles_b=IBUPROFEN, fetch_smiles=False)
    assert res["error"] is None
    assert 0.0 <= res["probability"] <= 1.0
    assert 0.0 <= res["percentile"] <= 100.0
    assert res["risk"]["level"] in {"HIGH", "MEDIUM", "LOW"}


def test_same_molecule_is_rejected(inference):
    # the same molecule written two different ways
    res = inference.predict(smiles_a=ASPIRIN, smiles_b="OC(=O)c1ccccc1OC(C)=O", fetch_smiles=False)
    assert res["error"] and "same molecule" in res["error"]


def test_invalid_smiles_is_rejected(inference):
    res = inference.predict(smiles_a="not-a-molecule", smiles_b=ASPIRIN, fetch_smiles=False)
    assert res["error"]


def test_names_resolve_from_local_cache(no_pubchem):
    for name in ("Aspirin", "aspirin", "  WARFARIN ", "paracetamol"):
        found = resolve(name)
        assert found["smiles"] and found["source"] == "cache", name


def test_misspelling_gets_a_suggestion(no_pubchem, inference):
    assert "Aspirin" in suggest("Asprin")
    res = inference.predict(name_a="Asprin", name_b="Ibuprofen")
    assert res["error"] and "Aspirin" in res["error"]


def test_risk_wording_is_neutral():
    clinical = ("co-administer", "patient", "caution", "review", "monitor")
    for text in RISK_TEXT.values():
        assert not any(word in text.lower() for word in clinical)


@pytest.mark.parametrize("pct,level", [(94.9, "MEDIUM"), (95.0, "HIGH"), (79.9, "LOW"), (80.0, "MEDIUM")])
def test_displayed_percentile_matches_band(pct, level):
    risk = risk_from_percentile(pct)
    assert risk["level"] == level
    shown = int(risk["description"].split("above ")[1].split("%")[0]) if level != "LOW" else None
    if shown is not None:
        assert (shown >= 95) == (level == "HIGH")
