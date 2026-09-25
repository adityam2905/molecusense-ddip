"""The app says when a score is less trustworthy instead of looking equally confident."""

import requests

import utils.drug_lookup as dl
from utils.mol_graph import structure_warnings
from tests.conftest import ASPIRIN, IBUPROFEN

# Built from scratch, so certainly not a drug in TWOSIDES.
NOVEL = "CCCCCCCCCCCCCCCCCCCC(=O)NC1=CC=C(C=C1)S(=O)(=O)N"


def test_deployed_model_knows_its_training_drugs(inference):
    assert inference.training_drugs and len(inference.training_drugs) > 500
    assert inference.seen_in_training(ASPIRIN) is True
    assert inference.seen_in_training(NOVEL) is False


def test_known_pair_has_no_notes(inference):
    res = inference.predict(smiles_a=ASPIRIN, smiles_b=IBUPROFEN, fetch_smiles=False)
    assert res["notes"] == []


def test_unseen_drug_is_flagged(inference):
    res = inference.predict(smiles_a=NOVEL, smiles_b=ASPIRIN, name_a="Novel", name_b="Aspirin",
                            fetch_smiles=False)
    assert res["seen_a"] is False and res["seen_b"] is True
    assert any("Novel was not in the training data" in n for n in res["notes"])


def test_structure_warnings():
    assert structure_warnings(ASPIRIN) == []
    assert any("inorganic" in w for w in structure_warnings("[Na+].[Cl-]"))
    assert any("very small" in w for w in structure_warnings("C"))
    salt = structure_warnings("C(C(=O)[O-])C(CC(=O)[O-])(C(=O)[O-])O.[K+].[K+].[K+]")
    assert any("separate parts" in w for w in salt)


def test_table_salt_is_flagged(inference, no_pubchem):
    res = inference.predict(name_a="Warfarin", name_b="Sodium Chloride")
    assert res["error"] is None
    assert any("inorganic" in n for n in res["notes"])


def test_charcoal_is_refused(inference, no_pubchem):
    res = inference.predict(name_a="Activated Charcoal", name_b="Aspirin")
    assert res["error"] and "not a single molecule" in res["error"]


def test_stereoisomers_get_an_explanation(inference, no_pubchem):
    res = inference.predict(name_a="Ephedrine", name_b="Pseudoephedrine")
    assert res["error"] and "stereochemistry" in res["error"]


def test_pubchem_url_escapes_slashes(monkeypatch):
    urls = []

    class Resp:
        status_code = 404

    monkeypatch.setattr(requests, "get", lambda url, timeout: urls.append(url) or Resp())
    assert dl.pubchem_smiles("sulfamethoxazole/trimethoprim") is None
    assert "sulfamethoxazole%2Ftrimethoprim/property" in urls[0]
