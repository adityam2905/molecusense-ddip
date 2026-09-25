import os

import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ASPIRIN = "CC(=O)Oc1ccccc1C(=O)O"
IBUPROFEN = "CC(C)Cc1ccc(cc1)C(C)C(=O)O"
WARFARIN = "CC(=O)C(c1ccccc1)C1=C(O)c2ccccc2OC1=O"
CAFFEINE = "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"
METFORMIN = "CN(C)C(=N)N=C(N)N"


@pytest.fixture(scope="session")
def inference():
    """The committed, deployed model."""
    from utils.inference import DDIInference
    return DDIInference(checkpoint_dir=os.path.join(ROOT, "checkpoints"))


@pytest.fixture(scope="session")
def random_model():
    """An untrained model: properties that must hold for ANY weights."""
    from models.gnn_ddi import DDIPredictor
    torch.manual_seed(0)
    return DDIPredictor(dropout=0.0).eval()


@pytest.fixture
def no_pubchem(monkeypatch):
    """Tests must not depend on the network."""
    import utils.drug_lookup as dl
    monkeypatch.setattr(dl, "pubchem_smiles", lambda *a, **k: None)
