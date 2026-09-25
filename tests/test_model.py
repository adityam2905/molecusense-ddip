"""The model's core guarantees: order invariance and meaningful attention."""

import numpy as np
import pytest
import torch
from torch_geometric.data import Batch

from utils.mol_graph import smiles_to_graph
from tests.conftest import ASPIRIN, IBUPROFEN, WARFARIN, CAFFEINE, METFORMIN

PAIRS = [(ASPIRIN, IBUPROFEN), (WARFARIN, CAFFEINE), (METFORMIN, ASPIRIN)]


def _batch(smiles):
    return Batch.from_data_list([smiles_to_graph(smiles)])


@pytest.mark.parametrize("a,b", PAIRS)
def test_random_model_is_order_invariant(random_model, a, b):
    with torch.no_grad():
        ab = random_model(_batch(a), _batch(b))
        ba = random_model(_batch(b), _batch(a))
    assert torch.allclose(ab, ba, atol=1e-6)


@pytest.mark.parametrize("a,b", PAIRS)
def test_deployed_model_is_order_invariant(inference, a, b):
    ab = inference.predict(smiles_a=a, smiles_b=b, fetch_smiles=False)
    ba = inference.predict(smiles_a=b, smiles_b=a, fetch_smiles=False)
    assert ab["error"] is None and ba["error"] is None
    assert ab["probability"] == pytest.approx(ba["probability"], abs=1e-6)
    assert ab["percentile"] == ba["percentile"]
    assert ab["risk"]["level"] == ba["risk"]["level"]


def test_attention_varies_across_atoms(inference):
    res = inference.predict(smiles_a=ASPIRIN, smiles_b=WARFARIN, fetch_smiles=False)
    attn = np.asarray(res["attention_a"])
    assert len(attn) == 13  # aspirin's heavy atoms
    assert attn.std() > 0.01


def test_attention_is_not_just_bond_count(inference):
    # The old bug: every atom scored exactly 1 / (bonds + 1).
    from rdkit import Chem
    res = inference.predict(smiles_a=ASPIRIN, smiles_b=WARFARIN, fetch_smiles=False)
    degrees = np.array([a.GetDegree() for a in Chem.MolFromSmiles(ASPIRIN).GetAtoms()])
    assert not np.allclose(res["attention_a"], 1.0 / (degrees + 1), atol=1e-3)


def test_attention_does_not_depend_on_partner(inference):
    # Documented behaviour: each molecule is encoded on its own, so the
    # heatmap is per-molecule, not a pair explanation.
    with_warfarin = inference.predict(smiles_a=ASPIRIN, smiles_b=WARFARIN, fetch_smiles=False)
    with_caffeine = inference.predict(smiles_a=ASPIRIN, smiles_b=CAFFEINE, fetch_smiles=False)
    np.testing.assert_allclose(with_warfarin["attention_a"], with_caffeine["attention_a"], atol=1e-6)
