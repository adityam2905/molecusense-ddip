"""Splits don't leak, and fake (non-interacting) pairs are built correctly."""

from collections import Counter

import numpy as np
import pandas as pd
import pytest

from data.splits import make_split
from data.data_loader import (
    _dedupe_by_structure, _generate_balanced_negatives, _generate_negative_name_pairs,
)

# Simple distinct molecules: CC, CCC, ... (valid SMILES, unique structures)
MOLS = ["C" * k for k in range(2, 42)]


def _pairs_df(n_pairs=400, seed=0):
    rng = np.random.default_rng(seed)
    seen, rows = set(), []
    while len(rows) < n_pairs:
        a, b = sorted(rng.choice(len(MOLS), 2, replace=False))
        if (a, b) not in seen:
            seen.add((a, b))
            rows.append({"smiles_a": MOLS[a], "smiles_b": MOLS[b], "label": int(rng.random() < 0.5)})
    return pd.DataFrame(rows)


def _drugs(df, idx):
    return set(df["smiles_a"].iloc[idx]) | set(df["smiles_b"].iloc[idx])


@pytest.mark.parametrize("mode", ["pair", "drug"])
def test_split_indices_are_disjoint(mode):
    df = _pairs_df()
    s = make_split(df, mode=mode, seed=1)
    train, val, test = set(s["train"]), set(s["val"]), set(s["test"])
    assert not (train & val) and not (train & test) and not (val & test)
    assert len(test) > 0 and len(val) > 0 and len(train) > 0


def test_pair_split_covers_every_row():
    df = _pairs_df()
    s = make_split(df, mode="pair", seed=1)
    assert sorted(s["train"] + s["val"] + s["test"]) == list(range(len(df)))


def test_drug_split_holds_out_whole_drugs():
    df = _pairs_df()
    s = make_split(df, mode="drug", seed=1)
    train_drugs = _drugs(df, s["train"])
    # every test pair has at least one drug never seen in training
    for i in s["test"]:
        assert df["smiles_a"][i] not in train_drugs or df["smiles_b"][i] not in train_drugs
    assert all(k >= 1 for k in s["test_new_drugs"])


def test_pair_split_is_reproducible():
    df = _pairs_df()
    assert make_split(df, "pair", seed=3) == make_split(df, "pair", seed=3)


def test_structure_cleanup_removes_leaks():
    df = pd.DataFrame([
        {"name_a": "Ethanol", "name_b": "Propane", "smiles_a": "CCO", "smiles_b": "CCC", "label": 1},
        # same molecules as row 0 under different spellings / order -> duplicate
        {"name_a": "Propane", "name_b": "Alcohol", "smiles_a": "CCC", "smiles_b": "OCC", "label": 1},
        # fake pair that is structurally the known interaction -> removed
        {"name_a": "Alcohol", "name_b": "Propane", "smiles_a": "OCC", "smiles_b": "CCC", "label": 0},
        # a molecule with itself -> removed
        {"name_a": "Ethanol", "name_b": "Alcohol", "smiles_a": "CCO", "smiles_b": "OCC", "label": 0},
        {"name_a": "Ethanol", "name_b": "Butane", "smiles_a": "CCO", "smiles_b": "CCCC", "label": 0},
    ])
    out = _dedupe_by_structure(df)
    assert len(out) == 2
    assert list(out["label"]) == [1, 0]


def test_negative_pairs_avoid_known_pairs_and_repeats():
    names = [f"D{i}" for i in range(15)]
    known = {("D0", "D1"), ("D2", "D3")}
    neg = _generate_negative_name_pairs(names, known, n_neg=60, seed=0)
    keys = [tuple(sorted(p)) for p in neg[["name_a", "name_b"]].values]
    assert len(keys) == len(set(keys))
    assert not (set(keys) & known)
    assert all(a != b for a, b in keys)


def test_balanced_negatives_match_each_drugs_count():
    rng = np.random.default_rng(0)
    names = [f"D{i:02d}" for i in range(60)]
    pos = set()
    while len(pos) < 150:
        a, b = rng.choice(names, 2, replace=False)
        pos.add(tuple(sorted((a, b))))
    pos_df = pd.DataFrame(sorted(pos), columns=["name_a", "name_b"])
    neg, pos_kept = _generate_balanced_negatives(pos_df, exclude_pairs=pos, seed=0)
    cp = Counter(pos_kept["name_a"].tolist() + pos_kept["name_b"].tolist())
    cn = Counter(neg["name_a"].tolist() + neg["name_b"].tolist())
    assert cp == cn
    keys = {tuple(sorted(p)) for p in neg[["name_a", "name_b"]].values}
    assert not (keys & pos)
