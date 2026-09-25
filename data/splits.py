"""
data/splits.py  —  Train / validation / test splits
────────────────────────────────────────────────────
Shared by train.py and experiments/evaluate.py so
every script evaluates on exactly the same pairs.

Two modes:
  pair  Random split of pairs. A test drug usually also appears in training
        (paired with other drugs), so this measures "new pairs of known drugs".
  drug  Whole drugs are held out. Every test pair contains at least one drug
        never seen in training (and no validation drug); every validation pair
        contains at least one validation drug. This measures how the model
        handles drugs it has never seen.

Drugs are identified by canonical SMILES, so two names for the same molecule
count as one drug.
"""

import numpy as np
import torch
from rdkit import Chem


def _canon(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    return Chem.MolToSmiles(mol) if mol is not None else smiles


def make_split(df, mode: str = "pair", val_frac: float = 0.2,
               test_frac: float = 0.1, seed: int = 42) -> dict:
    """
    Returns {"train": [...], "val": [...], "test": [...]} of row positions in
    `df`, plus "test_new_drugs": for each test row, how many of its two drugs
    are absent from training (0, 1 or 2).
    """
    n = len(df)
    if mode == "pair":
        n_test = max(1, int(n * test_frac))
        n_val = max(1, int(n * val_frac))
        # Same permutation as torch.utils.data.random_split used previously.
        perm = torch.randperm(n, generator=torch.Generator().manual_seed(seed)).tolist()
        n_train = n - n_val - n_test
        split = {"train": perm[:n_train],
                 "val": perm[n_train:n_train + n_val],
                 "test": perm[n_train + n_val:]}
    elif mode == "drug":
        split = _drug_split(df, val_frac, test_frac, seed)
    else:
        raise ValueError(f"Unknown split mode {mode!r}; use 'pair' or 'drug'")

    a = [_canon(s) for s in df["smiles_a"]]
    b = [_canon(s) for s in df["smiles_b"]]
    train_drugs = {a[i] for i in split["train"]} | {b[i] for i in split["train"]}
    split["test_new_drugs"] = [int(a[i] not in train_drugs) + int(b[i] not in train_drugs)
                               for i in split["test"]]
    return split


def _drug_split(df, val_frac, test_frac, seed):
    a = [_canon(s) for s in df["smiles_a"]]
    b = [_canon(s) for s in df["smiles_b"]]
    n = len(df)
    drugs = sorted(set(a) | set(b))
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(drugs))

    pairs_of = {d: set() for d in drugs}
    for i, (x, y) in enumerate(zip(a, b)):
        pairs_of[x].add(i)
        pairs_of[y].add(i)

    # Hold out drugs one at a time (random order) until the pairs they touch
    # reach the target share. Popular drugs touch many pairs, so group sizes
    # are approximate.
    test_drugs, test_rows = set(), set()
    val_drugs, val_rows = set(), set()
    for k in order:
        d = drugs[k]
        if len(test_rows) < test_frac * n:
            test_drugs.add(d)
            test_rows |= pairs_of[d]
        elif len(val_rows - test_rows) < val_frac * n:
            val_drugs.add(d)
            val_rows |= pairs_of[d]
        else:
            break

    test, val, train = [], [], []
    for i in range(n):
        in_test = a[i] in test_drugs or b[i] in test_drugs
        in_val = a[i] in val_drugs or b[i] in val_drugs
        if in_test and in_val:
            continue  # touches both held-out groups: dropped to keep them separate
        (test if in_test else val if in_val else train).append(i)
    return {"train": train, "val": val, "test": test}
