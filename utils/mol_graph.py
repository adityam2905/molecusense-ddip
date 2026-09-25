"""
utils/mol_graph.py  —  SMILES → PyG Data
─────────────────────────────────────────
Converts a SMILES string into a PyTorch Geometric Data object.
  Nodes = atoms (24-dim feature vector)
  Edges = bonds (6-dim feature vector, bidirectional)
"""

import torch
from torch_geometric.data import Data
from rdkit import Chem


ATOM_TYPES      = ['C', 'N', 'O', 'S', 'F', 'P', 'Cl', 'Br', 'I', 'Other']
HYBRIDIZATION   = [
    Chem.rdchem.HybridizationType.SP,
    Chem.rdchem.HybridizationType.SP2,
    Chem.rdchem.HybridizationType.SP3,
    Chem.rdchem.HybridizationType.OTHER,
]
BOND_TYPES = [
    Chem.rdchem.BondType.SINGLE,
    Chem.rdchem.BondType.DOUBLE,
    Chem.rdchem.BondType.TRIPLE,
    Chem.rdchem.BondType.AROMATIC,
]

NODE_FEAT_DIM = 24
EDGE_FEAT_DIM = 6


def one_hot(value, categories):
    vec = [0] * len(categories)
    idx = categories.index(value) if value in categories else len(categories) - 1
    vec[idx] = 1
    return vec


def atom_features(atom) -> list:
    symbol = atom.GetSymbol() if atom.GetSymbol() in ATOM_TYPES else 'Other'
    degree = min(atom.GetDegree(), 5)
    hyb    = atom.GetHybridization()
    return (
        one_hot(symbol, ATOM_TYPES)               # 10
        + one_hot(degree, [0, 1, 2, 3, 4, 5])    # 6
        + [atom.GetFormalCharge()]                # 1
        + [atom.GetTotalNumHs()]                  # 1
        + one_hot(hyb, HYBRIDIZATION)             # 4
        + [int(atom.GetIsAromatic())]             # 1
        + [int(atom.IsInRing())]                  # 1
    )  # total: 24


def bond_features(bond) -> list:
    bt = bond.GetBondType()
    return (
        one_hot(bt, BOND_TYPES)          # 4
        + [int(bond.GetIsConjugated())]  # 1
        + [int(bond.IsInRing())]         # 1
    )  # total: 6


MIN_HEAVY_ATOMS = 4


def structure_warnings(smiles: str) -> list[str]:
    """
    Reasons the model's reading of this structure is not meaningful or is
    muddied: bare ions and elements, tiny molecules, and salts or mixtures
    whose extra parts (counterions, water) get averaged in with the drug.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return []
    notes = []
    frags = Chem.GetMolFrags(mol, asMols=True)
    largest = max(f.GetNumHeavyAtoms() for f in frags)
    if not any(a.GetSymbol() == "C" for a in mol.GetAtoms()):
        notes.append("is inorganic (no carbon atoms), so there is little structure for the model to read")
    elif largest < MIN_HEAVY_ATOMS:
        notes.append(f"is a very small molecule ({largest} heavy atoms), so there is little "
                     "structure for the model to read")
    if len(frags) > 1:
        notes.append(f"is written as {len(frags)} separate parts (e.g. a salt with its counterions); "
                     "the model averages over all of them")
    return notes


def smiles_to_graph(smiles: str) -> Data | None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    x = torch.tensor([atom_features(a) for a in mol.GetAtoms()], dtype=torch.float)

    edge_index, edge_attr = [], []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bf = bond_features(bond)
        edge_index += [[i, j], [j, i]]
        edge_attr  += [bf, bf]

    if not edge_index:
        edge_index = [[0, 0]]
        edge_attr  = [[0] * EDGE_FEAT_DIM]

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr  = torch.tensor(edge_attr,  dtype=torch.float)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
