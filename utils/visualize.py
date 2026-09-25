"""
utils/visualize.py  —  Per-molecule attention heatmaps
───────────────────────────────────────────────────────
Draws a molecule with atoms coloured by the GAT's attention scores.

Each molecule is encoded on its own, so a drug's scores are the same whatever
it is paired with. The heatmap shows what the model picks out in that
molecule, not why a particular pair got its score.
"""

import io

import matplotlib
import numpy as np
from PIL import Image
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Draw import rdMolDraw2D


def _normalize(scores: np.ndarray, eps=1e-8) -> np.ndarray:
    """Min-max normalize scores to [0, 1]."""
    mn, mx = scores.min(), scores.max()
    return (scores - mn) / (mx - mn + eps)


def draw_molecule_attention(
    smiles: str,
    attention_scores: np.ndarray,
    size: tuple = (400, 300),
    colormap: str = "YlOrRd",
) -> Image.Image:
    """Draw a 2D structure with each atom coloured by its attention score."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")

    AllChem.Compute2DCoords(mol)
    n_atoms = mol.GetNumAtoms()

    scores = np.asarray(attention_scores, dtype=float)
    if len(scores) < n_atoms:
        scores = np.pad(scores, (0, n_atoms - len(scores)))
    scores = scores[:n_atoms]

    cmap = matplotlib.colormaps[colormap]
    atom_colors = {i: tuple(cmap(float(s))[:3]) for i, s in enumerate(_normalize(scores))}

    drawer = rdMolDraw2D.MolDraw2DCairo(size[0], size[1])
    drawer.drawOptions().addStereoAnnotation = False
    drawer.drawOptions().addAtomIndices = False
    drawer.DrawMolecule(
        mol,
        highlightAtoms=list(range(n_atoms)),
        highlightAtomColors=atom_colors,
        highlightBonds=[],
        highlightBondColors={},
        highlightAtomRadii={i: 0.4 for i in range(n_atoms)},
    )
    drawer.FinishDrawing()
    return Image.open(io.BytesIO(drawer.GetDrawingText())).convert("RGB")


def top_k_atoms(smiles: str, attention_scores: np.ndarray, k: int = 3) -> list[dict]:
    """The k highest-attention atoms with their basic properties."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return []

    scores = np.asarray(attention_scores)[:mol.GetNumAtoms()]
    result = []
    for idx in np.argsort(scores)[::-1][:k]:
        atom = mol.GetAtomWithIdx(int(idx))
        result.append({
            "index":       int(idx),
            "symbol":      atom.GetSymbol(),
            "attention":   float(scores[idx]),
            "is_aromatic": bool(atom.GetIsAromatic()),
            "in_ring":     bool(atom.IsInRing()),
            "degree":      int(atom.GetDegree()),
        })
    return result
