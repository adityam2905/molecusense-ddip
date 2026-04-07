"""
utils/visualize.py  —  Phase 4: Interpretability & Atom Attention Visualization
─────────────────────────────────────────────────────────────────────────────────
Draws molecules with atoms highlighted by GAT attention scores.
High-attention atoms = the model considers these structurally important
for predicting the drug-drug interaction.

Outputs
───────
  - PIL Image  (for Streamlit / notebooks)
  - SVG string (for web embedding)
  - matplotlib figure (for reports)
"""

import io
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image

from rdkit import Chem
from rdkit.Chem import Draw, rdMolDescriptors, AllChem
from rdkit.Chem.Draw import rdMolDraw2D


def _normalize(scores: np.ndarray, eps=1e-8) -> np.ndarray:
    """Min-max normalize scores to [0, 1]."""
    mn, mx = scores.min(), scores.max()
    return (scores - mn) / (mx - mn + eps)


def attention_to_colors(scores: np.ndarray, colormap="YlOrRd"):
    """
    Map normalized attention scores to RGBA tuples using a matplotlib colormap.
    Returns list of (r, g, b, a) tuples, one per atom.
    """
    norm   = _normalize(scores)
    cmap   = cm.get_cmap(colormap)
    colors = [cmap(float(s)) for s in norm]
    return colors


def draw_molecule_attention(
    smiles: str,
    attention_scores: np.ndarray,
    title: str = "",
    size: tuple = (400, 300),
    colormap: str = "YlOrRd",
) -> Image.Image:
    """
    Draw a 2D molecule structure with atoms colored by attention score.

    Parameters
    ----------
    smiles           : SMILES string of the molecule
    attention_scores : np.ndarray of shape [num_atoms]
    title            : optional label below the image
    size             : (width, height) in pixels
    colormap         : matplotlib colormap name

    Returns
    -------
    PIL Image
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")

    AllChem.Compute2DCoords(mol)
    n_atoms = mol.GetNumAtoms()

    # Pad or trim attention scores to match actual atom count
    if len(attention_scores) < n_atoms:
        attention_scores = np.pad(
            attention_scores, (0, n_atoms - len(attention_scores))
        )
    else:
        attention_scores = attention_scores[:n_atoms]

    norm_scores = _normalize(attention_scores)
    cmap        = cm.get_cmap(colormap)

    # Build atom color dict for RDKit drawer
    atom_colors = {}
    for i, score in enumerate(norm_scores):
        r, g, b, _ = cmap(float(score))
        atom_colors[i] = (r, g, b)

    highlight_atoms = list(range(n_atoms))

    # Use RDKit's MolDraw2DSVG for clean output
    drawer = rdMolDraw2D.MolDraw2DCairo(size[0], size[1])
    drawer.drawOptions().addStereoAnnotation = False
    drawer.drawOptions().addAtomIndices      = False

    drawer.DrawMolecule(
        mol,
        highlightAtoms=highlight_atoms,
        highlightAtomColors=atom_colors,
        highlightBonds=[],
        highlightBondColors={},
        highlightAtomRadii={i: 0.4 for i in range(n_atoms)},
    )
    drawer.FinishDrawing()

    img_bytes = drawer.GetDrawingText()
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    return img


def draw_pair_attention(
    smiles_a: str,
    smiles_b: str,
    attn_a: np.ndarray,
    attn_b: np.ndarray,
    name_a: str = "Drug A",
    name_b: str = "Drug B",
    prob: float = None,
    size: tuple = (400, 300),
) -> Image.Image:
    """
    Draw both molecules side by side with attention highlighting.
    Returns a single composite PIL Image.
    """
    img_a = draw_molecule_attention(smiles_a, attn_a, title=name_a, size=size)
    img_b = draw_molecule_attention(smiles_b, attn_b, title=name_b, size=size)

    # Composite
    W  = img_a.width + img_b.width + 20
    H  = max(img_a.height, img_b.height) + 60
    canvas = Image.new("RGB", (W, H), "white")
    canvas.paste(img_a, (0, 30))
    canvas.paste(img_b, (img_a.width + 20, 30))

    # Add text labels using matplotlib
    fig, ax = plt.subplots(1, 1, figsize=(W / 100, H / 100), dpi=100)
    ax.imshow(np.array(canvas))
    ax.axis("off")
    ax.text(img_a.width / 2, 15, name_a,
            ha="center", va="top", fontsize=11, fontweight="bold")
    ax.text(img_a.width + 20 + img_b.width / 2, 15, name_b,
            ha="center", va="top", fontsize=11, fontweight="bold")

    if prob is not None:
        risk_color = "#e74c3c" if prob > 0.6 else ("#f39c12" if prob > 0.4 else "#27ae60")
        ax.text(W / 2, H - 10,
                f"Interaction probability: {prob:.1%}",
                ha="center", va="bottom", fontsize=12,
                fontweight="bold", color=risk_color)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=100)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def draw_attention_colorbar(figsize=(5, 0.5)) -> Image.Image:
    """Return a colorbar legend for the attention heatmap."""
    fig, ax = plt.subplots(figsize=figsize)
    norm = matplotlib.colors.Normalize(vmin=0, vmax=1)
    cb   = matplotlib.colorbar.ColorbarBase(
        ax, cmap=cm.get_cmap("YlOrRd"), norm=norm, orientation="horizontal"
    )
    cb.set_label("Attention score (model focus)", fontsize=9)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=100)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


# ── Top-k important atoms ──────────────────────────────────────────────────────

def top_k_atoms(smiles: str, attention_scores: np.ndarray, k: int = 3) -> list[dict]:
    """
    Return the top-k atoms by attention score with their properties.
    Useful for generating human-readable explanations.
    """
    mol  = Chem.MolFromSmiles(smiles)
    if mol is None:
        return []

    n = mol.GetNumAtoms()
    scores = attention_scores[:n]
    top_idx = np.argsort(scores)[::-1][:k]

    result = []
    for idx in top_idx:
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


# ── Quick demo ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys; sys.path.insert(0, ".")

    smiles_a = "CC(=O)Oc1ccccc1C(=O)O"   # Aspirin
    smiles_b = "CC(C)Cc1ccc(cc1)C(C)C(=O)O"  # Ibuprofen

    n_a = Chem.MolFromSmiles(smiles_a).GetNumAtoms()
    n_b = Chem.MolFromSmiles(smiles_b).GetNumAtoms()

    # Random attention for demo (replace with model output)
    attn_a = np.random.rand(n_a)
    attn_b = np.random.rand(n_b)

    img = draw_pair_attention(
        smiles_a, smiles_b, attn_a, attn_b,
        name_a="Aspirin", name_b="Ibuprofen", prob=0.73,
    )
    img.save("visualizations/demo_attention.png")
    print("Saved: visualizations/demo_attention.png")

    top = top_k_atoms(smiles_a, attn_a, k=3)
    print("\nTop-3 atoms in Aspirin by attention:")
    for a in top:
        print(f"  Atom {a['index']} ({a['symbol']}) — score: {a['attention']:.3f}, "
              f"aromatic: {a['is_aromatic']}, ring: {a['in_ring']}")
