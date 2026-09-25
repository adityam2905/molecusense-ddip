"""
utils/inference.py  —  Inference Engine (used by CLI + Streamlit)
──────────────────────────────────────────────────────────────────
Loads a trained GNN checkpoint and its calibration file, fetches SMILES from
PubChem if needed, and returns the prediction with attention scores.

What a prediction reports:
  percentile   Where the pair's score falls among validation pairs NOT known to
               interact. This drives the risk band and doesn't depend on the
               50/50 class mix used in training.
  probability  Temperature-calibrated probability. Calibrated for a population
               where HALF of all pairs interact (the training mix), so it
               overstates risk for real-world pairs; shown for reference only.
"""

import os
import json
import requests
import numpy as np
import torch
from rdkit import Chem
from torch_geometric.data import Batch

from utils.mol_graph import smiles_to_graph
from utils.calibration import percentile, risk_from_percentile
from models.gnn_ddi import DDIPredictor


# Used only for checkpoints trained before calibration.json existed.
_LEGACY_THRESHOLDS = [(0.70, "HIGH"), (0.50, "MEDIUM"), (0.00, "LOW")]


COMMON_DRUGS = {
    "aspirin":           "CC(=O)Oc1ccccc1C(=O)O",
    "ibuprofen":         "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
    "paracetamol":       "CC(=O)Nc1ccc(O)cc1",
    "acetaminophen":     "CC(=O)Nc1ccc(O)cc1",
    "caffeine":          "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
    "metformin":         "CN(C)C(=N)N=C(N)N",
    "warfarin":          "CC(=O)C(c1ccccc1)C1=C(O)c2ccccc2OC1=O",
    "simvastatin":       "CCC(C)(C)C(=O)O[C@H]1C[C@@H](C)C=C2C=C[C@H](C)[C@H](CC[C@@H]3C[C@@H](O)CC(=O)O3)C12",
}


def pubchem_smiles(name: str) -> str | None:
    """Fetch SMILES for a drug name from PubChem, with local fallbacks."""
    name_clean = name.lower().strip()
    if name_clean in COMMON_DRUGS:
        return COMMON_DRUGS[name_clean]

    url = (f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/"
           f"{requests.utils.quote(name)}/property/CanonicalSMILES/JSON")
    try:
        r = requests.get(url, timeout=8)
        if r.status_code == 200:
            props = r.json().get("PropertyTable", {}).get("Properties", [{}])[0]
            return props.get("CanonicalSMILES") or props.get("ConnectivitySMILES") or props.get("IsomericSMILES")
    except Exception:
        pass
    return None


def _canonical(smiles: str) -> str | None:
    mol = Chem.MolFromSmiles(smiles)
    return Chem.MolToSmiles(mol) if mol is not None else None


class DDIInference:
    """
    Wraps a trained DDIPredictor for inference.

    Parameters
    ----------
    checkpoint_dir : directory containing best_model.pt, training_meta.json
                     and calibration.json
    device         : "cpu" or "cuda"
    """

    def __init__(self, checkpoint_dir: str = "checkpoints", device: str = "cpu"):
        self.device = torch.device(device)
        self.checkpoint_dir = checkpoint_dir

        meta_path = os.path.join(checkpoint_dir, "training_meta.json")
        self.meta = {}
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                self.meta = json.load(f)

        args = self.meta.get("args", {})
        n_classes = self.meta.get("n_classes", 1)
        if n_classes != 1:
            raise ValueError("DDIInference only supports binary (yes/no) models.")

        self.model = DDIPredictor(
            hidden_dim=args.get("hidden", 64),
            embed_dim=args.get("embed", 256),
            heads=args.get("heads", 4),
            dropout=0.0,
            n_classes=1,
        )
        ckpt_path = os.path.join(checkpoint_dir, "best_model.pt")
        if os.path.exists(ckpt_path):
            self.model.load_state_dict(torch.load(ckpt_path, map_location=self.device))
        else:
            print(f"[DDIInference] No checkpoint found at {ckpt_path}. "
                  "Model weights are random — run train.py first.")
        self.model.to(self.device).eval()

        self.temperature = 1.0
        self.reference_logits = None
        cal_path = os.path.join(checkpoint_dir, "calibration.json")
        if os.path.exists(cal_path):
            with open(cal_path) as f:
                cal = json.load(f)
            self.temperature = cal["temperature"]
            self.reference_logits = np.asarray(cal["reference_logits"], dtype=float)
        else:
            print(f"[DDIInference] No calibration.json in {checkpoint_dir}; "
                  "falling back to fixed probability thresholds.")

    @property
    def calibrated(self) -> bool:
        return self.reference_logits is not None

    def score_logit(self, logit: float) -> dict:
        """Probability, percentile and risk band for a raw model logit."""
        prob = float(1 / (1 + np.exp(-logit / self.temperature)))
        if self.calibrated:
            pct = percentile(logit, self.reference_logits)
            risk = risk_from_percentile(pct)
        else:
            pct = None
            level = next(lvl for t, lvl in _LEGACY_THRESHOLDS if prob >= t)
            risk = {"level": level, "percentile": None,
                    "description": f"Uncalibrated model: probability {prob:.1%}"}
        risk["probability"] = prob
        return {"probability": prob, "percentile": pct, "risk": risk}

    def predict(
        self,
        smiles_a: str = None,
        smiles_b: str = None,
        name_a: str   = None,
        name_b: str   = None,
        fetch_smiles: bool = True,
    ) -> dict:
        """
        Predict interaction between two drugs, given SMILES or drug names.

        Returns a dict with: smiles_a, smiles_b, name_a, name_b, probability,
        percentile, risk, attention_a, attention_b, top_atoms_a, top_atoms_b,
        error (None on success).
        """
        from utils.visualize import top_k_atoms

        if smiles_a is None and name_a and fetch_smiles:
            smiles_a = pubchem_smiles(name_a)
        if smiles_b is None and name_b and fetch_smiles:
            smiles_b = pubchem_smiles(name_b)

        if not smiles_a:
            return {"error": f"Could not resolve SMILES for drug A ({name_a})"}
        if not smiles_b:
            return {"error": f"Could not resolve SMILES for drug B ({name_b})"}

        g_a = smiles_to_graph(smiles_a)
        g_b = smiles_to_graph(smiles_b)
        if g_a is None:
            return {"error": f"Invalid SMILES for drug A: {smiles_a}"}
        if g_b is None:
            return {"error": f"Invalid SMILES for drug B: {smiles_b}"}
        if _canonical(smiles_a) == _canonical(smiles_b):
            return {"error": "Both inputs are the same molecule. A drug-drug "
                             "interaction needs two different drugs."}

        ba = Batch.from_data_list([g_a]).to(self.device)
        bb = Batch.from_data_list([g_b]).to(self.device)
        with torch.no_grad():
            logit_t, attn_a_t, attn_b_t = self.model(ba, bb, return_attention=True)

        logit = float(logit_t.squeeze().item())
        attn_a = attn_a_t.cpu().numpy()
        attn_b = attn_b_t.cpu().numpy()

        return {
            "smiles_a":    smiles_a,
            "smiles_b":    smiles_b,
            "name_a":      name_a or "Drug A",
            "name_b":      name_b or "Drug B",
            "logit":       logit,
            **self.score_logit(logit),
            "attention_a": attn_a,
            "attention_b": attn_b,
            "top_atoms_a": top_k_atoms(smiles_a, attn_a, k=5),
            "top_atoms_b": top_k_atoms(smiles_b, attn_b, k=5),
            "error":       None,
        }
