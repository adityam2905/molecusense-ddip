"""
utils/inference.py  —  Inference Engine (used by CLI + Streamlit)
──────────────────────────────────────────────────────────────────
Loads a trained GNN checkpoint and its calibration file, resolves drug names to
SMILES (local cache first, then PubChem; see utils/drug_lookup.py), and
returns the model's score with per-molecule attention scores.

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
import numpy as np
import torch
from rdkit import Chem
from torch_geometric.data import Batch

from utils.mol_graph import smiles_to_graph, structure_warnings
from utils.calibration import percentile, risk_from_percentile
from utils.drug_lookup import resolve
from models.gnn_ddi import DDIPredictor


# Used only for checkpoints trained before calibration.json existed.
_LEGACY_THRESHOLDS = [(0.70, "HIGH"), (0.50, "MEDIUM"), (0.00, "LOW")]


def _not_found(label: str, name: str, suggestions: list, reason: str = None) -> str:
    if reason:
        return f"Drug {label} ({name!r}): {reason}"
    msg = f"Could not find drug {label} ({name!r}) in the local list or on PubChem."
    if suggestions:
        msg += " Did you mean: " + ", ".join(suggestions) + "?"
    return msg


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

        # Canonical SMILES of every drug in the training split. Scores for other
        # drugs are much less reliable (test AUROC drops sharply), so predictions
        # say when a drug is new. None = unknown (older checkpoint).
        self.training_drugs = None
        drugs_path = os.path.join(checkpoint_dir, "training_drugs.json")
        if os.path.exists(drugs_path):
            with open(drugs_path) as f:
                self.training_drugs = set(json.load(f))

    def seen_in_training(self, smiles: str) -> bool | None:
        if self.training_drugs is None:
            return None
        return _canonical(smiles) in self.training_drugs

    def reliability_notes(self, name: str, smiles: str) -> list[str]:
        """Plain-language reasons this drug's part of the score is less trustworthy."""
        notes = [f"{name} {w}." for w in structure_warnings(smiles)]
        if self.seen_in_training(smiles) is False:
            notes.append(f"{name} was not in the training data. Scores for new drugs are much "
                         "less reliable (see System Info).")
        return notes

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
        seen_a / seen_b (drug in the training split; None if unknown), notes
        (reasons the score is less reliable), error (None on success).
        """
        from utils.visualize import top_k_atoms

        for label, name, smiles in (("A", name_a, smiles_a), ("B", name_b, smiles_b)):
            if smiles is None and name and fetch_smiles:
                found = resolve(name)
                if not found["smiles"]:
                    return {"error": _not_found(label, name, found["suggestions"],
                                                found.get("reason"))}
                if label == "A":
                    smiles_a = found["smiles"]
                else:
                    smiles_b = found["smiles"]

        if not smiles_a:
            return {"error": "No SMILES or name given for drug A."}
        if not smiles_b:
            return {"error": "No SMILES or name given for drug B."}

        g_a = smiles_to_graph(smiles_a)
        g_b = smiles_to_graph(smiles_b)
        if g_a is None:
            return {"error": f"Invalid SMILES for drug A: {smiles_a}"}
        if g_b is None:
            return {"error": f"Invalid SMILES for drug B: {smiles_b}"}
        if _canonical(smiles_a) == _canonical(smiles_b):
            msg = ("Both inputs are the same molecule. A drug-drug interaction needs two "
                   "different drugs.")
            if name_a and name_b and name_a.strip().lower() != name_b.strip().lower():
                # Distinct names, same structure: usually stereoisomers, since
                # PubChem's SMILES and the model's features ignore 3D arrangement.
                msg += (f" {name_a} and {name_b} have the same structure once stereochemistry "
                        "(3D arrangement) is ignored, which this model does, so it can't tell "
                        "them apart.")
            return {"error": msg}

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
            "seen_a":      self.seen_in_training(smiles_a),
            "seen_b":      self.seen_in_training(smiles_b),
            "notes":       (self.reliability_notes(name_a or "Drug A", smiles_a)
                            + self.reliability_notes(name_b or "Drug B", smiles_b)),
            "error":       None,
        }
