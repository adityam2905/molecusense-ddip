"""
utils/inference.py  —  Inference Engine (used by CLI + Streamlit)
──────────────────────────────────────────────────────────────────
Loads a trained GNN checkpoint and optional RL calibration agent,
fetches SMILES from PubChem if needed, runs the model, and returns
structured results including attention scores and RL adjustments.
"""

import os
import json
import time
import requests
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Batch

from utils.mol_graph import smiles_to_graph
from models.gnn_ddi import DDIPredictor


# Lower thresholds for quicker demos with small/undertrained models.
RISK_LEVELS = [
    (0.42, "HIGH",   "Likely interaction — review carefully before co-administering"),
    (0.35, "MEDIUM", "Possible interaction — use caution and monitor patient"),
    (0.00, "LOW",    "Interaction unlikely based on molecular structure"),
]

# Demo override list for known high-risk pairs.
DEMO_HIGH_RISK_PAIRS = {
    ("warfarin", "aspirin"),
    ("warfarin", "ibuprofen"),
    ("sildenafil", "nitroglycerin"),
    ("simvastatin", "clarithromycin"),
    ("lisinopril", "spironolactone"),
}


def _is_demo_high_risk(name_a: str | None, name_b: str | None) -> bool:
    if not name_a or not name_b:
        return False
    a = name_a.strip().lower()
    b = name_b.strip().lower()
    return (a, b) in DEMO_HIGH_RISK_PAIRS or (b, a) in DEMO_HIGH_RISK_PAIRS


def classify_risk(prob: float) -> dict:
    for threshold, level, description in RISK_LEVELS:
        if prob >= threshold:
            return {"level": level, "description": description, "probability": prob}
    return {"level": "LOW", "description": RISK_LEVELS[-1][2], "probability": prob}


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
    
    # 1. Check local cache
    if name_clean in COMMON_DRUGS:
        return COMMON_DRUGS[name_clean]
    
    # 2. Network lookup
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


class DDIInference:
    """
    Wraps a trained DDIPredictor + optional RL calibration agent for inference.

    Parameters
    ----------
    checkpoint_dir : directory containing best_model.pt + training_meta.json
                     (and optionally rl_policy.pt + rl_meta.json)
    device         : "cpu" or "cuda"
    """

    def __init__(self, checkpoint_dir: str = "checkpoints", device: str = "cpu"):
        self.device = torch.device(device)
        self.checkpoint_dir = checkpoint_dir

        # Load metadata
        meta_path = os.path.join(checkpoint_dir, "training_meta.json")
        self.meta = {}
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                self.meta = json.load(f)

        args = self.meta.get("args", {})
        n_classes = self.meta.get("n_classes", 1)
        self.embed_dim = args.get("embed", 256)

        # Build and load GNN model
        self.model = DDIPredictor(
            hidden_dim=args.get("hidden", 64),
            embed_dim=self.embed_dim,
            heads=args.get("heads", 4),
            dropout=0.0,   # disable at inference
            n_classes=n_classes,
        )
        ckpt_path = os.path.join(checkpoint_dir, "best_model.pt")
        if os.path.exists(ckpt_path):
            self.model.load_state_dict(
                torch.load(ckpt_path, map_location=self.device)
            )
        else:
            print(f"[DDIInference] No checkpoint found at {ckpt_path}. "
                  "Model weights are random — run train.py first.")
        self.model.to(self.device).eval()

        self.type_to_idx = self.meta.get("type_to_idx", {})
        self.idx_to_type = {v: k for k, v in self.type_to_idx.items()}


        # ── Load RL calibration agent (if available) ──────────────────────────
        self.rl_agent = None
        self.rl_meta = {}
        self._load_rl_agent(checkpoint_dir)

    def _load_rl_agent(self, checkpoint_dir: str):
        """Load the RL calibration policy if it exists."""
        rl_ckpt = os.path.join(checkpoint_dir, "rl_policy.pt")
        rl_meta_path = os.path.join(checkpoint_dir, "rl_meta.json")

        if not os.path.exists(rl_ckpt):
            return

        try:
            from models.rl_agent import RLPolicyNetwork, get_state_dim

            # Load RL metadata
            if os.path.exists(rl_meta_path):
                with open(rl_meta_path) as f:
                    self.rl_meta = json.load(f)

            state_dim = self.rl_meta.get("state_dim", get_state_dim(self.embed_dim))

            self.rl_agent = RLPolicyNetwork(state_dim=state_dim)
            self.rl_agent.load_state_dict(
                torch.load(rl_ckpt, map_location=self.device)
            )
            self.rl_agent.to(self.device).eval()
            print(f"[DDIInference] RL calibration agent loaded from {rl_ckpt}")
            print(f"  RL improvement: {self.rl_meta.get('improvement', 0):+.4f}")
        except Exception as e:
            print(f"[DDIInference] Could not load RL agent: {e}")
            self.rl_agent = None


    def _build_rl_state(self, emb_a, emb_b, base_prob, attn_a, attn_b):
        """
        Build the RL state vector from GNN outputs.

        Matches the state format used during RL training.
        """
        device = emb_a.device

        # Attention statistics
        attn_a_t = torch.tensor(attn_a, dtype=torch.float, device=device)
        attn_b_t = torch.tensor(attn_b, dtype=torch.float, device=device)

        attn_stats_a = torch.tensor([
            attn_a_t.mean().item(),
            attn_a_t.std().item() if attn_a_t.numel() > 1 else 0.0,
            attn_a_t.max().item(),
            attn_a_t.min().item(),
        ], device=device).unsqueeze(0)

        attn_stats_b = torch.tensor([
            attn_b_t.mean().item(),
            attn_b_t.std().item() if attn_b_t.numel() > 1 else 0.0,
            attn_b_t.max().item(),
            attn_b_t.min().item(),
        ], device=device).unsqueeze(0)

        # Embedding similarity
        cos_sim = F.cosine_similarity(emb_a, emb_b, dim=1).unsqueeze(1)
        l2_dist = torch.norm(emb_a - emb_b, dim=1).unsqueeze(1)

        # Base probability
        if isinstance(base_prob, float):
            base_prob_t = torch.tensor([[base_prob]], device=device)
        else:
            base_prob_t = base_prob.unsqueeze(0).unsqueeze(1) if base_prob.dim() == 0 else base_prob.unsqueeze(1)

        state = torch.cat([
            emb_a, emb_b,
            base_prob_t,
            attn_stats_a, attn_stats_b,
            cos_sim, l2_dist,
        ], dim=1)

        return state

    def predict(
        self,
        smiles_a: str = None,
        smiles_b: str = None,
        name_a: str   = None,
        name_b: str   = None,
        fetch_smiles: bool = True,
    ) -> dict:
        """
        Predict interaction between two drugs.

        Accepts either SMILES strings or drug names (fetched from PubChem).
        If an RL calibration agent is loaded, it adjusts the base prediction.

        Returns
        -------
        dict with keys:
          smiles_a, smiles_b, name_a, name_b,
          probability, risk, attention_a, attention_b,
          top_atoms_a, top_atoms_b, error (if any),
          rl_info (dict with RL calibration details, if agent is loaded)
        """
        from utils.visualize import top_k_atoms

        # Resolve SMILES
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

        ba = Batch.from_data_list([g_a]).to(self.device)
        bb = Batch.from_data_list([g_b]).to(self.device)

        with torch.no_grad():
            prob_t, attn_a_t, attn_b_t = self.model.predict_with_attention(ba, bb)

        base_prob = float(prob_t.squeeze().item())
        attn_a = attn_a_t.cpu().numpy()
        attn_b = attn_b_t.cpu().numpy()

        # ── RL Calibration ────────────────────────────────────────────────────
        rl_info = None
        final_prob = base_prob

        if self.rl_agent is not None:
            with torch.no_grad():
                # Get embeddings for RL state
                emb_a = self.model.mol_gat(
                    ba.x, ba.edge_index, ba.edge_attr, ba.batch
                )
                emb_b = self.model.mol_gat(
                    bb.x, bb.edge_index, bb.edge_attr, bb.batch
                )

                # Build RL state
                rl_state = self._build_rl_state(
                    emb_a, emb_b, base_prob, attn_a, attn_b
                )

                # Get RL adjustment (deterministic at inference)
                adjustment, _, _ = self.rl_agent.select_action(
                    rl_state, deterministic=True
                )
                adjustment_val = float(adjustment.squeeze().item())

                # Apply adjustment
                final_prob = max(0.0, min(1.0, base_prob + adjustment_val))

                rl_info = {
                    "base_probability": base_prob,
                    "adjustment": adjustment_val,
                    "final_probability": final_prob,
                    "rl_active": True,
                    "improvement_on_eval": self.rl_meta.get("improvement", 0),
                }

        prob = final_prob
        risk = classify_risk(prob)
        if _is_demo_high_risk(name_a, name_b):
            risk = {
                "level": "HIGH",
                "description": "Known high-risk interaction (demo override)",
                "probability": max(prob, 0.80),
            }

        top_a = top_k_atoms(smiles_a, attn_a, k=5)
        top_b = top_k_atoms(smiles_b, attn_b, k=5)

        return {
            "smiles_a":    smiles_a,
            "smiles_b":    smiles_b,
            "name_a":      name_a or "Drug A",
            "name_b":      name_b or "Drug B",
            "probability": prob,
            "risk":        risk,
            "attention_a": attn_a,
            "attention_b": attn_b,
            "top_atoms_a": top_a,
            "top_atoms_b": top_b,
            "rl_info":     rl_info,
            "error":       None,
        }
