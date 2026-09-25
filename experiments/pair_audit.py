"""
experiments/pair_audit.py  —  Does the model score pairs, or single drugs?
──────────────────────────────────────────────────────────────────────────
Scores every pair among N drugs from data/smiles_cache.csv and fits the
best "no pair information" model to those scores:

    score(a, b) ≈ μ + s_a + s_b

where s_a is one fixed "interaction-proneness" number per drug. R² is the share
of the model's output this explains. A high R² means the model mostly rates
each drug on its own, and the pairing barely matters.

Also reports how many of these (mostly non-interacting) random pairs land in
each risk band, and scores a few sanity-check pairs.

Usage (from the project root)
─────
  python -m experiments.pair_audit --checkpoint_dir checkpoints
Writes results/pair_audit_<checkpoint name>.json
"""

import os
import sys
import json
import argparse

import numpy as np
import pandas as pd
import torch
from rdkit import Chem, RDLogger
from torch_geometric.data import Batch

sys.path.insert(0, ".")
from utils.inference import DDIInference
from utils.mol_graph import smiles_to_graph

RDLogger.DisableLog("rdApp.*")

SANITY_PAIRS = [
    ("Aspirin + Ibuprofen", "CC(=O)Oc1ccccc1C(=O)O", "CC(C)Cc1ccc(cc1)C(C)C(=O)O"),
    ("Warfarin + table salt", "CC(=O)C(c1ccccc1)C1=C(O)c2ccccc2OC1=O", "[Na+].[Cl-]"),
    ("Metformin + caffeine", "CN(C)C(=N)N=C(N)N", "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"),
    ("Aspirin + aspirin", "CC(=O)Oc1ccccc1C(=O)O", "CC(=O)Oc1ccccc1C(=O)O"),
]


def additive_fit(m: np.ndarray):
    """
    Exact least-squares fit of m[i, j] ≈ μ + s_i + s_j over all i ≠ j of a
    symmetric matrix (closed form; the diagonal is ignored).
    Returns (s, r2).
    """
    n = m.shape[0]
    off = ~np.eye(n, dtype=bool)
    mu = m[off].mean()
    row_mean = (m.sum(axis=1) - np.diag(m)) / (n - 1)
    s = (n - 1) * (row_mean - mu) / (n - 2)
    pred = mu + s[:, None] + s[None, :]
    iu = np.triu_indices(n, k=1)
    resid = m[iu] - pred[iu]
    r2 = 1 - (resid ** 2).sum() / ((m[iu] - m[iu].mean()) ** 2).sum()
    return s, float(r2)


def load_drugs(n_drugs: int, seed: int) -> pd.DataFrame:
    cache = pd.read_csv("data/smiles_cache.csv", keep_default_na=False)
    rows, seen = [], set()
    for name, smi in sorted(zip(cache["name"], cache["smiles"])):
        mol = Chem.MolFromSmiles(smi) if smi else None
        if mol is None:
            continue
        canon = Chem.MolToSmiles(mol)
        if canon not in seen:  # synonyms for the same molecule count once
            seen.add(canon)
            rows.append((name, smi))
    drugs = pd.DataFrame(rows, columns=["name", "smiles"])
    if 0 < n_drugs < len(drugs):
        drugs = drugs.sample(n=n_drugs, random_state=seed).sort_values("name")
    return drugs.reset_index(drop=True)


@torch.no_grad()
def pair_logits(inf: DDIInference, smiles: list) -> np.ndarray:
    model = inf.model
    graphs = [smiles_to_graph(s) for s in smiles]
    batch = Batch.from_data_list(graphs).to(inf.device)
    emb = model.mol_gat(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
    n = len(smiles)
    m = np.zeros((n, n))
    for i in range(n):
        pair = torch.cat([emb[i] + emb, torch.abs(emb[i] - emb)], dim=1)
        m[i] = model.classifier(pair).squeeze(-1).cpu().numpy()
    return m


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--checkpoint_dir", default="checkpoints")
    p.add_argument("--n_drugs", type=int, default=158, help="0 = every cached drug")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    tag = os.path.basename(os.path.normpath(args.checkpoint_dir))
    inf = DDIInference(checkpoint_dir=args.checkpoint_dir)
    drugs = load_drugs(args.n_drugs, args.seed)
    n = len(drugs)

    logits = pair_logits(inf, drugs["smiles"].tolist())
    probs = 1 / (1 + np.exp(-logits / inf.temperature))
    s_logit, r2_logit = additive_fit(logits)
    _, r2_prob = additive_fit(probs)

    iu = np.triu_indices(n, k=1)
    levels = [inf.score_logit(x)["risk"]["level"] for x in logits[iu]]
    band_share = {lvl: levels.count(lvl) / len(levels) for lvl in ("HIGH", "MEDIUM", "LOW")}

    order = np.argsort(-s_logit)
    top = [{"name": drugs["name"][i], "score": round(float(s_logit[i]), 3)} for i in order[:10]]

    sanity = []
    for label, a, b in SANITY_PAIRS:
        res = inf.predict(smiles_a=a, smiles_b=b, fetch_smiles=False)
        sanity.append({"pair": label, "error": res["error"]} if res["error"] else
                      {"pair": label, "level": res["risk"]["level"],
                       "percentile": res["percentile"], "probability": res["probability"]})

    out = {
        "checkpoint": tag, "calibrated": inf.calibrated, "n_drugs": n, "n_pairs": len(levels),
        "r2_per_drug_logit": r2_logit, "r2_per_drug_probability": r2_prob,
        "band_share": band_share, "top_drugs_by_proneness": top, "sanity_pairs": sanity,
    }
    os.makedirs("results", exist_ok=True)
    path = os.path.join("results", f"pair_audit_{tag}.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)

    print(f"{tag}: {n} drugs, {len(levels):,} pairs")
    print(f"  Output explained by one score per drug: R² = {r2_prob:.3f} (probability), "
          f"{r2_logit:.3f} (logit)")
    print(f"  Risk bands: HIGH {band_share['HIGH']:.1%}  MEDIUM {band_share['MEDIUM']:.1%}  "
          f"LOW {band_share['LOW']:.1%}")
    print("  Most 'interaction-prone' drugs: " + ", ".join(d["name"] for d in top[:6]))
    for s in sanity:
        print(f"  {s['pair']}: " + (s["error"] if "error" in s else
              f"{s['level']} (prob {s['probability']:.1%}"
              + (f", percentile {s['percentile']:.0f})" if s["percentile"] is not None else ")")))
    print(f"  Saved {path}")


if __name__ == "__main__":
    main()
