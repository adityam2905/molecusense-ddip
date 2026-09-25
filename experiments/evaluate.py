"""
experiments/evaluate.py  —  GNN vs. simple baselines on the same split
──────────────────────────────────────────────────────────────────────
Rebuilds the exact dataset and split a checkpoint was trained on (checked by
fingerprint) and scores simple baselines on the same test pairs:

  drug_popularity    score = how many training interacting pairs each drug is in
                     (no learning; the "drugs common in FDA reports" shortcut)
  drug_identity      logistic regression on which two drugs they are
                     (one learned number per drug; no pair info, no chemistry)
  fingerprint_single logistic regression on fp(A) + fp(B): chemistry, but each
                     drug scored on its own (additive, no pair info)
  fingerprint_pair   logistic regression on [fp(A) + fp(B), fp(A) AND fp(B)]:
                     chemistry plus simple pair information

If the single-drug baselines come close to the GNN, the GNN isn't using much
beyond per-drug signals.

Usage (from the project root)
─────
  python -m experiments.evaluate --checkpoint_dir checkpoints
Writes results/eval_<checkpoint name>.json
"""

import os
import sys
import json
import hashlib
import argparse
from collections import Counter

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import rdFingerprintGenerator
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

sys.path.insert(0, ".")
from data.data_loader import load_dataset
from data.splits import make_split

RDLogger.DisableLog("rdApp.*")
C_GRID = (0.01, 0.1, 1.0, 10.0)


def fingerprint_of(df) -> str:
    # Same format as DDIDataset.fingerprint(), without building graphs.
    h = hashlib.sha256()
    for a, b, y in zip(df["smiles_a"], df["smiles_b"], df["label"]):
        h.update(f"{a}|{b}|{int(y)}\n".encode())
    return h.hexdigest()


def auroc_report(y, score, test_idx, new_drugs) -> dict:
    y_t, s_t = y[test_idx], score[test_idx]
    out = {"test_auroc": float(roc_auc_score(y_t, s_t)), "by_new_drugs": {}}
    new = np.asarray(new_drugs)
    for k in (0, 1, 2):
        mask = new == k
        if mask.sum() >= 20 and len(set(y_t[mask])) == 2:
            out["by_new_drugs"][str(k)] = {"n": int(mask.sum()),
                                           "auroc": float(roc_auc_score(y_t[mask], s_t[mask]))}
    return out


def fit_logreg(X, y, split) -> tuple:
    """Pick C on the validation set, return (test-ready scores for all rows, C)."""
    best = None
    for c in C_GRID:
        clf = LogisticRegression(C=c, max_iter=3000)
        clf.fit(X[split["train"]], y[split["train"]])
        val = roc_auc_score(y[split["val"]], clf.decision_function(X[split["val"]]))
        if best is None or val > best[0]:
            best = (val, c, clf)
    return best[2].decision_function(X), best[1]


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--checkpoint_dir", default="checkpoints")
    args = p.parse_args()

    tag = os.path.basename(os.path.normpath(args.checkpoint_dir))
    with open(os.path.join(args.checkpoint_dir, "training_meta.json")) as f:
        meta = json.load(f)
    a = meta["args"]

    df = load_dataset(source=a["source"], path=a["data"], max_pairs=a["max_pairs"],
                      negatives=a.get("negatives", "degree"))
    if fingerprint_of(df) != meta.get("dataset_fingerprint"):
        raise RuntimeError(f"Dataset doesn't match the one {tag} was trained on; "
                           "retrain it with the current train.py first.")
    split = make_split(df, mode=a.get("split", "pair"), val_frac=a["val_frac"],
                       test_frac=a["test_frac"], seed=a["seed"])
    y = df["label"].to_numpy().astype(int)

    canon = {s: Chem.MolToSmiles(Chem.MolFromSmiles(s))
             for s in set(df["smiles_a"]) | set(df["smiles_b"])}
    da = df["smiles_a"].map(canon).to_numpy()
    db = df["smiles_b"].map(canon).to_numpy()
    tr = np.asarray(split["train"])

    results = {}

    # 1. Drug popularity (no learning)
    pos_count = Counter(list(da[tr][y[tr] == 1]) + list(db[tr][y[tr] == 1]))
    pop = np.array([pos_count[x] + pos_count[z] for x, z in zip(da, db)], dtype=float)
    results["drug_popularity"] = auroc_report(y, pop, split["test"], split["test_new_drugs"])

    # 2. Drug identity (multi-hot over training drugs; unseen drugs -> all zeros)
    vocab = {d: i for i, d in enumerate(sorted(set(da[tr]) | set(db[tr])))}
    X_id = np.zeros((len(df), len(vocab)), dtype=np.float32)
    for i, (x, z) in enumerate(zip(da, db)):
        for d in (x, z):
            if d in vocab:
                X_id[i, vocab[d]] += 1
    score, c = fit_logreg(X_id, y, split)
    results["drug_identity"] = {**auroc_report(y, score, split["test"], split["test_new_drugs"]), "C": c}

    # 3-4. Morgan fingerprints (radius 2, 2048 bits)
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    fps = {s: gen.GetFingerprintAsNumPy(Chem.MolFromSmiles(s)).astype(np.float32)
           for s in canon.values()}
    fa = np.stack([fps[x] for x in da])
    fb = np.stack([fps[z] for z in db])

    score, c = fit_logreg(fa + fb, y, split)
    results["fingerprint_single"] = {**auroc_report(y, score, split["test"], split["test_new_drugs"]), "C": c}
    score, c = fit_logreg(np.hstack([fa + fb, fa * fb]), y, split)
    results["fingerprint_pair"] = {**auroc_report(y, score, split["test"], split["test_new_drugs"]), "C": c}

    gnn = {"test_auroc": meta["test_metrics"]["auroc"],
           "by_new_drugs": meta.get("test_auroc_by_new_drugs", {})}
    out = {
        "checkpoint": tag, "negatives": a.get("negatives"), "split": a.get("split", "pair"),
        "split_sizes": meta.get("split_sizes"), "test_new_drug_counts": meta.get("test_new_drug_counts"),
        "gnn": gnn, "gnn_test_metrics": meta["test_metrics"],
        "calibration": meta.get("calibration"), "temperature": meta.get("temperature"),
        "test_band_rates": meta.get("test_band_rates"), "baselines": results,
    }
    os.makedirs("results", exist_ok=True)
    path = os.path.join("results", f"eval_{tag}.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)

    print(f"{tag}  (negatives={out['negatives']}, split={out['split']})")
    print(f"  {'model':<20} {'test AUROC':>10}   by unseen drugs in pair")
    for name, r in [("GNN", gnn)] + list(results.items()):
        extra = "  ".join(f"{k} new: {v['auroc']:.3f}" for k, v in r["by_new_drugs"].items())
        print(f"  {name:<20} {r['test_auroc']:>10.3f}   {extra}")
    print(f"  Saved {path}")


if __name__ == "__main__":
    main()
