# MolecuSense — DDI-GNN Drug Interaction Predictor
Compact demo of a GAT-based DDI model with optional RL probability calibration.
Trained on TWOSIDES (polypharmacy) and supports PubChem SMILES lookup.

---

## Quickstart (inference)

1) Install dependencies

```bash
pip install torch torchvision
pip install torch-geometric
pip install -r requirements.txt
```

2) Run the app

```bash
streamlit run app/streamlit_app.py
```

If `checkpoints/best_model.pt` exists, the GNN loads automatically. If
`checkpoints/rl_policy.pt` exists, the RL calibration policy loads too.

---

## Train (optional)

### 1) Download TWOSIDES

| Source | URL | Notes |
|--------|-----|-------|
| Tatonetti Lab | http://tatonettilab.org/resources/tatonetti-stm.html | File: `3003377s-s6.csv` |
| SNAP Stanford | https://snap.stanford.edu/biodata/datasets/10017 | Smaller subset |

Place the file as `data/TWOSIDES.csv.gz` (or pass `--data`).

### 2) Train GNN

```bash
python train.py --data data/TWOSIDES.csv.gz --max_pairs 5000 --epochs 50
```

Outputs:
- `checkpoints/best_model.pt`
- `checkpoints/training_meta.json`
- `checkpoints/training_curves.png`

### 3) Train RL calibration (optional)

```bash
python train_rl.py --data data/TWOSIDES.csv.gz --max_pairs 5000 --episodes 30
```

Outputs:
- `checkpoints/rl_policy.pt`
- `checkpoints/rl_meta.json`
- `checkpoints/rl_training_curves.png`

The RL policy is a small MLP that adjusts the base probability by a bounded
delta in [-0.3, +0.3]. It runs fast at inference and only helps if training
improves metrics (see `rl_meta.json`).

---

## CLI prediction

```bash
python predict.py --name_a Warfarin --name_b Aspirin --show_atoms
```

If PubChem lookup fails, common drugs have local SMILES fallbacks and cached
results in `data/smiles_cache.csv`.

---

## App pages

- **Single Pair** — drug name or SMILES, attention maps, optional RL adjustment
- **Batch Predict** — upload CSV with `drug_a`, `drug_b`, download results
- **System Info** — model metrics and training curves

---

## Supported TWOSIDES formats

| Format | Drug columns | Side effect column | PRR column |
|--------|--------------|--------------------|------------|
| Tatonetti original | `drug_1_concept_name`, `drug_2_concept_name` | `condition_concept_name` | `PRR` |
| SNAP biodata | `Drug1`, `Drug2` | `Side_Effect_Name` | `PRR_mean` |
| Simplified CSV | `drug1`, `drug2` | `side_effect` | optional |

STITCH numeric IDs are not supported for PubChem lookup.

---

## Project structure

```
app/streamlit_app.py  Streamlit UI (MolecuSense)
checkpoints/          Model checkpoints + training meta
data/                 TWOSIDES loader + SMILES cache
models/               GNN + RL policy
utils/                Inference + visualization
train.py              GNN training
train_rl.py           RL calibration training
predict.py            CLI inference
```

---

## Disclaimer

Research use only. Not a clinical tool.
