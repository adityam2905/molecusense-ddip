# MolecuSense: Drug-Drug Interaction (DDI) Screening

## Abstract
MolecuSense gives a model score for how likely a drug pair is to be reported
together as interacting in FDA adverse event data (TWOSIDES). It converts drug
names or SMILES into molecular graphs, scores the pair with a Graph Attention
Network (GAT), and reports the result as a percentile against non-interacting
pairs. It also shows a per-molecule attention heatmap. A Streamlit app supports
single-pair and batch scoring.

The deployed model reaches 0.806 test AUROC (73.4% accuracy). The main finding
is a limitation, though. The model mostly scores each drug on its own (how
often it appears in FDA reports), and simple baselines that never see the pair
do as well or better. Once that signal is removed from the data, no model
beats chance. See the README for the full results.

## 1. Data
- **Source:** TWOSIDES, drug pairs with side-effect signals (PRR ≥ 2). Labels
  reflect statistical association in reports, not verified causation.
- **Sample:** 39,657 pairs (19,860 interacting, 19,797 not) over 861 drugs,
  drawn from 211,985 known interacting pairs.
- **Non-interacting pairs:** drug pairs never reported together. They're drawn
  in proportion to each drug's frequency in real pairs, which only partly
  removes the popularity shortcut.
- **Cleaning:** pairs are de-duplicated by molecule. Pairs of a molecule with
  itself, and fake pairs that are really known interactions, are removed.

## 2. Model
- **Encoder:** a three-layer GAT with bond features, run on each molecule
  separately.
- **Pair representation:** `[A + B, |A − B|]`, so drug order doesn't matter.
- **Classifier:** an MLP that outputs an interaction logit.
- **Heatmap:** last-layer attention per atom. Each molecule is encoded before
  pairing, so the heatmap shows how the model reads that molecule and is the
  same whatever the partner.
- **Training:** AdamW with cosine learning-rate decay. Early stopping on
  validation AUROC (patience 8) keeps the best epoch.

## 3. Calibration and score bands
- **Temperature scaling** is fitted on the validation set (T = 1.15, test ECE
  0.029 → 0.026).
- **Score bands** use the percentile of the pair's score among validation
  pairs not known to interact: high above 95%, medium above 80%.
- **Why a percentile:** training uses a 50/50 class mix, so raw probabilities
  overstate how often pairs interact.
- **Wording is neutral** ("Model score: high"), not clinical advice.

## 4. Evaluation
- **Pair split:** random pairs. Test drugs are also seen in training.
- **Drug split:** whole drugs held out, reported separately for one and for
  two unseen drugs.
- **Baselines on the same test pairs:**
  - drug popularity
  - drug identity
  - single-drug fingerprint
  - pair fingerprint
- **Pair audit:** the share of the model's output explained by one score per
  drug.

| Test AUROC | GNN | Best baseline |
|---|---|---|
| Known drugs | 0.806 | 0.865 (drug identity) |
| 1 unseen drug | 0.679 | 0.717 (drug identity) |
| 2 unseen drugs (112 pairs) | **0.596** | 0.554 (single-drug fingerprint) |

## 5. Engineering
- **Drug lookup:** local list of 1,000+ drugs first, then PubChem, with
  spelling suggestions. Batch mode looks up each distinct name once and is
  capped at 200 rows.
- **Tests:** 36 pytest tests (order invariance, attention behaviour, split
  leakage, checkpoint loading, app rendering), run by GitHub Actions on every
  push.
- **Reproducibility:** cached pair tables, fixed seeds, and a dataset
  fingerprint saved with each model. Every reported number is in `results/`.

## 6. Experiments that didn't help
- **RL probability adjustment** (REINFORCE): +0.2% test accuracy (noise).
  Removed from the app; code kept in `experiments/train_rl.py`.
- **Exactly balanced non-interacting pairs:** removes the popularity shortcut
  entirely, but every model then falls to chance. Kept as a diagnostic.

## 7. Limitations
- Scores drugs more than pairs (91% of output explained per drug).
- Simple baselines outperform the GNN except when both drugs are unseen.
- Weak on unseen drugs (0.68 AUROC with one new drug, 0.60 with two).
- "Non-interacting" means "not reported", not "safe".
- Research use only; not a clinical tool.
