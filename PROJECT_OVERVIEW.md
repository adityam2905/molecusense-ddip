# MolecuSense: Drug-Drug Interaction (DDI) Screening

## Abstract
MolecuSense scores drug pairs for how likely they are to be reported together
as interacting in FDA adverse event data (TWOSIDES). It converts drug names or
SMILES into molecular graphs, scores the pair with a Graph Attention Network
(GAT), reports the result as a percentile against non-interacting pairs, and
shows atom-level attention maps. A Streamlit app supports single-pair and
batch screening.

The main finding is a limitation. The model mostly scores each drug on its
own (how often it appears in FDA reports), and once that signal is removed
from the data, neither the GNN nor simpler baselines beat chance. See the
README for the full results.

## 1. Data
- **Source:** TWOSIDES, drug pairs with side-effect signals (PRR ≥ 2). Labels
  reflect statistical association in reports, not verified causation.
- **Sample:** 9,982 pairs (4,995 interacting, 4,987 not) over 850 drugs.
- **Non-interacting pairs:** drug pairs never reported together. They're drawn
  in proportion to each drug's frequency in real pairs, which only partly
  removes the popularity shortcut.
- **Cleaning:** pairs are de-duplicated by molecule. Pairs of a molecule with
  itself, and fake pairs that are really known interactions, are removed.

## 2. Model
- **Encoder:** a three-layer GAT with bond features, run on each molecule.
- **Pair representation:** `[A + B, |A − B|]`, so drug order doesn't matter.
- **Classifier:** an MLP that outputs an interaction logit.

## 3. Calibration and risk bands
- **Temperature scaling** is fitted on the validation set (test ECE
  0.069 → 0.033).
- **Risk bands** use the percentile of the pair's score among validation pairs
  not known to interact: HIGH above 95%, MEDIUM above 80%.
- **Why a percentile:** training uses a 50/50 class mix, so raw probabilities
  overstate real-world risk.

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

## 5. Experiments that didn't help
- **RL probability adjustment** (REINFORCE): +0.2% test accuracy (noise).
  Removed from the app; code kept in `experiments/train_rl.py`.
- **Exactly balanced non-interacting pairs:** removes the popularity shortcut
  entirely, but every model then falls to chance. Kept as a diagnostic.

## 6. Limitations
- Scores drugs more than pairs (89% of output explained per drug).
- Weak on unseen drugs (0.59 AUROC; chance when both drugs are new).
- Simple fingerprint baselines outperform the GNN.
- "Non-interacting" means "not reported", not "safe".
- Research use only; not a clinical tool.
