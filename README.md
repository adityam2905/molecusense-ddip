# ⚗️ MolecuSense: Drug-Drug Interaction Screening

[![CI](https://github.com/adityam2905/molecusense-ddip/actions/workflows/ci.yml/badge.svg)](https://github.com/adityam2905/molecusense-ddip/actions/workflows/ci.yml)

**🔗 [Live demo](https://molecusense-ddip.streamlit.app/)** *(free hosting — the app may take a minute to wake up)*

Type in two drugs (by name or SMILES) and MolecuSense gives a **model score**
for how likely the pair is to be reported together as interacting. It also
shows a heatmap of how the model reads each molecule.

The scores come from a **Graph Attention Network (GAT)** trained on
**TWOSIDES**, a large dataset of drug pairs reported together in FDA adverse
event reports.

> **The honest summary:** the model mostly scores *each drug on its own*
> (how often it shows up in FDA reports), not the chemistry of the specific
> pair. Simple baselines that never look at the pair do as well or better.
> When that shortcut was removed from the data, no model did better than
> chance. Details are in [Results](#results).
>
> **Research use only. Not a clinical tool.** The app reports a model score,
> not a risk assessment or advice.

---

## How it works

1. **Find the molecule.** A drug name is looked up in a local list of 1,000+
   drugs ([`data/smiles_cache.csv`](data/smiles_cache.csv)) first, then on
   PubChem. Misspellings get suggestions ("Asprin" → *Did you mean: Aspirin?*).
   You can also paste a SMILES string directly.
2. **Turn it into a graph.** Each atom becomes a node and each bond an edge.
3. **Read each molecule.** Three GAT layers pass information between
   neighbouring atoms, then the molecule is summarised as one vector.
4. **Combine the pair.** The two vectors are combined as `[A + B, |A − B|]`,
   so the answer is the same whichever drug you enter first. Entering the
   same molecule twice is rejected.
5. **Score it.** The app reports the pair's **percentile**: how its score
   compares with pairs *not* known to interact.
   - **Model score: high** — above 95% of those pairs.
   - **Model score: medium** — above 80%.
   - **Model score: low** — everything else.

   So about 5% of non-interacting pairs score high, by design.

The app also shows a calibrated probability, but only for reference. Training
used equal numbers of interacting and non-interacting pairs, so that
probability assumes half of all drug pairs interact.

---

## Results

All numbers come from the JSON files in [`results/`](results/) and can be
reproduced with the commands under [Retraining](#retraining-optional).

### Deployed model

GAT, 523,521 parameters, trained on 20,000 interacting pairs with
frequency-matched non-interacting pairs, random pair split. Best epoch 45 of 50
(early stopping on validation AUROC, patience 8).

| Split | AUROC | AUPRC | F1 | Accuracy |
|---|---|---|---|---|
| Train (27,761 pairs) | 0.829 | 0.838 | 0.751 | 74.7% |
| Validation (7,931) | 0.796 | 0.804 | 0.725 | 72.1% |
| **Test (3,965)** | **0.806** | **0.814** | **0.741** | **73.4%** |

**Calibration** (test set, temperature T = 1.15): expected calibration error
0.029 → **0.026**, Brier score 0.180 → **0.180**.
Test pairs scored high: 37.8% of interacting pairs, 5.1% of non-interacting.

The score is useful, but read the next sections before quoting it. Most of
it is the drug-popularity shortcut, and simple baselines beat it.

### Data

| Step | Count |
|---|---|
| Rows in the TWOSIDES file | 42,920,391 |
| Rows with a meaningful signal (PRR ≥ 2) | 33,775,356 |
| Unique interacting drug pairs | 211,985 |
| **Pairs used for training** | **39,657** (19,860 interacting + 19,797 not) |
| Unique drugs in those pairs | 861 |

TWOSIDES only lists pairs that *were* reported together, so "non-interacting"
pairs are built by pairing drugs that TWOSIDES never reports together. **How
they're built turns out to matter more than anything else in this project.**

### Problem 1: the model scores drugs, not pairs

In the 20k sample a typical drug appears in 38 interacting pairs, and the most
common in 212. The original code built non-interacting pairs by picking drugs
**at random**, so common drugs showed up far more in real pairs than in fake
ones. A model could score well just by learning "common drug → interaction".

To test this, three baselines that **can't see the pair** were scored on the
same test pairs as the GNN:

- **Drug popularity:** no learning at all; just counts how often each drug
  appears in training interacting pairs.
- **Drug identity:** learns one number per drug.
- **Single-drug fingerprint:** uses chemistry, but scores each drug on its own
  and adds the two scores.

A fourth, **pair fingerprint**, adds simple pair information.

**Test AUROC** (0.5 = guessing, 1.0 = perfect):

| Model | Random fake pairs (10k) | Frequency-matched (10k) | **Frequency-matched (40k, deployed)** | Exactly balanced (10k) |
|---|---|---|---|---|
| **GNN** | **0.806** | **0.668** | **0.806** | **0.408** |
| Drug popularity (no learning) | 0.888 | 0.757 | 0.834 | 0.488 |
| Drug identity | 0.920 | 0.809 | 0.865 | 0.398 |
| Single-drug fingerprint | 0.907 | 0.807 | 0.862 | 0.401 |
| Pair fingerprint | 0.899 | 0.801 | 0.858 | 0.434 |

(10k = 5,000 interacting pairs; 40k = 20,000.)

What this shows:

1. **With random fake pairs, just counting how often each drug appears beats
   the GNN** (0.888 vs 0.806). The task was mostly solvable without looking at
   pairs or chemistry.
2. **Frequency-matched fake pairs** pick each drug in proportion to how often
   it appears in real pairs. This lowers every score but doesn't remove the
   shortcut. Common drugs are reported with so many partners that most of
   their candidate fake pairs turn out to be real interactions and get
   rejected.
3. **More data helped the GNN more than the baselines.** Going from 10k to 40k
   pairs raised the GNN by 0.138 and the popularity baseline by 0.077,
   so the gap shrank from 0.089 to 0.028. It still doesn't beat them.
4. **Exactly balanced fake pairs** make every drug appear equally often in
   real and fake pairs (1,989 real pairs of "hub" drugs had to be dropped).
   **Every model then falls to chance or below**, including the one that sees
   pair information. (Below chance is a known side effect of exact balancing:
   a drug with extra real pairs in training must have extra fake pairs in
   test.)

The same shows up in the model's own output. Scoring all 12,403 pairs among
158 drugs, **one fixed score per drug explains 91% of the deployed model's
output** (89% for the 10k model, 88% for the original).

**Conclusion:** almost all of the learnable signal is "how often is this drug
reported", and no model here finds detectable pair-specific chemistry.

### Problem 2: new drugs

The standard split puts random *pairs* in the test set, so test drugs almost
always also appear in training. A second split holds out **whole drugs**
instead:

| Test AUROC (40k pairs) | New pairs of known drugs | 1 unseen drug (3,398 pairs) | 2 unseen drugs (112 pairs) |
|---|---|---|---|
| **GNN** | **0.806** | **0.679** | **0.596** |
| Drug popularity | 0.834 | 0.705 | 0.500 |
| Drug identity | 0.865 | 0.717 | 0.500 |
| Single-drug fingerprint | 0.862 | 0.706 | 0.554 |
| Pair fingerprint | 0.858 | 0.707 | 0.541 |

With one unseen drug the GNN is slightly behind the baselines (the known drug's
popularity still carries them). With **two unseen drugs** it is the best model
(0.596 vs ≤ 0.554), the first sign that it learned something from chemistry.
Only 112 pairs have two new drugs, so treat that column as rough. At 10k
pairs the same numbers were 0.592 and 0.496.

### Problem 3: it flagged almost everything

Two fixes:
- The score bands now use the **percentile against non-interacting pairs**
  instead of a raw probability that assumes half of all pairs interact.
- **Temperature scaling**, fitted on the validation set, calibrates the
  probability.

| | Original model | 10k model | Deployed (40k) model |
|---|---|---|---|
| Random pairs of 158 common drugs scored high | 39% | 10% | 17% |
| Test non-interacting pairs scored high | — | 4.7% | 5.1% |
| Test interacting pairs scored high | — | 18.4% | 37.8% |
| Aspirin + ibuprofen | 90.4% | 87th pct | 99.9th pct |
| Warfarin + table salt | 80.1% | 91st pct | 98.9th pct |
| Metformin + caffeine | 94.0% | 80th pct | 99.8th pct |
| Aspirin + aspirin | 95.1% | Rejected | Rejected |

On the test set the bands behave as designed: 5% of non-interacting pairs
score high, and interacting pairs score high twice as often as before. But the
last three rows show the popularity shortcut again. All six drugs are among
the most reported in TWOSIDES, so the deployed model puts them above almost
every non-interacting pair, including warfarin + table salt.

### Overfitting and early stopping

The first model trained for a fixed 50 epochs, with validation loss flat from
about epoch 25 while training loss kept falling. `train.py` now keeps the best
epoch by validation AUROC and stops after 8 epochs without an improvement of
at least 0.002. The held-out-drug model stopped at epoch 23 (best: 15). The
deployed model reached the 50-epoch cap with its best at epoch 45 and less than
0.01 AUROC gained in the last 10 epochs, as the learning rate annealed toward
zero. The train–test gap is small (0.829 vs 0.806).

### RL experiment

A reinforcement-learning layer that nudges the probability was trained on
validation pairs of the 10k model and tested once: **62.0% → 62.2% accuracy**
(2 of 998 pairs), which is noise. It's **not used by the app**; the code stays
in [`experiments/train_rl.py`](experiments/train_rl.py) as a documented
experiment ([`results/rl_checkpoints_5k.json`](results/rl_checkpoints_5k.json)).

---

## Quick start

```bash
pip install -r requirements.txt
streamlit run app/streamlit_app.py     # the trained model is included, so this just works
```

Predict from the command line:

```bash
python predict.py --name_a Warfarin --name_b Aspirin --show_atoms
python predict.py --smiles_a "CC(=O)Oc1ccccc1C(=O)O" --smiles_b "CC(C)Cc1ccc(cc1)C(C)C(=O)O"
```

### Tests

```bash
pip install -r requirements-dev.txt
pytest
```

36 tests, run on every push by GitHub Actions
([`.github/workflows/ci.yml`](.github/workflows/ci.yml)). They check:

- **Model:** the same score in either drug order; attention varies across
  atoms and isn't just bond count; a molecule's heatmap doesn't change with
  its partner.
- **Data:** train/val/test never share pairs; the drug split never shares
  drugs; fake pairs never repeat or hit a known interaction; the split is
  reproducible.
- **Inference:** the deployed checkpoint loads with its calibration and makes a
  prediction; same-molecule and invalid inputs are rejected; names resolve from
  the local list; misspellings get suggestions; wording stays neutral.
- **App:** the Streamlit pages render, with no PubChem calls.

### Retraining (optional)

1. Download TWOSIDES from the
   [Tatonetti Lab](http://tatonettilab.org/resources/tatonetti-stm.html)
   (file `3003377s-s6.csv`, ~740 MB compressed) and save it as
   `data/TWOSIDES.csv.gz`.
2. Train. The first run scans the 43M-row file (~5 min) and caches the pair
   table in `data/cache/`. Training then takes ~1 min per epoch on CPU.

   ```bash
   # Deployed model (frequency-matched fake pairs, pair split) -> checkpoints/
   python train.py --data data/TWOSIDES.csv.gz --max_pairs 20000 --epochs 50

   # Hold out whole drugs
   python train.py --data data/TWOSIDES.csv.gz --max_pairs 20000 --split drug --save_dir checkpoints_drug_split

   # Other fake-pair modes for comparison: --negatives uniform | balanced
   # Early stopping: --patience 8 --min_delta 0.002
   ```

3. Evaluate. Each script writes a JSON summary to `results/`.

   ```bash
   python -m experiments.evaluate   --checkpoint_dir checkpoints   # GNN vs baselines
   python -m experiments.pair_audit --checkpoint_dir checkpoints   # per-drug R², score-band rates
   python -m experiments.train_rl   --checkpoint_dir checkpoints --episodes 30
   ```

`train.py` also fits the temperature and saves the reference scores used for
percentiles (`checkpoints/calibration.json`).

---

## App pages

- **Single Pair:** enter two drugs, see the model score, percentile and a
  heatmap of how the model reads each molecule.
- **Batch Predict:** upload a CSV with `drug_a` and `drug_b` columns (up to
  200 rows) and download the results. Each distinct name is looked up once,
  local list first. Unknown names come back with spelling suggestions.
- **System Info:** test AUROC on known and unseen drugs, calibration, and what
  the score mostly measures.

---

## Project structure

```
app/streamlit_app.py      The Streamlit app
models/gnn_ddi.py         The GAT model and the per-atom attention scores
models/rl_agent.py        RL layer (experiment only)
data/data_loader.py       Loads TWOSIDES, builds real and fake pairs, caches the pair table
data/splits.py            Pair split and held-out-drug split (shared by every script)
data/ddi_dataset.py       Turns pairs into graphs; dataset fingerprint
data/smiles_cache.csv     Local name → SMILES list (checked before PubChem)
utils/drug_lookup.py      Name lookup: local list, PubChem, spelling suggestions
utils/mol_graph.py        SMILES → graph
utils/calibration.py      Temperature scaling, ECE, Brier score, percentile score bands
utils/inference.py        Loads the model and scores pairs (used by the app and CLI)
utils/visualize.py        Draws the attention heatmaps
train.py                  Trains the GNN with early stopping, then calibrates it
predict.py                Command-line predictions
experiments/evaluate.py   GNN vs. baselines on the same split
experiments/pair_audit.py How much of the output one score per drug explains
experiments/train_rl.py   RL experiment
tests/                    pytest suite (model, data, inference, app)
.github/workflows/ci.yml  Runs the tests on every push
results/                  JSON summaries behind every number in this README
checkpoints/              The deployed model, its metrics and calibration
```

Other `checkpoints_*` folders (the 10k and comparison models) and
`data/cache/` are created locally and not committed. Their summaries are in
`results/` with a `_5k` suffix (5,000 interacting pairs).

---

## More detail

<details>
<summary><b>How the fake (non-interacting) pairs are built</b></summary>

- **uniform** (original): pick two drugs at random. Common drugs end up
  mostly in real pairs, so popularity predicts the label.
- **degree** (deployed): pick drugs in proportion to how often they appear in
  real pairs. Helps, but not enough (see Results).
- **balanced** (diagnostic): list each drug once per real pair it's in,
  shuffle, and pair them off, so every drug appears exactly as often in fake
  pairs as in real ones. Hub drugs without enough non-interacting partners
  lose some real pairs.

All modes reject fake pairs that are known interactions (checked against all
211,985 known pairs), duplicates, and a molecule paired with itself. Pairs are
also de-duplicated by molecule, since different names can be the same
structure.

</details>

<details>
<summary><b>What the heatmap shows</b></summary>

Each atom is coloured by how much attention its neighbours give it in the last
GAT layer. Each molecule is encoded **on its own**, before the pair is
combined, so a drug's heatmap is **the same whatever it's paired with**. It
shows how the model reads that molecule, not which atoms drive an interaction.
Given the results above, it mostly reflects what the model finds
characteristic of each drug.

</details>

<details>
<summary><b>Supported TWOSIDES formats</b></summary>

| Format | Drug columns | Side effect column | PRR column |
|---|---|---|---|
| Tatonetti original | `drug_1_concept_name`, `drug_2_concept_name` | `condition_concept_name` | `PRR` |
| SNAP biodata | `Drug1`, `Drug2` | `Side_Effect_Name` | `PRR_mean` |
| Simplified CSV | `drug1`, `drug2` | `side_effect` | optional |

</details>

---

## Deployment

The live demo runs on [Streamlit Community Cloud](https://share.streamlit.io)
from the `main` branch:

- **Main file:** `app/streamlit_app.py`
- **Python:** 3.12 (3.11 or newer works)
- **Sharing:** set the app to **public** under **Settings → Sharing**.

`requirements.txt` installs CPU-only PyTorch, because the default build
includes several GB of GPU libraries. `packages.txt` adds the system libraries
RDKit needs to draw molecules.

**Updating the model:** retrain, commit the files in `checkpoints/` (including
`calibration.json`) and `results/`, push, then click **Reboot app**.

---

## Limitations

- **Scores drugs more than pairs:** 91% of the output is explained by one
  score per drug, so commonly reported drugs score high with almost anything
  (warfarin + table salt is at the 99th percentile).
- **No detectable pair-specific signal:** with exactly balanced data, every
  model is at chance.
- **Weaker than simple baselines** on known drugs and on pairs with one new
  drug. It only leads when both drugs are new (112 test pairs).
- **"Non-interacting" isn't confirmed:** it only means TWOSIDES has no report
  of the pair.
- **Still a slice of the data:** 20,000 of the 211,985 known pairs.

### What would help next

- Train on all of TWOSIDES. The GNN gained more than the baselines from
  10k → 40k pairs.
- Use a source with curated interactions *and* confirmed non-interactions
  (e.g. DrugBank) instead of relying on the absence of reports.
- Keep the balanced and held-out-drug evaluations as the bar any new model
  has to clear.

---

## Bugs found and fixed

- The model mostly learned drug popularity, because fake pairs picked drugs
  at random.
- It flagged 39% of random pairs as HIGH risk, and scored a drug paired with
  itself 95% HIGH.
- The app used clinical-sounding wording ("HIGH RISK"). It now says "Model
  score: high/medium/low".
- The heatmap was presented as showing which atoms drove the interaction, but
  it doesn't depend on the partner drug at all. It's now labelled as how the
  model reads each molecule.
- Training ran a fixed 50 epochs with validation loss flat from epoch ~25.
  Early stopping now keeps the best epoch.
- The app knew only 8 hard-coded drugs and ignored the local SMILES list.
  Batch mode made up to two PubChem calls per row with no row limit.
  Misspellings got no suggestion.
- A displayed percentile of "95" could sit in the MEDIUM band (94.97 rounded
  up).
- Building exactly balanced fake pairs slowed quadratically (a set copied on
  every step), taking hours at 20k pairs.
- The RL layer was shown in the app as "calibrating" results but changed
  nothing measurable.
- The test set only contained drugs seen in training, and nothing measured
  unseen drugs.
- Different names for the same molecule created self-pairs, duplicate pairs,
  and fake pairs that were really known interactions.
- The attention heatmap only reflected bond counts: every atom scored exactly
  1 ÷ (bonds + 1).
- Swapping the drug order changed the prediction.
- Five famous drug pairs were hard-coded to show HIGH risk.
- Fake pairs changed on every run (Python's set ordering), so results
  couldn't be reproduced.
- The RL layer was chosen and scored on the GNN's test pairs.
- Multi-class training used the wrong loss and couldn't run.
- Loading the full TWOSIDES file ran out of memory.
- The "Validate Dataset" button crashed.
- The deployed app had no trained model and tried to install the multi-GB GPU
  version of PyTorch.
- There were no tests. There are now 36, run in CI.
