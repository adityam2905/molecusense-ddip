# ⚗️ MolecuSense: Drug-Drug Interaction Screening

**🔗 [Live demo](https://molecusense-ddip.streamlit.app/)** *(free hosting — the app may take a minute to wake up)*

Type in two drugs (by name or SMILES) and MolecuSense scores how likely they
are to be reported together as an interacting pair. It also shows which atoms
in each molecule the model focused on.

The scores come from a **Graph Attention Network (GAT)** trained on
**TWOSIDES**, a large dataset of drug pairs reported together in FDA adverse
event reports.

> **The honest summary:** the model mostly scores *each drug on its own*
> (how often it shows up in FDA reports), not the chemistry of the specific
> pair. When that shortcut was removed from the data, neither this model nor
> simpler baselines could do better than chance. Details are in
> [Results](#results).
>
> **Research use only. Not a clinical tool.**

---

## How it works

1. **Find the molecule.** A drug name is looked up on PubChem to get its
   SMILES string (a text description of the molecule). You can also paste a
   SMILES string directly.
2. **Turn it into a graph.** Each atom becomes a node and each bond an edge.
3. **Read each molecule.** Three GAT layers pass information between
   neighbouring atoms, then the molecule is summarised as one vector.
4. **Combine the pair.** The two vectors are combined as `[A + B, |A − B|]`,
   so the answer is the same whichever drug you enter first. Entering the
   same molecule twice is rejected.
5. **Score it.** The app reports the pair's **percentile**: how its score
   compares with pairs *not* known to interact.
   - **HIGH:** above 95% of those pairs.
   - **MEDIUM:** above 80%.
   - **LOW:** everything else.

   So about 5% of non-interacting pairs get flagged HIGH, by design.

The app also shows a calibrated probability, but only for reference. Training
used equal numbers of interacting and non-interacting pairs, so that
probability assumes half of all drug pairs interact, which overstates
real-world risk.

---

## Results

All numbers come from the JSON files in [`results/`](results/) and can be
reproduced with the commands under [Retraining](#retraining-optional).

### Data

| Step | Count |
|---|---|
| Rows in the TWOSIDES file | 42,920,391 |
| Rows with a meaningful signal (PRR ≥ 2) | 33,775,356 |
| Unique interacting drug pairs | 211,985 |
| **Pairs used for training** | **9,982** (4,995 interacting + 4,987 not) |
| Unique drugs in those pairs | 850 |

TWOSIDES only lists pairs that *were* reported together, so "non-interacting"
pairs are built by pairing drugs that TWOSIDES never reports together. **How
they're built turns out to matter more than anything else in this project.**

### Problem 1: the model scores drugs, not pairs

In the sampled real pairs, a typical drug appears about 10 times (a third
appear 5 times or fewer), while common ones like nifedipine, omeprazole and
amlodipine appear about 40 times. The original code built non-interacting
pairs by picking drugs **at random**, so common drugs showed up far more in
real pairs than in fake ones. A model could score well
just by learning "common drug → interaction".

To test this, three baselines that **can't see the pair** were scored on the
same test pairs as the GNN:

- **Drug popularity:** no learning at all; just counts how often each drug
  appears in training interacting pairs.
- **Drug identity:** learns one number per drug.
- **Single-drug fingerprint:** uses chemistry, but scores each drug on its own
  and adds the two scores.

A fourth, **pair fingerprint**, adds simple pair information.

**Test AUROC** (0.5 = guessing, 1.0 = perfect):

| Model | Random fake pairs (original) | Frequency-matched fake pairs (deployed) | Exactly balanced fake pairs |
|---|---|---|---|
| **GNN** | **0.806** | **0.668** | **0.408** |
| Drug popularity (no learning) | 0.888 | 0.757 | 0.488 |
| Drug identity | 0.920 | 0.809 | 0.398 |
| Single-drug fingerprint | 0.907 | 0.807 | 0.401 |
| Pair fingerprint | 0.899 | 0.801 | 0.434 |

What this shows:

1. **With the original random fake pairs, just counting how often each drug
   appears beats the GNN** (0.888 vs 0.806). The task was mostly solvable
   without looking at pairs or chemistry.
2. **Frequency-matched fake pairs** pick each drug in proportion to how often
   it appears in real pairs. This lowers every score but doesn't remove the
   shortcut. Common drugs are reported with so many partners that most of
   their candidate fake pairs turn out to be real interactions and get
   rejected. Among the 50 most common drugs, 85% of appearances are still in
   real pairs (was 92%).
3. **Exactly balanced fake pairs** make every drug appear equally often in
   real and fake pairs. To do that, 1,989 real pairs of "hub" drugs had to be
   dropped. **Every model then falls to chance or below**, including the one
   that sees pair information. (Below chance is a known side effect of exact
   balancing: a drug with extra real pairs in training must have extra fake
   pairs in test.)

The same shows up in the model's own output. Scoring all 12,403 pairs among
158 drugs, **one fixed score per drug explains 89% of the deployed model's
output** (88% for the original model).

**Conclusion:** in this 10,000-pair sample, almost all the learnable signal is
"how often is this drug reported", and these models find no detectable
pair-specific chemistry. The GNN is also weaker than simple fingerprint
baselines.

### Problem 2: new drugs

The standard split puts random *pairs* in the test set, so test drugs almost
always also appear in training. **All 998 test pairs** in the original split
contain only drugs seen in training. A second split holds out **whole drugs**
instead (frequency-matched fake pairs):

| Test AUROC | New pairs of known drugs | 1 unseen drug (858 pairs) | 2 unseen drugs (33 pairs) |
|---|---|---|---|
| **GNN** | **0.668** | **0.592** | **0.496** |
| Drug popularity | 0.757 | 0.614 | 0.500 |
| Single-drug fingerprint | 0.807 | 0.685 | 0.615 |
| Pair fingerprint | 0.801 | 0.685 | 0.650 |

On drugs it has never seen, the GNN is barely above chance, and at chance
when both drugs are new. (Only 33 pairs have two new drugs, so that column is
rough.)

### Problem 3: it flagged almost everything

Two fixes:
- The risk bands now use the **percentile against non-interacting pairs**
  instead of a raw probability that assumes half of all pairs interact.
- **Temperature scaling**, fitted on the validation set, calibrates the
  probability.

| | Original model | Deployed model |
|---|---|---|
| Random pairs of 158 drugs flagged HIGH | 39% | 10% |
| Test non-interacting pairs flagged HIGH | — | 4.7% |
| Test interacting pairs flagged HIGH | — | 18.4% |
| Aspirin + ibuprofen | HIGH (90.4%) | MEDIUM (87th percentile) |
| Warfarin + table salt | HIGH (80.1%) | MEDIUM (91st percentile) |
| Metformin + caffeine | HIGH (94.0%) | MEDIUM (80th percentile) |
| Aspirin + aspirin | HIGH (95.1%) | Rejected (same molecule) |

Warfarin + table salt still ranks higher than it should. That's consistent
with Problem 1: warfarin is a commonly reported drug.

**Calibration** (deployed model, test set): expected calibration error
0.069 → **0.033**, Brier score 0.231 → **0.227**, with temperature T = 1.71.

### Deployed model

GAT, 523,521 parameters, 50 epochs, frequency-matched fake pairs, pair split.

| Split | AUROC | AUPRC | F1 | Accuracy |
|---|---|---|---|---|
| Train | 0.754 | 0.760 | 0.698 | 67.6% |
| Validation | 0.671 | 0.667 | 0.636 | 61.2% |
| **Test** | **0.668** | **0.687** | **0.651** | **62.0%** |

These replace the 0.778 AUROC reported earlier. That number came from the
random fake pairs, where drug popularity alone scored 0.888, so it measured
the shortcut rather than the pairs.

### RL experiment

A reinforcement-learning layer that nudges the probability was trained on
validation pairs and tested once on the test set: **62.0% → 62.2% accuracy**
(2 of 998 pairs), which is noise. It's **no longer used by the app**; the code
stays in [`experiments/train_rl.py`](experiments/train_rl.py) as a documented
experiment.

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

### Retraining (optional)

1. Download TWOSIDES from the
   [Tatonetti Lab](http://tatonettilab.org/resources/tatonetti-stm.html)
   (file `3003377s-s6.csv`, ~740 MB compressed) and save it as
   `data/TWOSIDES.csv.gz`.
2. Train. The first run scans the 43M-row file (~5 min) and caches the pair
   table in `data/cache/`; training then takes ~13 min on CPU.

   ```bash
   # Deployed model (frequency-matched fake pairs, pair split) -> checkpoints/
   python train.py --data data/TWOSIDES.csv.gz --max_pairs 5000 --epochs 50

   # Hold out whole drugs
   python train.py --data data/TWOSIDES.csv.gz --max_pairs 5000 --split drug --save_dir checkpoints_drug_split

   # Other fake-pair modes for comparison: --negatives uniform | balanced
   ```

3. Evaluate. Each script writes a JSON summary to `results/`.

   ```bash
   python -m experiments.evaluate   --checkpoint_dir checkpoints   # GNN vs baselines
   python -m experiments.pair_audit --checkpoint_dir checkpoints   # per-drug R², risk-band rates
   python -m experiments.train_rl   --checkpoint_dir checkpoints --episodes 30
   ```

`train.py` also fits the temperature and saves the reference scores used for
percentiles (`checkpoints/calibration.json`).

---

## App pages

- **Single Pair:** enter two drugs, see the risk band, percentile and an
  attention heatmap for each molecule.
- **Batch Predict:** upload a CSV with `drug_a` and `drug_b` columns and
  download the results.
- **System Info:** test AUROC on known and unseen drugs, calibration, and what
  the score mostly measures.

---

## Project structure

```
app/streamlit_app.py      The Streamlit app
models/gnn_ddi.py         The GAT model and the attention heatmap scores
models/rl_agent.py        RL layer (experiment only)
data/data_loader.py       Loads TWOSIDES, builds real and fake pairs, caches the pair table
data/splits.py            Pair split and held-out-drug split (shared by every script)
data/ddi_dataset.py       Turns pairs into graphs; dataset fingerprint
utils/mol_graph.py        SMILES → graph
utils/calibration.py      Temperature scaling, ECE, Brier score, percentile risk bands
utils/inference.py        Loads the model and scores pairs (used by the app and CLI)
utils/visualize.py        Draws the attention heatmaps
train.py                  Trains the GNN, then calibrates it
predict.py                Command-line predictions
experiments/evaluate.py   GNN vs. baselines on the same split
experiments/pair_audit.py How much of the output one score per drug explains
experiments/train_rl.py   RL experiment
results/                  JSON summaries behind every number in this README
checkpoints/              The deployed model, its metrics and calibration
```

Other `checkpoints_*` folders (comparison models) and `data/cache/` are
created locally and not committed.

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
<summary><b>What the attention heatmap shows</b></summary>

Each atom is coloured by how much attention its neighbours give it in the last
GAT layer. This is a hint about what the model used, not a chemical
explanation. Given the results above, it mostly reflects what the model finds
characteristic of each drug, not how two drugs interact.

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

- **Scores drugs more than pairs:** 89% of the output is explained by one
  score per drug.
- **No detectable pair-specific signal:** with exactly balanced data, every
  model is at chance.
- **Weak on unseen drugs:** 0.59 AUROC, and chance when both drugs are new.
- **Weaker than simple baselines:** fingerprint logistic regression beats the
  GNN on the original, frequency-matched and held-out-drug data (on the
  balanced data everything is at chance).
- **"Non-interacting" isn't confirmed:** it only means TWOSIDES has no report
  of the pair.
- **Small slice of the data:** 10,000 of the 211,985 known pairs.

### What would help next

- Train on far more of TWOSIDES. Pair-specific signal may only appear at
  scale.
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
