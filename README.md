# ⚗️ MolecuSense: Drug-Drug Interaction Predictor

**🔗 [Live demo](https://molecusense-ddip.streamlit.app/)** *(free hosting — the app may take a minute to wake up)*

Type in two drugs (by name or SMILES) and MolecuSense predicts how likely they
are to interact. It also shows which atoms in each molecule the model focused
on.

The predictions come from a **Graph Attention Network (GAT)**, a neural network
that reads each molecule as a graph of atoms and bonds. It was trained on
**TWOSIDES**, a large dataset of drug pairs reported together in FDA adverse
event reports.

> **Research use only. Not a clinical tool.**

---

## How it works

1. **Find the molecule.** A drug name is looked up on PubChem to get its
   SMILES string (a text description of the molecule). You can also paste a
   SMILES string directly.
2. **Turn it into a graph.** Each atom becomes a node (element, charge,
   aromaticity, and so on) and each bond becomes an edge (single, double,
   aromatic, in a ring).
3. **Read each molecule.** Three GAT layers pass information between
   neighbouring atoms, then the whole molecule is summarised as one vector.
4. **Combine the pair.** The two vectors are combined as
   `[A + B, |A − B|]`, so **the answer is the same whichever drug you enter
   first**.
5. **Predict.** A small neural network turns the combined vector into an
   interaction probability, shown as **LOW** (< 50%), **MEDIUM** (50–70%) or
   **HIGH** (≥ 70%) risk.

An optional **reinforcement learning (RL) layer** can then nudge the
probability up or down by at most 0.3. It currently makes no measurable
difference (see below).

---

## Results

### Data

| Step | Count |
|---|---|
| Rows in the TWOSIDES file | 42,920,391 |
| Rows with a meaningful signal (PRR ≥ 2) | 33,775,356 |
| Unique interacting drug pairs | 211,985 |
| **Pairs used for training** | **10,000** (5,000 interacting + 5,000 not) |
| Unique drugs in those pairs | 850 |
| Split | 7,000 train / 2,000 validation / 1,000 test |

TWOSIDES only lists pairs that *were* reported together. The 5,000
"non-interacting" pairs are random pairs of the same drugs that TWOSIDES never
reports together (checked against all 211,985 known pairs, not just the ones
sampled).

### GNN model

Trained for 50 epochs. Half the pairs interact, so random guessing would score
**0.50 AUROC and 50% accuracy**.

| Split | AUROC | AUPRC | F1 | Accuracy |
|---|---|---|---|---|
| Train | 0.849 | 0.847 | 0.777 | 75.0% |
| Validation | 0.773 | 0.777 | 0.731 | 69.2% |
| **Test** | **0.778** | **0.786** | **0.720** | **68.0%** |

- **AUROC:** how well the model ranks interacting pairs above non-interacting
  ones (1.0 is perfect, 0.5 is guessing).
- **AUPRC:** the same idea, focused on the interacting pairs.
- **F1:** balances how many of its "interacts" calls are right against how
  many real interactions it finds.

### RL calibration layer

The RL layer was trained on 1,500 validation pairs, picked using 500 others,
and tested once on the 1,000 test pairs it never saw.

| | Test accuracy |
|---|---|
| GNN alone | 68.0% |
| GNN + RL | 68.0% |

It learned to make almost no adjustment, so **it adds nothing right now.** It
is kept in the app to show the idea, not because it helps.

**Honest caveats:**

- **Results vary between runs.** The non-interacting pairs are a random sample.
  With the same code, two different random samples gave test AUROC 0.805 and
  0.778, so treat any single number as ±0.03. The 0.778 run above can be
  reproduced exactly.
- **Drugs aren't new at test time.** The split is by *pair*, so a drug in the
  test set usually also appears in training, paired with other drugs. These
  numbers show how well it predicts **new pairs of known drugs**, not how it
  handles a drug it has never seen.
- **"Non-interacting" isn't confirmed.** It only means TWOSIDES has no report
  of the pair, not that the combination is safe.
- **It only uses a small slice of the data:** 10,000 of the 211,985 known
  pairs.

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
2. Train the GNN (~20 min on CPU, including loading the 43M-row file):

   ```bash
   python train.py --data data/TWOSIDES.csv.gz --max_pairs 5000 --epochs 50
   ```

3. Train the RL layer (~6 min). It automatically reuses the GNN's exact dataset
   and split, and refuses to run if the data has changed:

   ```bash
   python train_rl.py --episodes 30
   ```

Both scripts write to `checkpoints/`: model weights, metrics
(`training_meta.json`, `rl_meta.json`) and training curves.

---

## App pages

- **Single Pair:** enter two drugs, see the risk level, the RL adjustment and
  an attention heatmap for each molecule.
- **Batch Predict:** upload a CSV with `drug_a` and `drug_b` columns and
  download the results.
- **System Info:** the model's metrics and training curves.

---

## Project structure

```
app/streamlit_app.py    The Streamlit app
app/style.css           The app's styling
models/gnn_ddi.py       The GAT model and the attention heatmap scores
models/rl_agent.py      The RL calibration layer and its training environment
data/data_loader.py     Loads TWOSIDES, builds positive and negative pairs
data/ddi_dataset.py     Turns pairs into graphs; dataset fingerprint
data/smiles_cache.csv   Cached PubChem lookups
utils/mol_graph.py      SMILES → graph (atom and bond features)
utils/inference.py      Loads the model and makes predictions (used by the app and CLI)
utils/visualize.py      Draws the attention heatmaps
train.py                Trains the GNN
train_rl.py             Trains the RL layer
predict.py              Command-line predictions
checkpoints/            The trained model and its metrics
requirements.txt        Python packages (CPU-only PyTorch)
packages.txt            System libraries needed on Streamlit Cloud
```

---

## More detail

<details>
<summary><b>What the attention heatmap shows</b></summary>

Each atom is coloured by how much attention its neighbours give it in the last
GAT layer. Darker atoms had more influence on the molecule's summary.

This is a hint about what the model used, not a chemical explanation, and it
only reflects the last of the three layers.

</details>

<details>
<summary><b>How the RL layer works</b></summary>

- **Input:** the GNN's view of both molecules, its probability, and summary
  statistics of the attention.
- **Output:** an adjustment between −0.3 and +0.3 added to the probability.
- **Training:** REINFORCE (a policy-gradient method). It is rewarded when the
  adjusted prediction is right and when it moves the probability towards the
  right answer, and penalised for the opposite.
- The GNN is frozen, so the RL layer can never make the GNN itself worse.

Like the GNN, it gives the same answer in either drug order.

</details>

<details>
<summary><b>Supported TWOSIDES formats</b></summary>

| Format | Drug columns | Side effect column | PRR column |
|---|---|---|---|
| Tatonetti original | `drug_1_concept_name`, `drug_2_concept_name` | `condition_concept_name` | `PRR` |
| SNAP biodata | `Drug1`, `Drug2` | `Side_Effect_Name` | `PRR_mean` |
| Simplified CSV | `drug1`, `drug2` | `side_effect` | optional |

Numeric STITCH IDs can't be looked up on PubChem, so those files aren't
supported.

</details>

<details>
<summary><b>Other training options</b></summary>

- `--source toy`: a built-in set of 12 pairs to check the pipeline works in
  seconds (the results mean nothing).
- `--multiclass`: predict the type of interaction instead of yes/no. Neither
  the app nor the RL layer supports this mode; they expect a yes/no model.
- `--source csv --data your.csv`: your own pairs, with `smiles_a`, `smiles_b`
  and `label` columns.

</details>

---

## Deployment

The live demo runs on [Streamlit Community Cloud](https://share.streamlit.io)
from the `main` branch:

- **Main file:** `app/streamlit_app.py`
- **Python:** 3.12 (3.11 or newer works)
- **Sharing:** set the app to **public** under **Settings → Sharing**, or
  visitors are sent to a Streamlit login page.

`requirements.txt` installs the CPU-only build of PyTorch; the default build
includes several GB of GPU libraries that Streamlit Cloud can't install.
`packages.txt` adds the system libraries RDKit needs to draw molecules.

**Updating the model:** retrain, commit the files in `checkpoints/`, push, then
click **Reboot app** in the Streamlit dashboard.

---

## Limitations

- **Moderate accuracy:** about 2 in 3 test pairs are classified correctly.
- **Not tested on new drugs:** see the caveats above.
- **The RL layer adds nothing** in its current form.
- **Still improving when training stopped:** the best checkpoint was the last
  of 50 epochs, so more epochs or more data would probably help.
- **Name lookups need PubChem:** unusual or ambiguous names may not be found;
  pasting SMILES always works.
- **Two separate PubChem lookups:** training and the app look up names with
  different timeouts, so a drug can be dropped from training but still work
  in the app.

---

## Bugs found and fixed

- The attention heatmap showed nothing learned: every atom scored exactly
  1 ÷ (bonds + 1) because of how the score was read out.
- Swapping the drug order changed the prediction.
- Five famous drug pairs were hard-coded to show HIGH risk, and the risk
  thresholds had been lowered to make a weak model look decisive.
- "Non-interacting" pairs were only checked against the 5,000 sampled pairs,
  so some were real interactions from the rest of TWOSIDES.
- The same non-interacting pair could be generated twice and end up in both
  training and test.
- The non-interacting pairs changed on every run (Python's set ordering), so
  results couldn't be reproduced.
- The RL layer was chosen and scored on the GNN's test pairs, so its reported
  gain was measured on the data used to pick it.
- The RL script defaulted to a different dataset size than the GNN.
- Multi-class training used the wrong loss and couldn't run.
- Training crashed at the end when AUROC couldn't be calculated.
- RL training silently produced nonsense on multi-class models.
- Loading the full TWOSIDES file ran out of memory.
- Training accuracy was never reported, only a noisy per-epoch loss.
- The "Validate Dataset" button crashed.
- The deployed app had no trained model (the checkpoints were never committed)
  and tried to install the multi-GB GPU version of PyTorch.
