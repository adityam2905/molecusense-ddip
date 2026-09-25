# ⚗️ MolecuSense

[![CI](https://github.com/adityam2905/molecusense-ddip/actions/workflows/ci.yml/badge.svg)](https://github.com/adityam2905/molecusense-ddip/actions/workflows/ci.yml)

**🔗 [Try the live app](https://molecusense-ddip.streamlit.app/)** *(may take a minute to wake up)*

Enter two drugs and MolecuSense estimates how likely they are to be reported
together as an interacting pair. It uses a graph neural network that reads
each drug's molecular structure. The model is trained on
[TWOSIDES](http://tatonettilab.org/resources/tatonetti-stm.html), a dataset of
drug pairs from FDA side-effect reports.

> **Research project, not medical advice.**

---

## Results

| Test set | AUROC | Accuracy |
|---|---|---|
| New pairs of drugs the model has seen | **0.806** | **73.4%** |
| Pairs with one new drug | 0.679 | — |
| Pairs with two new drugs | 0.596 | — |

*AUROC: 0.5 is guessing, 1.0 is perfect. Trained on 39,657 drug pairs
covering 861 drugs.*

**What the numbers really mean:** the model mostly learns *which drugs show
up often* in FDA reports, not how two specific drugs interact.

- A simple count of how often each drug appears scores **0.834**, better than
  the model.
- The model only beats simple methods when **both** drugs are new.

The full analysis is in [docs/RESULTS.md](docs/RESULTS.md).

---

## How it works

1. **Look up the drug.** Names are checked against a local list of 1,000+
   drugs, then PubChem. Typos get suggestions ("Asprin" → Aspirin).
2. **Build a graph.** Atoms become nodes and bonds become edges.
3. **Read each molecule.** A graph attention network (GAT) turns each
   molecule into a vector.
4. **Score the pair.** The two vectors are combined so drug order doesn't
   matter, then scored.
5. **Compare.** The result is shown as a percentile against pairs *not* known
   to interact:
   - **High:** above 95% of them.
   - **Medium:** above 80%.
   - **Low:** everything else.
6. **Warn.** The app flags drugs it never trained on, tiny or inorganic
   molecules (like table salt), and salts.

The app also shows a heatmap of which atoms the model focuses on in each
molecule.

---

## Run it

```bash
pip install -r requirements.txt
streamlit run app/streamlit_app.py          # the trained model is included
```

From the command line:

```bash
python predict.py --name_a Warfarin --name_b Aspirin
```

Run the tests (45, also run by CI on every push):

```bash
pip install -r requirements-dev.txt
pytest
```

### Retrain (optional)

1. Download TWOSIDES (`3003377s-s6.csv`, ~740 MB) from the
   [Tatonetti Lab](http://tatonettilab.org/resources/tatonetti-stm.html) and
   save it as `data/TWOSIDES.csv.gz`.
2. Train and evaluate. This takes about an hour on a CPU.

```bash
python train.py                                             # -> checkpoints/
python -m experiments.evaluate   --checkpoint_dir checkpoints   # compare with simple baselines
python -m experiments.pair_audit --checkpoint_dir checkpoints   # how much is per-drug
```

---

## Project structure

```
app/            Streamlit app
models/         The graph neural network
data/           Loading TWOSIDES, building pairs, train/test splits, drug name list
utils/          Drug lookup, SMILES → graph, calibration, prediction, heatmaps
experiments/    Comparison with simple baselines and the per-drug audit
tests/          Automated tests
checkpoints/    The trained model used by the app
results/        Saved metrics behind every number in the docs
docs/           Detailed results and analysis
train.py        Train a model
predict.py      Score a pair from the command line
```

---

## Limitations

- **Scores drugs more than pairs.** Frequently reported drugs score high with
  almost anything.
- **Weak on new drugs.** Accuracy drops for drugs outside the training data.
- **"Not interacting" isn't confirmed.** It only means no report was found.
- **Uses a slice of the data:** 20,000 of the 211,985 known interacting pairs.
- **Ignores stereochemistry and salt forms.** Mirror-image drugs look
  identical, and salts include their counterions.

---

## Deployment

The app runs on [Streamlit Community Cloud](https://share.streamlit.io) from
the `main` branch, with `app/streamlit_app.py` as the main file.
`requirements.txt` installs CPU-only PyTorch to keep the install small.

To update the model: retrain, commit `checkpoints/` and `results/`, push, then
click **Reboot app**.
