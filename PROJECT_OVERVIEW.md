# MolecuSense: Drug-Drug Interaction (DDI) Prediction System

## Abstract
MolecuSense is a research-focused drug-drug interaction (DDI) prediction system
that combines a Graph Attention Network (GAT) with an optional reinforcement
learning (RL) calibration policy. The system converts drug names or SMILES into
molecular graphs, predicts interaction likelihood, and provides atom-level
attention visualizations to improve interpretability. A Streamlit application
supports single-pair and batch screening workflows.

## 1. Problem Context and Motivation
DDIs contribute to adverse drug events and are difficult to detect using manual
rules or simple similarity heuristics. A model that learns from molecular
structure can generalize to unseen combinations and provide probabilistic
signals for screening. TWOSIDES, a large-scale polypharmacy dataset derived
from FDA adverse event reports, enables learning interaction patterns at scale.

## 2. System Overview
MolecuSense consists of:
- A GAT-based GNN for base interaction prediction.
- A lightweight RL calibration policy that optionally adjusts confidence.
- A data pipeline that resolves drug names to SMILES and builds molecular
  graphs with atom and bond features.
- A Streamlit UI for interactive inference and batch screening.

## 3. Data and Preprocessing
### 3.1 Dataset
The system is trained on TWOSIDES, which provides drug pairs and their
associated side effects from pharmacovigilance reports. Labels are derived from
statistical signals such as PRR (Proportional Reporting Ratio), and therefore
represent association rather than clinically verified causation.

### 3.2 SMILES Resolution
For drug-name inputs, the system attempts to resolve SMILES via PubChem. It
includes local fallbacks for common drugs and caches resolved SMILES in
`data/smiles_cache.csv` to reduce repeated queries.

### 3.3 Graph Construction
Each molecule is converted to a graph:
- Nodes represent atoms with feature vectors.
- Edges represent bonds with typed bond features.
This representation enables the GNN to learn from molecular topology.

## 4. Model Architecture
### 4.1 Base Model (GAT)
- Encoder: 3-layer Graph Attention Network.
- Classifier: MLP that consumes concatenated embeddings from drug A and drug B.
- Output: Base probability of interaction.

The attention mechanism highlights atoms that most influence the prediction,
which supports interpretability via heatmaps.

### 4.2 RL Calibration Policy (Optional)
- Policy network: small MLP trained with REINFORCE.
- State: GNN embeddings, base probability, and attention statistics.
- Action: adjustment delta in the range [-0.3, +0.3].
- Output: Calibrated probability = base probability + delta, clamped to [0, 1].

The calibration policy does not change GNN weights and has minimal inference
cost. Its purpose is to improve probability calibration when training shows
measurable gains in metrics (see `checkpoints/rl_meta.json`).

## 5. Inference Workflow
1. Input drug names or SMILES.
2. Resolve SMILES (PubChem lookup or fallback cache).
3. Construct molecular graphs for both drugs.
4. Run GNN to obtain base probability and attention weights.
5. Optionally run RL policy to adjust the probability.
6. Return probability, risk level, and attention visualizations.

## 6. User Interface
The Streamlit app provides:
- Single Pair: interactive prediction with attention maps and RL adjustment.
- Batch Predict: CSV upload with `drug_a`, `drug_b` columns and downloadable
  results.
- System Info: model metrics and training curves.

## 7. Benefits of RL Calibration
Base models are often miscalibrated, meaning predicted probabilities do not
match real-world correctness rates. The RL calibration policy learns when to
nudge probabilities up or down based on internal model signals, which can
improve confidence reliability for threshold-based screening or ranking.

## 8. Limitations
- TWOSIDES labels are derived statistically and are not clinically verified.
- PubChem resolution can fail for ambiguous or uncommon drug names.
- RL improvements depend on training size and data quality.
- The system is intended for research use only and is not a clinical tool.

## 9. Outputs and Artifacts
- `checkpoints/best_model.pt`: trained GNN weights.
- `checkpoints/training_meta.json`: training metrics and configuration.
- `checkpoints/training_curves.png`: GNN training curves.
- `checkpoints/rl_policy.pt`: RL policy weights (optional).
- `checkpoints/rl_meta.json`: RL training metrics (optional).

## 10. Conclusion
MolecuSense delivers a structure-aware DDI prediction pipeline with optional
probability calibration. The combination of attention-based GNNs and a
lightweight RL policy provides both predictive capability and improved
confidence calibration, making the system suitable for research-level screening
and exploratory analysis.

## Disclaimer
Research use only. Not a clinical tool.
