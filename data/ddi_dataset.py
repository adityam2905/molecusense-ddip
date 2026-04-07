"""
data/ddi_dataset.py  —  Phase 1 & 2: Dataset + Imbalance Handling
──────────────────────────────────────────────────────────────────
Wraps the loaded DataFrame into a PyTorch Dataset.
Includes weighted sampler to handle class imbalance automatically.
"""

import torch
import numpy as np
from torch.utils.data import Dataset, WeightedRandomSampler
from torch_geometric.data import Data, Batch
from data.data_loader import load_dataset
from utils.mol_graph import smiles_to_graph


class DDIDataset(Dataset):
    """
    Each item: (graph_A, graph_B, label, meta)
    meta = {"name_a", "name_b", "interaction_type", "smiles_a", "smiles_b"}
    """

    def __init__(self, df=None, source="toy", path=None):
        if df is None:
            df = load_dataset(source=source, path=path)

        self.samples = []
        skipped = 0
        for _, row in df.iterrows():
            g_a = smiles_to_graph(row["smiles_a"])
            g_b = smiles_to_graph(row["smiles_b"])
            if g_a is None or g_b is None:
                skipped += 1
                continue
            self.samples.append({
                "graph_a": g_a,
                "graph_b": g_b,
                "label":   torch.tensor(float(row["label"]), dtype=torch.float),
                "meta": {
                    "name_a":           str(row.get("name_a", "")),
                    "name_b":           str(row.get("name_b", "")),
                    "interaction_type": str(row.get("interaction_type", "Unknown")),
                    "smiles_a":         str(row["smiles_a"]),
                    "smiles_b":         str(row["smiles_b"]),
                }
            })

        if skipped:
            print(f"[DDIDataset] Skipped {skipped} pairs with invalid SMILES.")
        print(f"[DDIDataset] Loaded {len(self.samples)} valid pairs.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        return s["graph_a"], s["graph_b"], s["label"], s["meta"]

    def get_labels(self) -> list:
        return [int(s["label"].item()) for s in self.samples]

    def get_weighted_sampler(self) -> WeightedRandomSampler:
        """
        Returns a WeightedRandomSampler that oversamples the minority class.
        Use this as the DataLoader sampler to handle class imbalance without
        discarding data.
        """
        labels = self.get_labels()
        class_counts = np.bincount(labels)
        weights = 1.0 / class_counts[labels]
        sampler = WeightedRandomSampler(
            weights=torch.tensor(weights, dtype=torch.double),
            num_samples=len(labels),
            replacement=True,
        )
        return sampler

    def pos_weight(self) -> torch.Tensor:
        """
        Compute pos_weight for BCEWithLogitsLoss.
        pos_weight = num_negatives / num_positives
        """
        labels = self.get_labels()
        n_pos = sum(labels)
        n_neg = len(labels) - n_pos
        return torch.tensor(n_neg / max(n_pos, 1), dtype=torch.float)


# ── Custom collate ─────────────────────────────────────────────────────────────

def ddi_collate(batch):
    """
    Collate list of (graph_a, graph_b, label, meta) into batched tensors.
    Returns: (batch_a, batch_b, labels, metas)
    """
    graphs_a, graphs_b, labels, metas = zip(*batch)
    return (
        Batch.from_data_list(list(graphs_a)),
        Batch.from_data_list(list(graphs_b)),
        torch.stack(labels),
        list(metas),
    )


if __name__ == "__main__":
    from torch.utils.data import DataLoader

    ds = DDIDataset(source="toy")
    print(f"Pos weight: {ds.pos_weight():.3f}")

    loader = DataLoader(
        ds, batch_size=4,
        collate_fn=ddi_collate,
        sampler=ds.get_weighted_sampler(),
    )
    ba, bb, labels, metas = next(iter(loader))
    print(f"Batch A: {ba.num_graphs} graphs | x={ba.x.shape}")
    print(f"Labels: {labels}")
    print(f"Meta sample: {metas[0]}")
