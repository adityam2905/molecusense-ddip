"""
models/gnn_ddi.py  —  GAT drug-pair model
──────────────────────────────────────────
  Drug graph ──► GATConv×3 (bond features, residuals) ──► mean+max pool
             ──► projection ──► molecule embedding (256)
  [A + B, |A − B|] ──► MLP(512→512→128→1) ──► interaction logit

Each molecule is encoded separately, so its per-atom attention scores (used
for the heatmap) don't depend on the partner drug. The pair combination is
symmetric, so drug order doesn't change the prediction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool, global_max_pool


class MolGAT(nn.Module):
    """
    Graph Attention Network encoder for a single molecule.

    Parameters
    ----------
    node_feat_dim : int   Node input features (24)
    edge_feat_dim : int   Edge input features (6)
    hidden_dim    : int   Hidden size per attention head
    embed_dim     : int   Final molecule embedding size
    heads         : int   Number of attention heads per layer
    dropout       : float Dropout probability
    """

    def __init__(
        self,
        node_feat_dim: int = 24,
        edge_feat_dim: int = 6,
        hidden_dim: int    = 64,
        embed_dim: int     = 256,
        heads: int         = 4,
        dropout: float     = 0.2,
    ):
        super().__init__()
        self.dropout = dropout

        # Input projection: lift node features to hidden_dim
        self.input_proj = nn.Linear(node_feat_dim, hidden_dim * heads)

        # GAT Layer 1: hidden_dim*heads → hidden_dim, multi-head
        self.gat1 = GATConv(
            hidden_dim * heads, hidden_dim,
            heads=heads, dropout=dropout,
            edge_dim=edge_feat_dim,
            concat=True,   # output: hidden_dim * heads
        )
        self.bn1 = nn.BatchNorm1d(hidden_dim * heads)

        # GAT Layer 2
        self.gat2 = GATConv(
            hidden_dim * heads, hidden_dim,
            heads=heads, dropout=dropout,
            edge_dim=edge_feat_dim,
            concat=True,
        )
        self.bn2 = nn.BatchNorm1d(hidden_dim * heads)

        # GAT Layer 3: collapse to single head for clean pooling
        self.gat3 = GATConv(
            hidden_dim * heads, hidden_dim,
            heads=1, dropout=dropout,
            edge_dim=edge_feat_dim,
            concat=False,  # output: hidden_dim
        )
        self.bn3 = nn.BatchNorm1d(hidden_dim)

        # Pool mean + max, then project to embed_dim
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )


    def forward(self, x, edge_index, edge_attr, batch, return_attention=False):
        """
        x            : [N_atoms, node_feat_dim]
        edge_index   : [2, E]
        edge_attr    : [E, edge_feat_dim]
        batch        : [N_atoms]
        return_attention : if True, also returns per-atom attention scores

        Returns
        -------
        emb          : [num_graphs, embed_dim]
        (optionally) atom_scores : [N_atoms] — mean attention received per atom
        """
        # Input projection
        x = F.elu(self.input_proj(x))
        x = F.dropout(x, p=self.dropout, training=self.training)

        # Layer 1
        h1 = self.gat1(x, edge_index, edge_attr=edge_attr)
        h1 = self.bn1(F.elu(h1))
        h1 = h1 + F.pad(x, (0, h1.shape[1] - x.shape[1]))  # residual (pad if needed)

        # Layer 2
        h2 = self.gat2(h1, edge_index, edge_attr=edge_attr)
        h2 = self.bn2(F.elu(h2))
        h2 = h2 + h1  # residual

        # Layer 3
        h3, (ei3, aw3) = self.gat3(h2, edge_index, edge_attr=edge_attr,
                                    return_attention_weights=True)
        h3 = self.bn3(F.elu(h3))

        # Dual pooling: mean + max
        pool_mean = global_mean_pool(h3, batch)  # [G, hidden_dim]
        pool_max  = global_max_pool(h3,  batch)  # [G, hidden_dim]
        pooled    = torch.cat([pool_mean, pool_max], dim=1)  # [G, hidden_dim*2]

        emb = self.proj(pooled)  # [G, embed_dim]

        # Build per-atom attention scores from last layer.
        if return_attention:
            # aw3: [E, 1] attention weights. GAT's softmax normalizes attention
            # to sum to 1 over each TARGET node's incoming edges — so averaging
            # by target (ei3[1]) is mathematically forced to equal 1/degree for
            # every atom, regardless of what the network learned, and carries
            # no information. Instead, average by SOURCE (ei3[0]): how much
            # attention each atom receives from the different neighbors it
            # contributes to. That total isn't constrained to any fixed value,
            # so it actually reflects the learned attention pattern.
            n_atoms = x.shape[0]
            sources = ei3[0]   # atoms contributing to their neighbors' updates
            atom_scores = torch.zeros(n_atoms, device=x.device)
            atom_scores.scatter_add_(0, sources, aw3.squeeze(-1))
            counts = torch.zeros(n_atoms, device=x.device)
            counts.scatter_add_(0, sources, torch.ones_like(aw3.squeeze(-1)))
            atom_scores = atom_scores / (counts + 1e-8)
            return emb, atom_scores

        return emb


class DDIPredictor(nn.Module):
    """
    Full DDI prediction model (Phase 3).

    Shared GAT encoder for both drugs + MLP classifier.
    """

    def __init__(
        self,
        node_feat_dim: int = 24,
        edge_feat_dim: int = 6,
        hidden_dim: int    = 64,
        embed_dim: int     = 256,
        heads: int         = 4,
        dropout: float     = 0.3,
        n_classes: int     = 1,   # 1 = binary; >1 = multi-class interaction types
    ):
        super().__init__()
        self.n_classes = n_classes
        self.mol_gat = MolGAT(
            node_feat_dim=node_feat_dim,
            edge_feat_dim=edge_feat_dim,
            hidden_dim=hidden_dim,
            embed_dim=embed_dim,
            heads=heads,
            dropout=dropout,
        )

        # Classifier: symmetric combination of (emb_A, emb_B) → prediction.
        # Plain concat([emb_A, emb_B]) is NOT invariant to swapping A and B —
        # [emb_a, emb_b] != [emb_b, emb_a] as vectors, so nothing forces the
        # MLP to treat "A interacts with B" the same as "B interacts with A",
        # even though drug-drug interaction is inherently a symmetric relation.
        # (emb_a + emb_b) and |emb_a - emb_b| are both exactly swap-invariant
        # by construction, so the order the two drugs are passed in can no
        # longer change the prediction.
        clf_input = embed_dim * 2
        self.classifier = nn.Sequential(
            nn.Linear(clf_input, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, n_classes),
        )

    def forward(self, batch_a, batch_b, return_attention=False):
        """
        Returns logits [B, n_classes] or [B] for binary.
        If return_attention=True, also returns (attn_a, attn_b) per-atom scores.
        """
        if return_attention:
            emb_a, attn_a = self.mol_gat(
                batch_a.x, batch_a.edge_index, batch_a.edge_attr, batch_a.batch,
                return_attention=True
            )
            emb_b, attn_b = self.mol_gat(
                batch_b.x, batch_b.edge_index, batch_b.edge_attr, batch_b.batch,
                return_attention=True
            )
        else:
            emb_a = self.mol_gat(batch_a.x, batch_a.edge_index, batch_a.edge_attr, batch_a.batch)
            emb_b = self.mol_gat(batch_b.x, batch_b.edge_index, batch_b.edge_attr, batch_b.batch)

        pair   = torch.cat([emb_a + emb_b, torch.abs(emb_a - emb_b)], dim=1)
        logits = self.classifier(pair)

        if self.n_classes == 1:
            logits = logits.squeeze(-1)

        if return_attention:
            return logits, attn_a, attn_b
        return logits

    @torch.no_grad()
    def predict_proba(self, batch_a, batch_b):
        self.eval()
        logits = self.forward(batch_a, batch_b)
        if self.n_classes == 1:
            return torch.sigmoid(logits)
        return torch.softmax(logits, dim=-1)

    @torch.no_grad()
    def predict_with_attention(self, batch_a, batch_b):
        """Returns (probability, atom_attention_a, atom_attention_b)."""
        self.eval()
        logits, attn_a, attn_b = self.forward(batch_a, batch_b, return_attention=True)
        prob = torch.sigmoid(logits) if self.n_classes == 1 else torch.softmax(logits, dim=-1)
        return prob, attn_a, attn_b


# ── Smoke test ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys; sys.path.insert(0, ".")
    from data.ddi_dataset import DDIDataset, ddi_collate
    from torch.utils.data import DataLoader

    ds     = DDIDataset(source="toy")
    loader = DataLoader(ds, batch_size=4, collate_fn=ddi_collate)
    ba, bb, labels, metas = next(iter(loader))

    model  = DDIPredictor()
    logits = model(ba, bb)
    print(f"Logits shape : {logits.shape}")
    print(f"Probs        : {torch.sigmoid(logits).tolist()}")
    print(f"Labels       : {labels.tolist()}")
    print(f"Parameters   : {sum(p.numel() for p in model.parameters()):,}")

    # Attention
    prob, attn_a, attn_b = model.predict_with_attention(ba, bb)
    print(f"Attention A shape: {attn_a.shape}")
    print(f"Attention A (first mol): {attn_a[:5]}")
