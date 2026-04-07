"""
train.py  —  Phase 2 & 5: Full Training Pipeline
──────────────────────────────────────────────────
Features:
  - Weighted sampling + pos_weight for class imbalance (Phase 2)
  - Validation AUROC, AUPRC, F1, accuracy per epoch
  - Multi-class interaction type classification (Phase 5)
  - LR scheduler, early stopping, best-model checkpointing
  - Training curve plots saved automatically

Usage
─────
  # TWOSIDES (default — uses data/TWOSIDES.csv.gz)
  python train.py --epochs 50

  # TWOSIDES with custom sample size
  python train.py --max_pairs 5000 --epochs 100

  # Toy data (instant pipeline verification)
  python train.py --source toy --epochs 50

  # Multi-class (interaction type)
  python train.py --source toy --multiclass --epochs 100
"""

import os
import sys
import json
import argparse
import time

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    f1_score, accuracy_score, classification_report,
)
from tqdm import tqdm

sys.path.insert(0, ".")
from data.data_loader import load_dataset, dataset_stats
from data.ddi_dataset import DDIDataset, ddi_collate
from models.gnn_ddi import DDIPredictor


# ── Args ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Train DDI-GNN (Phase 2+5)")
    p.add_argument("--source",     default="twosides",
                   choices=["toy", "drugbank", "twosides", "csv"])
    p.add_argument("--data",       default=None,   help="Path to data file")
    p.add_argument("--max_pairs",  type=int,   default=10000,
                   help="Max positive drug pairs for TWOSIDES (default 10000)")
    p.add_argument("--epochs",     type=int,   default=50)
    p.add_argument("--lr",         type=float, default=1e-3)
    p.add_argument("--batch",      type=int,   default=32)
    p.add_argument("--hidden",     type=int,   default=64)
    p.add_argument("--embed",      type=int,   default=256)
    p.add_argument("--heads",      type=int,   default=4)
    p.add_argument("--dropout",    type=float, default=0.3)
    p.add_argument("--val_frac",   type=float, default=0.2)
    p.add_argument("--test_frac",  type=float, default=0.1)
    p.add_argument("--seed",       type=int,   default=42)
    p.add_argument("--patience",   type=int,   default=10,
                   help="Early stopping patience (epochs)")
    p.add_argument("--multiclass", action="store_true",
                   help="Classify interaction type (Phase 5)")
    p.add_argument("--save_dir",   default="checkpoints")
    p.add_argument("--no_weighted_sampler", action="store_true")
    return p.parse_args()


# ── Epoch helpers ──────────────────────────────────────────────────────────────

def run_epoch(model, loader, optimizer, criterion, device, train=True):
    model.train() if train else model.eval()
    total_loss  = 0.0
    all_probs, all_preds, all_labels = [], [], []

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for ba, bb, labels, _ in loader:
            ba, bb  = ba.to(device), bb.to(device)
            labels  = labels.to(device)

            if train:
                optimizer.zero_grad()

            logits = model(ba, bb)
            loss   = criterion(logits, labels)

            if train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            total_loss += loss.item() * labels.size(0)
            probs = torch.sigmoid(logits).detach().cpu().numpy()
            preds = (probs >= 0.5).astype(int)
            all_probs.extend(probs.tolist())
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.cpu().numpy().tolist())

    avg_loss = total_loss / max(len(loader.dataset), 1)

    try:
        auroc = roc_auc_score(all_labels, all_probs)
        auprc = average_precision_score(all_labels, all_probs)
    except ValueError:
        auroc = auprc = float("nan")

    f1  = f1_score(all_labels, all_preds, zero_division=0)
    acc = accuracy_score(all_labels, all_preds)

    return avg_loss, auroc, auprc, f1, acc


# ── Training loop ──────────────────────────────────────────────────────────────

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Load data
    print(f"\nLoading data (source={args.source})...")
    df = load_dataset(source=args.source, path=args.data, max_pairs=args.max_pairs)
    dataset_stats(df)

    # Multi-class: encode interaction types
    n_classes = 1
    type_to_idx = {}
    if args.multiclass:
        types = sorted(df["interaction_type"].unique())
        type_to_idx = {t: i for i, t in enumerate(types)}
        df["label"] = df["interaction_type"].map(type_to_idx)
        n_classes = len(types)
        print(f"Multi-class mode: {n_classes} interaction types")
        for t, i in type_to_idx.items():
            print(f"  {i}: {t}")

    # Dataset splits
    ds = DDIDataset(df=df)
    n  = len(ds)
    n_test  = max(1, int(n * args.test_frac))
    n_val   = max(1, int(n * args.val_frac))
    n_train = n - n_val - n_test

    train_ds, val_ds, test_ds = random_split(
        ds, [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(args.seed)
    )
    print(f"Split: {n_train} train / {n_val} val / {n_test} test")

    # DataLoaders
    sampler = None
    if not args.no_weighted_sampler:
        # Build sampler from training subset
        train_labels = [int(ds.samples[i]["label"].item()) for i in train_ds.indices]
        from torch.utils.data import WeightedRandomSampler
        import numpy as _np
        counts  = _np.bincount(train_labels)
        weights = 1.0 / counts[train_labels]
        sampler = WeightedRandomSampler(
            torch.tensor(weights, dtype=torch.double),
            num_samples=len(train_labels), replacement=True
        )

    train_loader = DataLoader(
        train_ds, batch_size=args.batch,
        sampler=sampler,
        collate_fn=ddi_collate,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch, shuffle=False,
        collate_fn=ddi_collate, num_workers=0,
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch, shuffle=False,
        collate_fn=ddi_collate, num_workers=0,
    )

    # Model
    model = DDIPredictor(
        hidden_dim=args.hidden,
        embed_dim=args.embed,
        heads=args.heads,
        dropout=args.dropout,
        n_classes=n_classes,
    ).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}\n")

    # Loss: handle imbalance with pos_weight
    pos_w    = ds.pos_weight().to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    print(f"pos_weight = {pos_w.item():.3f}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-5
    )

    # Training
    os.makedirs(args.save_dir, exist_ok=True)
    best_auroc   = 0.0
    patience_cnt = 0
    history      = {"train_loss": [], "val_loss": [], "val_auroc": [], "val_auprc": [], "val_f1": []}

    header = f"{'Ep':>4} {'TrLoss':>8} {'VlLoss':>8} {'AUROC':>7} {'AUPRC':>7} {'F1':>6} {'Acc':>6}"
    print(header)
    print("─" * len(header))

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        tr_loss, _, _, tr_f1, tr_acc = run_epoch(
            model, train_loader, optimizer, criterion, device, train=True
        )
        vl_loss, vl_auroc, vl_auprc, vl_f1, vl_acc = run_epoch(
            model, val_loader, optimizer, criterion, device, train=False
        )

        scheduler.step()

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(vl_loss)
        history["val_auroc"].append(vl_auroc)
        history["val_auprc"].append(vl_auprc)
        history["val_f1"].append(vl_f1)

        flag = ""
        if not np.isnan(vl_auroc) and vl_auroc > best_auroc:
            best_auroc = vl_auroc
            patience_cnt = 0
            torch.save(model.state_dict(), f"{args.save_dir}/best_model.pt")
            flag = " ✓"
        else:
            patience_cnt += 1

        print(f"{epoch:>4}  {tr_loss:>7.4f}  {vl_loss:>7.4f}  "
              f"{vl_auroc:>6.4f}  {vl_auprc:>6.4f}  {vl_f1:>5.4f}  "
              f"{vl_acc:>5.4f}  {time.time()-t0:.1f}s{flag}")

        if patience_cnt >= args.patience:
            print(f"\nEarly stopping at epoch {epoch} (patience={args.patience})")
            break

    # ── Test evaluation ──────────────────────────────────────────────────────
    print(f"\nLoading best checkpoint (AUROC={best_auroc:.4f})...")
    model.load_state_dict(torch.load(f"{args.save_dir}/best_model.pt", map_location=device))

    ts_loss, ts_auroc, ts_auprc, ts_f1, ts_acc = run_epoch(
        model, test_loader, optimizer, criterion, device, train=False
    )
    print(f"\n{'─'*40}")
    print(f"  Test loss  : {ts_loss:.4f}")
    print(f"  Test AUROC : {ts_auroc:.4f}")
    print(f"  Test AUPRC : {ts_auprc:.4f}")
    print(f"  Test F1    : {ts_f1:.4f}")
    print(f"  Test Acc   : {ts_acc:.4f}")
    print(f"{'─'*40}")

    # ── Save training curves ─────────────────────────────────────────────────
    _save_training_plot(history, args.save_dir)

    # ── Save metadata ────────────────────────────────────────────────────────
    meta = {
        "best_val_auroc": best_auroc,
        "test_auroc":     ts_auroc,
        "test_auprc":     ts_auprc,
        "n_classes":      n_classes,
        "type_to_idx":    type_to_idx,
        "args":           vars(args),
    }
    with open(f"{args.save_dir}/training_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nAll outputs saved to: {args.save_dir}/")
    return model


def _save_training_plot(history: dict, save_dir: str):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(history["train_loss"], label="Train", linewidth=2)
    axes[0].plot(history["val_loss"],   label="Val",   linewidth=2)
    axes[0].set_title("Loss"); axes[0].legend(); axes[0].set_xlabel("Epoch")

    axes[1].plot(history["val_auroc"], color="tab:orange", linewidth=2)
    axes[1].set_title("Validation AUROC"); axes[1].set_xlabel("Epoch")
    axes[1].set_ylim(0, 1)

    axes[2].plot(history["val_auprc"], color="tab:green", linewidth=2)
    axes[2].plot(history["val_f1"],    color="tab:red",   linewidth=2, label="F1")
    axes[2].set_title("Val AUPRC & F1"); axes[2].legend(); axes[2].set_xlabel("Epoch")
    axes[2].set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(f"{save_dir}/training_curves.png", dpi=120, bbox_inches="tight")
    plt.close()
    print(f"Training curves saved to {save_dir}/training_curves.png")


if __name__ == "__main__":
    train(parse_args())
