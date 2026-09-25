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
  python train.py --max_pairs 20000 --epochs 50

  # Hold out whole drugs (tests drugs never seen in training)
  python train.py --max_pairs 20000 --split drug --save_dir checkpoints_drug_split

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
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    f1_score, accuracy_score,
)

sys.path.insert(0, ".")
from data.data_loader import load_dataset, dataset_stats
from data.ddi_dataset import DDIDataset, ddi_collate
from data.splits import make_split
from models.gnn_ddi import DDIPredictor
from utils.calibration import fit_temperature, calibration_report, band_rates


# ── Args ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Train DDI-GNN (Phase 2+5)")
    p.add_argument("--source",     default="twosides",
                   choices=["toy", "drugbank", "twosides", "csv"])
    p.add_argument("--data",       default=None,   help="Path to data file")
    p.add_argument("--max_pairs",  type=int,   default=20000,
                   help="Max positive drug pairs for TWOSIDES (default 20000)")
    p.add_argument("--negatives",  default="degree", choices=["balanced", "degree", "uniform"],
                   help="How non-interacting pairs are built: every drug appears as often "
                        "as in real pairs (balanced), drugs drawn in proportion to their "
                        "count (degree), or drawn equally (uniform)")
    p.add_argument("--split",      default="pair", choices=["pair", "drug"],
                   help="pair: random pairs (test drugs also seen in training); "
                        "drug: whole drugs held out of training")
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
    p.add_argument("--patience",   type=int,   default=8,
                   help="Early stopping patience (epochs)")
    p.add_argument("--min_delta",  type=float, default=0.002,
                   help="Smallest gain in validation AUROC that counts as an improvement")
    p.add_argument("--multiclass", action="store_true",
                   help="Classify interaction type (Phase 5)")
    p.add_argument("--save_dir",   default="checkpoints")
    p.add_argument("--no_weighted_sampler", action="store_true")
    return p.parse_args()


# ── Epoch helpers ──────────────────────────────────────────────────────────────

def run_epoch(model, loader, optimizer, criterion, device, n_classes=1, train=True):
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
            loss   = criterion(logits, labels.long()) if n_classes > 1 else criterion(logits, labels)

            if train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            total_loss += loss.item() * labels.size(0)

            if n_classes > 1:
                probs = torch.softmax(logits, dim=-1).detach().cpu().numpy()
                preds = probs.argmax(axis=1)
            else:
                probs = torch.sigmoid(logits).detach().cpu().numpy()
                preds = (probs >= 0.5).astype(int)

            all_probs.extend(probs.tolist())
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.cpu().numpy().tolist())

    avg_loss = total_loss / max(len(loader.dataset), 1)

    if n_classes > 1:
        # Binary AUROC/AUPRC don't apply directly to multi-class; use macro
        # one-vs-rest AUROC and leave AUPRC undefined (nan) rather than
        # silently computing a meaningless binary metric.
        try:
            auroc = roc_auc_score(all_labels, all_probs, multi_class="ovr", average="macro")
        except ValueError:
            auroc = float("nan")
        auprc = float("nan")
        f1  = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    else:
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
    df = load_dataset(source=args.source, path=args.data, max_pairs=args.max_pairs,
                      negatives=args.negatives)
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

    # Dataset splits (built from the dataset's own rows, so indices always line up)
    ds = DDIDataset(df=df)
    split = make_split(pd.DataFrame([s["meta"] for s in ds.samples]), mode=args.split,
                       val_frac=args.val_frac, test_frac=args.test_frac, seed=args.seed)
    train_ds, val_ds, test_ds = (Subset(ds, split[k]) for k in ("train", "val", "test"))
    n_train, n_val, n_test = len(train_ds), len(val_ds), len(test_ds)
    os.makedirs(args.save_dir, exist_ok=True)
    save_training_drugs([ds.samples[i]["meta"] for i in split["train"]], args.save_dir)
    new_drugs = np.bincount(split["test_new_drugs"], minlength=3)
    print(f"Split ({args.split}): {n_train} train / {n_val} val / {n_test} test"
          f"  |  test pairs with 0/1/2 unseen drugs: {new_drugs[0]}/{new_drugs[1]}/{new_drugs[2]}")

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

    # Loss: handle imbalance with pos_weight (binary) or class weights (multi-class)
    if n_classes > 1:
        train_labels = [int(ds.samples[i]["label"].item()) for i in train_ds.indices]
        counts = np.bincount(train_labels, minlength=n_classes)
        class_weights = torch.tensor(
            len(train_labels) / (n_classes * np.maximum(counts, 1)),
            dtype=torch.float, device=device,
        )
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        print(f"class_weights = {[round(w, 3) for w in class_weights.tolist()]}")
    else:
        # The weighted sampler already balances the classes; adding pos_weight
        # on top would correct the imbalance twice. Use one or the other.
        pos_w    = (torch.tensor(1.0) if sampler is not None else ds.pos_weight()).to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_w)
        print(f"pos_weight = {pos_w.item():.3f}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-5
    )

    # Training
    os.makedirs(args.save_dir, exist_ok=True)
    best_auroc   = 0.0
    best_score   = -1.0  # metric actually used to pick the checkpoint (see below)
    best_epoch   = 0
    patience_cnt = 0
    history      = {"train_loss": [], "val_loss": [], "val_auroc": [], "val_auprc": [], "val_f1": []}

    header = f"{'Ep':>4} {'TrLoss':>8} {'VlLoss':>8} {'AUROC':>7} {'AUPRC':>7} {'F1':>6} {'Acc':>6}"
    print(header)
    print("─" * len(header))

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        tr_loss, _, _, tr_f1, tr_acc = run_epoch(
            model, train_loader, optimizer, criterion, device, n_classes=n_classes, train=True
        )
        vl_loss, vl_auroc, vl_auprc, vl_f1, vl_acc = run_epoch(
            model, val_loader, optimizer, criterion, device, n_classes=n_classes, train=False
        )

        scheduler.step()

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(vl_loss)
        history["val_auroc"].append(vl_auroc)
        history["val_auprc"].append(vl_auprc)
        history["val_f1"].append(vl_f1)

        # AUROC can be undefined (NaN) for tiny or multi-class validation splits
        # that don't contain every class — fall back to F1 so a checkpoint still
        # gets saved instead of leaving best_model.pt missing at the end.
        score = vl_auroc if not np.isnan(vl_auroc) else vl_f1

        # Only a real improvement (> min_delta) counts. Otherwise tiny creeps in
        # val AUROC while the cosine schedule anneals the LR towards zero keep
        # resetting patience, so training always runs to the last epoch and the
        # "best" checkpoint is just the final one.
        flag = ""
        if score > best_score + args.min_delta:
            best_score = score
            best_epoch = epoch
            if not np.isnan(vl_auroc):
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

    # ── Final evaluation (train / val / test, all in eval mode — no dropout,
    #    no weighted resampling — so the three are directly comparable) ───────
    print(f"\nLoading best checkpoint (selection score={best_score:.4f}, AUROC={best_auroc:.4f})...")
    model.load_state_dict(torch.load(f"{args.save_dir}/best_model.pt", map_location=device))

    # Clean, unweighted, unshuffled pass over the training set — the training
    # DataLoader above uses a WeightedRandomSampler for gradient updates, which
    # is not a fair "training accuracy" measurement on its own.
    train_eval_loader = DataLoader(
        train_ds, batch_size=args.batch, shuffle=False,
        collate_fn=ddi_collate, num_workers=0,
    )

    tr_loss_f, tr_auroc_f, tr_auprc_f, tr_f1_f, tr_acc_f = run_epoch(
        model, train_eval_loader, optimizer, criterion, device, n_classes=n_classes, train=False
    )
    vl_loss_f, vl_auroc_f, vl_auprc_f, vl_f1_f, vl_acc_f = run_epoch(
        model, val_loader, optimizer, criterion, device, n_classes=n_classes, train=False
    )
    ts_loss, ts_auroc, ts_auprc, ts_f1, ts_acc = run_epoch(
        model, test_loader, optimizer, criterion, device, n_classes=n_classes, train=False
    )

    print(f"\n{'─'*62}")
    print(f"  {'Split':<6} {'Loss':>8} {'AUROC':>8} {'AUPRC':>8} {'F1':>8} {'Accuracy':>10}")
    print(f"  {'Train':<6} {tr_loss_f:>8.4f} {tr_auroc_f:>8.4f} {tr_auprc_f:>8.4f} {tr_f1_f:>8.4f} {tr_acc_f:>10.4f}")
    print(f"  {'Val':<6} {vl_loss_f:>8.4f} {vl_auroc_f:>8.4f} {vl_auprc_f:>8.4f} {vl_f1_f:>8.4f} {vl_acc_f:>10.4f}")
    print(f"  {'Test':<6} {ts_loss:>8.4f} {ts_auroc:>8.4f} {ts_auprc:>8.4f} {ts_f1:>8.4f} {ts_acc:>10.4f}")
    print(f"{'─'*62}")

    extra = {}
    if n_classes == 1:
        extra = _calibrate_and_report(model, val_loader, test_loader, split, device, args.save_dir)

    # ── Save training curves ─────────────────────────────────────────────────
    _save_training_plot(history, args.save_dir)

    # ── Save metadata ────────────────────────────────────────────────────────
    meta = {
        "best_val_auroc": best_auroc,
        "best_epoch":     best_epoch,
        "epochs_run":     len(history["train_loss"]),
        "history":        {k: [round(float(x), 5) for x in v] for k, v in history.items()},
        "train_metrics": {"loss": tr_loss_f, "auroc": tr_auroc_f, "auprc": tr_auprc_f,
                           "f1": tr_f1_f, "accuracy": tr_acc_f},
        "val_metrics":   {"loss": vl_loss_f, "auroc": vl_auroc_f, "auprc": vl_auprc_f,
                           "f1": vl_f1_f, "accuracy": vl_acc_f},
        "test_metrics":  {"loss": ts_loss, "auroc": ts_auroc, "auprc": ts_auprc,
                           "f1": ts_f1, "accuracy": ts_acc},
        "test_auroc":     ts_auroc,
        "test_auprc":     ts_auprc,
        "split_sizes":    [n_train, n_val, n_test],
        "test_new_drug_counts": new_drugs.tolist(),
        "dataset_fingerprint": ds.fingerprint(),
        **extra,
        "n_classes":      n_classes,
        "type_to_idx":    type_to_idx,
        "args":           vars(args),
    }
    with open(f"{args.save_dir}/training_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nAll outputs saved to: {args.save_dir}/")
    return model


def save_training_drugs(train_meta: list, save_dir: str):
    """
    Canonical SMILES of every drug in the training split, so the app can say
    when a drug was never seen (scores for those are much less reliable).
    """
    from data.splits import _canon
    drugs = sorted({_canon(m[k]) for m in train_meta for k in ("smiles_a", "smiles_b")})
    with open(os.path.join(save_dir, "training_drugs.json"), "w") as f:
        json.dump(drugs, f)
    return drugs


@torch.no_grad()
def collect_logits(model, loader, device):
    model.eval()
    logits, labels = [], []
    for ba, bb, y, _ in loader:
        logits.append(model(ba.to(device), bb.to(device)).cpu())
        labels.append(y)
    return torch.cat(logits).numpy(), torch.cat(labels).numpy()


def _calibrate_and_report(model, val_loader, test_loader, split, device, save_dir) -> dict:
    """
    Fit temperature scaling on validation, measure calibration on test, and
    save the scores of validation pairs NOT known to interact as the reference
    distribution the app uses for percentiles and risk bands.
    """
    val_logits, val_labels = collect_logits(model, val_loader, device)
    test_logits, test_labels = collect_logits(model, test_loader, device)

    temperature = fit_temperature(val_logits, val_labels)
    cal = calibration_report(test_logits, test_labels, temperature)
    reference = np.sort(val_logits[val_labels == 0])
    bands = band_rates(test_logits, test_labels, reference)

    print(f"\n  Temperature scaling (fit on val): T = {temperature:.3f}")
    print(f"  Test ECE   : {cal['ece_before']:.4f} -> {cal['ece_after']:.4f}")
    print(f"  Test Brier : {cal['brier_before']:.4f} -> {cal['brier_after']:.4f}")
    print("  Test pairs flagged HIGH: " + ", ".join(
        f"{k.replace('_', '-')} {v['HIGH']:.1%}" for k, v in bands.items()))

    # AUROC by how many of the pair's drugs were never seen in training.
    by_new = {}
    new = np.asarray(split["test_new_drugs"])
    for k in (0, 1, 2):
        mask = new == k
        if mask.sum() >= 20 and len(set(test_labels[mask])) == 2:
            by_new[str(k)] = {"n": int(mask.sum()),
                              "auroc": float(roc_auc_score(test_labels[mask], test_logits[mask]))}
    if by_new:
        print("  Test AUROC by unseen drugs in pair: " +
              ", ".join(f"{k} new: {v['auroc']:.4f} (n={v['n']})" for k, v in by_new.items()))

    with open(os.path.join(save_dir, "calibration.json"), "w") as f:
        json.dump({"temperature": temperature,
                   "reference_logits": [round(float(x), 5) for x in reference]}, f)

    return {"temperature": temperature, "calibration": cal,
            "test_band_rates": bands, "test_auroc_by_new_drugs": by_new}


def _save_training_plot(history: dict, save_dir: str):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(history["train_loss"], label="Train", linewidth=2)
    axes[0].plot(history["val_loss"],   label="Val",   linewidth=2)
    axes[0].set_title("Loss"); axes[0].legend(); axes[0].set_xlabel("Epoch")

    axes[1].plot(history["val_auroc"], color="tab:orange", linewidth=2)
    axes[1].set_title("Validation AUROC"); axes[1].set_xlabel("Epoch")
    axes[1].set_ylim(0, 1)

    axes[2].plot(history["val_auprc"], color="tab:green", linewidth=2, label="AUPRC")
    axes[2].plot(history["val_f1"],    color="tab:red",   linewidth=2, label="F1")
    axes[2].set_title("Val AUPRC & F1"); axes[2].legend(); axes[2].set_xlabel("Epoch")
    axes[2].set_ylim(0, 1)

    # Mark the epoch whose weights were kept (highest validation AUROC)
    best = int(np.nanargmax(history["val_auroc"]))
    for ax in axes:
        ax.axvline(best, color="grey", linestyle="--", linewidth=1)

    plt.tight_layout()
    plt.savefig(f"{save_dir}/training_curves.png", dpi=120, bbox_inches="tight")
    plt.close()
    print(f"Training curves saved to {save_dir}/training_curves.png")


if __name__ == "__main__":
    train(parse_args())
