"""
experiments/train_rl.py  —  EXPERIMENT: RL probability adjustment
──────────────────────────────────────────────────────────────────
Trains a REINFORCE policy that nudges the frozen GNN's probability by up to
±0.3. In every run so far it has made no measurable difference to test
accuracy, so it is NOT used by the app; calibration there is done with
temperature scaling (utils/calibration.py). Kept as a documented experiment.

The policy is trained and selected only on the GNN's validation pairs and
scored once on the GNN's test pairs (same split as train.py, via data/splits.py).

Usage (from the project root)
─────
  python -m experiments.train_rl --episodes 30
  python -m experiments.train_rl --checkpoint_dir checkpoints_drug_split

Outputs: experiments/output/rl/<checkpoint name>/ (policy + curves, not
committed) and results/rl_<checkpoint name>.json (summary, committed).
"""

import os
import sys
import json
import argparse
import time

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import Subset, random_split

sys.path.insert(0, ".")
from data.data_loader import load_dataset, dataset_stats
from data.ddi_dataset import DDIDataset
from data.splits import make_split
from models.gnn_ddi import DDIPredictor
from models.rl_agent import (
    RLPolicyNetwork, DDIEnvironment, RLTrainer, get_state_dim,
)


def parse_args():
    p = argparse.ArgumentParser(description="Train RL calibration agent (Phase 7)")
    p.add_argument("--checkpoint_dir", default="checkpoints",
                   help="Directory with trained GNN checkpoint")
    p.add_argument("--source",     default=None,
                   choices=["toy", "drugbank", "twosides", "csv"],
                   help="Defaults to the source the GNN was trained on")
    p.add_argument("--data",       default=None,
                   help="Defaults to the data file the GNN was trained on")
    p.add_argument("--max_pairs",  type=int, default=None,
                   help="Defaults to the max_pairs the GNN was trained with")
    p.add_argument("--select_frac", type=float, default=0.25,
                   help="Fraction of the GNN's validation pairs held out for "
                        "RL checkpoint selection (the rest train the RL policy)")
    p.add_argument("--episodes",   type=int, default=500,
                   help="Number of RL training episodes")
    p.add_argument("--lr",         type=float, default=3e-4,
                   help="RL learning rate")
    p.add_argument("--batch",      type=int, default=32,
                   help="Batch size for RL updates")
    p.add_argument("--eval_every", type=int, default=25,
                   help="Evaluate every N episodes")
    p.add_argument("--patience",   type=int, default=50,
                   help="Early stopping patience (episodes)")
    p.add_argument("--seed",       type=int, default=42)
    p.add_argument("--entropy_coeff", type=float, default=0.01,
                   help="Entropy coefficient for exploration")
    return p.parse_args()


def load_base_model(checkpoint_dir: str, device: torch.device) -> DDIPredictor:
    """Load the trained GNN model from checkpoint."""
    meta_path = os.path.join(checkpoint_dir, "training_meta.json")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(
            f"No training_meta.json found in {checkpoint_dir}. "
            "Run train.py first to train the base GNN model."
        )

    with open(meta_path) as f:
        meta = json.load(f)

    args = meta.get("args", {})
    n_classes = meta.get("n_classes", 1)

    if n_classes != 1:
        raise ValueError(
            f"Base checkpoint in {checkpoint_dir} was trained with "
            f"n_classes={n_classes} (multi-class interaction typing). "
            "RL calibration only supports binary base models: its reward, "
            "state, and action (a scalar delta added to a single base "
            "probability) all assume a sigmoid output in [0, 1]. Applying it "
            "to multi-class softmax logits would silently produce nonsense "
            "adjustments rather than a real calibration. Train the base GNN "
            "without --multiclass if you want RL calibration."
        )

    model = DDIPredictor(
        hidden_dim=args.get("hidden", 64),
        embed_dim=args.get("embed", 256),
        heads=args.get("heads", 4),
        dropout=0.0,  # no dropout at inference
        n_classes=n_classes,
    )

    ckpt_path = os.path.join(checkpoint_dir, "best_model.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"No best_model.pt found in {checkpoint_dir}. "
            "Run train.py first."
        )

    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.to(device).eval()

    # Freeze GNN parameters
    for p in model.parameters():
        p.requires_grad = False

    print(f"Loaded base GNN model from {ckpt_path}")
    print(f"  embed_dim={args.get('embed', 256)}, "
          f"hidden_dim={args.get('hidden', 64)}, "
          f"n_classes={n_classes}")
    return model, meta


def train_rl(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # ── Load base GNN model ──────────────────────────────────────────────────
    base_model, meta = load_base_model(args.checkpoint_dir, device)
    tag = os.path.basename(os.path.normpath(args.checkpoint_dir))
    out_dir = os.path.join("experiments", "output", "rl", tag)
    os.makedirs(out_dir, exist_ok=True)
    embed_dim = meta.get("args", {}).get("embed", 256)

    # ── Load data (must be the exact dataset the GNN was trained on) ─────────
    gnn_args = meta.get("args", {})
    for name in ("source", "data", "max_pairs"):
        given, trained = getattr(args, name), gnn_args.get(name)
        if given is None:
            setattr(args, name, trained)
        elif trained is not None and given != trained:
            raise ValueError(
                f"--{name}={given!r} differs from the GNN's training value "
                f"({trained!r}). RL must use the GNN's exact dataset so the "
                "GNN's train/val/test split can be reproduced; otherwise RL "
                "could train on pairs the GNN was tested on."
            )

    print(f"\nLoading data (source={args.source})...")
    df = load_dataset(source=args.source, path=args.data, max_pairs=args.max_pairs,
                      negatives=gnn_args.get("negatives", "degree"))
    dataset_stats(df)

    ds = DDIDataset(df=df)

    # The split below is only leak-free if this is byte-for-byte the dataset
    # the GNN was trained on — same sizes aren't enough (the pairs themselves
    # can differ, e.g. if the SMILES cache changed).
    expected_fp = meta.get("dataset_fingerprint")
    if expected_fp is None:
        raise RuntimeError(
            "GNN checkpoint has no dataset_fingerprint, so there's no way to "
            "confirm RL is splitting the same dataset. Retrain the GNN with "
            "the current train.py first."
        )
    if ds.fingerprint() != expected_fp:
        raise RuntimeError(
            "Rebuilt dataset does not match the one the GNN was trained on "
            "(fingerprint mismatch) — RL could end up training on the GNN's "
            "test pairs. Retrain the GNN before training RL."
        )

    # Reproduce train.py's split exactly (shared data/splits.py, same settings).
    split = make_split(pd.DataFrame([s["meta"] for s in ds.samples]),
                       mode=gnn_args.get("split", "pair"),
                       val_frac=gnn_args.get("val_frac", 0.2),
                       test_frac=gnn_args.get("test_frac", 0.1),
                       seed=gnn_args.get("seed", 42))
    gnn_val_ds, gnn_test_ds = Subset(ds, split["val"]), Subset(ds, split["test"])

    # The RL policy never touches the GNN's training pairs (the GNN is
    # overconfident on those, so calibrating there would be biased) or its
    # test pairs (kept untouched for the final, unbiased comparison). It is
    # trained and checkpoint-selected entirely inside the GNN's val pairs.
    n_select = max(1, int(len(gnn_val_ds) * args.select_frac))
    rl_train_ds, rl_select_ds = random_split(
        gnn_val_ds, [len(gnn_val_ds) - n_select, n_select],
        generator=torch.Generator().manual_seed(args.seed),
    )
    print(f"RL split (inside GNN val): {len(rl_train_ds)} train / "
          f"{len(rl_select_ds)} select  |  final test = GNN test ({len(gnn_test_ds)})\n")

    # ── Create RL components ─────────────────────────────────────────────────
    state_dim = get_state_dim(embed_dim)
    policy = RLPolicyNetwork(state_dim=state_dim)
    print(f"RL Policy parameters: {sum(p.numel() for p in policy.parameters()):,}")

    train_env = DDIEnvironment(base_model, rl_train_ds, device=device)
    select_env = DDIEnvironment(base_model, rl_select_ds, device=device)
    test_env = DDIEnvironment(base_model, gnn_test_ds, device=device)

    trainer = RLTrainer(
        policy=policy,
        environment=train_env,
        lr=args.lr,
        entropy_coeff=args.entropy_coeff,
        device=device,
    )

    # ── Training loop ────────────────────────────────────────────────────────
    history = {
        "episode": [], "loss": [], "mean_reward": [],
        "rl_accuracy": [], "base_accuracy": [],
        "eval_rl_accuracy": [], "eval_base_accuracy": [],
        "mean_adjustment": [],
    }

    best_eval_acc = 0.0
    patience_cnt = 0

    header = f"{'Ep':>5} {'Loss':>8} {'Reward':>8} {'RL Acc':>8} {'Base Acc':>8} {'Adj μ':>7} {'Adj σ':>7}"
    print(header)
    print("─" * len(header))

    for episode in range(1, args.episodes + 1):
        t0 = time.time()

        # Train one episode
        stats = trainer.train_episode(batch_size=args.batch)

        if "error" in stats:
            print(f"  Episode {episode}: {stats['error']}")
            continue

        history["episode"].append(episode)
        history["loss"].append(stats["loss"])
        history["mean_reward"].append(stats["mean_reward"])
        history["rl_accuracy"].append(stats["rl_accuracy"])
        history["base_accuracy"].append(stats["base_accuracy"])
        history["mean_adjustment"].append(stats["mean_adjustment"])

        flag = ""

        # Evaluate periodically
        if episode % args.eval_every == 0 or episode == 1:
            # Temporarily swap environment for evaluation
            trainer_env_backup = trainer.env
            trainer.env = select_env
            eval_stats = trainer.evaluate(batch_size=64)
            trainer.env = trainer_env_backup

            history["eval_rl_accuracy"].append(eval_stats["rl_accuracy"])
            history["eval_base_accuracy"].append(eval_stats["base_accuracy"])

            if eval_stats["rl_accuracy"] > best_eval_acc:
                best_eval_acc = eval_stats["rl_accuracy"]
                patience_cnt = 0
                # Save best RL policy
                torch.save(
                    policy.state_dict(),
                    os.path.join(out_dir, "rl_policy.pt")
                )
                flag = " ✓"
            else:
                patience_cnt += args.eval_every

            print(f"{episode:>5}  {stats['loss']:>7.4f}  {stats['mean_reward']:>7.4f}  "
                  f"{stats['rl_accuracy']:>7.4f}  {stats['base_accuracy']:>7.4f}  "
                  f"{stats['mean_adjustment']:>+6.4f}  {stats['std_adjustment']:>6.4f}  "
                  f"select_acc={eval_stats['rl_accuracy']:.4f}  "
                  f"{time.time()-t0:.1f}s{flag}")
        else:
            print(f"{episode:>5}  {stats['loss']:>7.4f}  {stats['mean_reward']:>7.4f}  "
                  f"{stats['rl_accuracy']:>7.4f}  {stats['base_accuracy']:>7.4f}  "
                  f"{stats['mean_adjustment']:>+6.4f}  {stats['std_adjustment']:>6.4f}  "
                  f"{time.time()-t0:.1f}s")

        if patience_cnt >= args.patience:
            print(f"\nEarly stopping at episode {episode} (patience={args.patience})")
            break

    # ── Final evaluation ─────────────────────────────────────────────────────
    print(f"\n{'─'*50}")
    print("Loading best RL policy...")
    rl_path = os.path.join(out_dir, "rl_policy.pt")
    if os.path.exists(rl_path):
        policy.load_state_dict(torch.load(rl_path, map_location=device))

    trainer.env = select_env
    select_eval = trainer.evaluate(batch_size=64)

    # The GNN test pairs were never used to train or select the RL policy,
    # so this is the only unbiased estimate of what RL calibration adds.
    trainer.env = test_env
    final_eval = trainer.evaluate(batch_size=64)
    improvement = final_eval['rl_accuracy'] - final_eval['base_accuracy']

    print(f"\n  Selection set ({select_eval['n_samples']} pairs, used to pick the policy — optimistic):")
    print(f"    Base {select_eval['base_accuracy']:.4f}  ->  RL {select_eval['rl_accuracy']:.4f}")
    print(f"  GNN test set ({final_eval['n_samples']} pairs, never seen by RL — unbiased):")
    print(f"    Base GNN accuracy  : {final_eval['base_accuracy']:.4f}")
    print(f"    RL accuracy        : {final_eval['rl_accuracy']:.4f}")
    print(f"    Improvement        : {improvement:+.4f} ({improvement*100:+.1f}%)")
    print(f"{'─'*50}")

    # ── Save RL metadata ─────────────────────────────────────────────────────
    rl_meta = {
        "best_select_accuracy": best_eval_acc,
        "select_rl_accuracy": select_eval["rl_accuracy"],
        "select_base_accuracy": select_eval["base_accuracy"],
        "test_rl_accuracy": final_eval["rl_accuracy"],
        "test_base_accuracy": final_eval["base_accuracy"],
        "improvement": improvement,
        "split": {"rl_train": len(rl_train_ds), "rl_select": len(rl_select_ds),
                  "test": len(gnn_test_ds)},
        "state_dim": state_dim,
        "embed_dim": embed_dim,
        "episodes": args.episodes,
        "lr": args.lr,
        "rl_enabled": True,
    }
    os.makedirs("results", exist_ok=True)
    results_path = os.path.join("results", f"rl_{tag}.json")
    with open(results_path, "w") as f:
        json.dump(rl_meta, f, indent=2)
    print(f"RL summary saved to {results_path}")

    # ── Save training curves ─────────────────────────────────────────────────
    _save_rl_plot(history, out_dir)

    return policy


def _save_rl_plot(history: dict, save_dir: str):
    """Save RL training curves."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))

    eps = history["episode"]

    # Loss curve
    axes[0].plot(eps, history["loss"], color="tab:blue", linewidth=1.5, alpha=0.7)
    axes[0].set_title("RL Policy Loss")
    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("Loss")

    # Reward curve
    axes[1].plot(eps, history["mean_reward"], color="tab:green", linewidth=1.5, alpha=0.7)
    axes[1].set_title("Mean Reward")
    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("Reward")

    # Accuracy comparison
    axes[2].plot(eps, history["rl_accuracy"], color="tab:orange", linewidth=1.5,
                 alpha=0.7, label="RL-enhanced")
    axes[2].plot(eps, history["base_accuracy"], color="tab:gray", linewidth=1.5,
                 alpha=0.5, linestyle="--", label="Base GNN")
    if history["eval_rl_accuracy"]:
        eval_eps = [eps[i] for i in range(len(eps))
                    if i < len(history["eval_rl_accuracy"])]
        eval_eps = list(range(1, len(history["eval_rl_accuracy"]) + 1))
        # Plot eval points at the right episode indices
        eval_indices = [e for e in eps if e == 1 or e % 25 == 0][:len(history["eval_rl_accuracy"])]
        if eval_indices:
            axes[2].scatter(eval_indices, history["eval_rl_accuracy"][:len(eval_indices)],
                           color="tab:red", s=30, zorder=5, label="Eval accuracy")
    axes[2].set_title("Accuracy: RL vs Base GNN")
    axes[2].set_xlabel("Episode")
    axes[2].set_ylabel("Accuracy")
    axes[2].legend()
    axes[2].set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(f"{save_dir}/rl_training_curves.png", dpi=120, bbox_inches="tight")
    plt.close()
    print(f"RL training curves saved to {save_dir}/rl_training_curves.png")


if __name__ == "__main__":
    train_rl(parse_args())
