"""
train_rl.py  —  Phase 7: Reinforcement Learning Fine-Tuning
────────────────────────────────────────────────────────────
Trains an RL confidence calibration agent on top of the frozen GNN model.
The RL agent learns to adjust the base model's interaction probability
predictions to improve accuracy.

Requires: a trained GNN checkpoint from train.py

Usage
─────
  # Default (500 episodes)
  python train_rl.py

  # Custom episodes and learning rate
  python train_rl.py --episodes 1000 --lr 1e-4

  # Use a specific checkpoint directory
  python train_rl.py --checkpoint_dir checkpoints/ --episodes 500
"""

import os
import sys
import json
import argparse
import time

import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import random_split

sys.path.insert(0, ".")
from data.data_loader import load_dataset, dataset_stats
from data.ddi_dataset import DDIDataset
from models.gnn_ddi import DDIPredictor
from models.rl_agent import (
    RLPolicyNetwork, DDIEnvironment, RLTrainer, get_state_dim,
)


def parse_args():
    p = argparse.ArgumentParser(description="Train RL calibration agent (Phase 7)")
    p.add_argument("--checkpoint_dir", default="checkpoints",
                   help="Directory with trained GNN checkpoint")
    p.add_argument("--source",     default="twosides",
                   choices=["toy", "drugbank", "twosides", "csv"])
    p.add_argument("--data",       default=None, help="Path to data file")
    p.add_argument("--max_pairs",  type=int, default=10000,
                   help="Max positive drug pairs (same as train.py)")
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
    embed_dim = meta.get("args", {}).get("embed", 256)

    # ── Load data ────────────────────────────────────────────────────────────
    print(f"\nLoading data (source={args.source})...")
    df = load_dataset(source=args.source, path=args.data, max_pairs=args.max_pairs)
    dataset_stats(df)

    ds = DDIDataset(df=df)
    n = len(ds)

    # Split: 80% RL training, 20% RL evaluation
    n_eval = max(1, int(n * 0.2))
    n_train = n - n_eval
    train_ds, eval_ds = random_split(
        ds, [n_train, n_eval],
        generator=torch.Generator().manual_seed(args.seed)
    )
    print(f"RL split: {n_train} train / {n_eval} eval\n")

    # ── Create RL components ─────────────────────────────────────────────────
    state_dim = get_state_dim(embed_dim)
    policy = RLPolicyNetwork(state_dim=state_dim)
    print(f"RL Policy parameters: {sum(p.numel() for p in policy.parameters()):,}")

    train_env = DDIEnvironment(base_model, train_ds, device=device)
    eval_env = DDIEnvironment(base_model, eval_ds, device=device)

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
            trainer.env = eval_env
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
                    os.path.join(args.checkpoint_dir, "rl_policy.pt")
                )
                flag = " ✓"
            else:
                patience_cnt += args.eval_every

            print(f"{episode:>5}  {stats['loss']:>7.4f}  {stats['mean_reward']:>7.4f}  "
                  f"{stats['rl_accuracy']:>7.4f}  {stats['base_accuracy']:>7.4f}  "
                  f"{stats['mean_adjustment']:>+6.4f}  {stats['std_adjustment']:>6.4f}  "
                  f"eval_acc={eval_stats['rl_accuracy']:.4f}  "
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
    rl_path = os.path.join(args.checkpoint_dir, "rl_policy.pt")
    if os.path.exists(rl_path):
        policy.load_state_dict(torch.load(rl_path, map_location=device))

    trainer.env = eval_env
    final_eval = trainer.evaluate(batch_size=64)

    print(f"\n  Final RL accuracy  : {final_eval['rl_accuracy']:.4f}")
    print(f"  Base GNN accuracy  : {final_eval['base_accuracy']:.4f}")
    improvement = final_eval['rl_accuracy'] - final_eval['base_accuracy']
    print(f"  Improvement        : {improvement:+.4f} ({improvement*100:+.1f}%)")
    print(f"  Mean reward        : {final_eval['mean_reward']:.4f}")
    print(f"{'─'*50}")

    # ── Save RL metadata ─────────────────────────────────────────────────────
    rl_meta = {
        "best_eval_accuracy": best_eval_acc,
        "final_rl_accuracy": final_eval["rl_accuracy"],
        "final_base_accuracy": final_eval["base_accuracy"],
        "improvement": improvement,
        "state_dim": state_dim,
        "embed_dim": embed_dim,
        "episodes": args.episodes,
        "lr": args.lr,
        "rl_enabled": True,
    }
    with open(os.path.join(args.checkpoint_dir, "rl_meta.json"), "w") as f:
        json.dump(rl_meta, f, indent=2)
    print(f"RL metadata saved to {args.checkpoint_dir}/rl_meta.json")

    # ── Save training curves ─────────────────────────────────────────────────
    _save_rl_plot(history, args.checkpoint_dir)

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
