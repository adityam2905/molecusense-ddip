"""
models/rl_agent.py  —  Reinforcement Learning Confidence Calibration Agent
─────────────────────────────────────────────────────────────────────────────
Uses REINFORCE (policy gradient) to learn optimal adjustments to the base
GNN model's interaction probability predictions.

RL Formulation
──────────────
  State   : [emb_a ‖ emb_b ‖ base_prob ‖ attn_stats_a ‖ attn_stats_b]
  Action  : continuous adjustment δ ∈ [-0.3, +0.3] to the base probability
  Reward  : +1 correct prediction, −1 incorrect; confidence bonus/penalty
  Policy  : Gaussian policy π(δ|s) = N(μ(s), σ(s))

Architecture
────────────
  State → MLP(state_dim → 256 → 128 → 64) → μ, log_σ
  Action δ ~ N(μ, σ), clamped to [-0.3, +0.3]
  Final prob = clamp(base_prob + δ, 0, 1)
"""

import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np


class RLPolicyNetwork(nn.Module):
    """
    Gaussian policy network for confidence calibration.

    Input  : state vector (GNN embeddings + base prob + attention stats)
    Output : mean μ and log-std log_σ for a Gaussian over the adjustment δ
    """

    def __init__(self, state_dim: int, hidden_dim: int = 256):
        super().__init__()

        self.feature_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
        )

        # Mean head: predicts the adjustment δ
        self.mu_head = nn.Linear(hidden_dim // 4, 1)

        # Log-std head: learned uncertainty
        self.log_std_head = nn.Linear(hidden_dim // 4, 1)

        # Value head (baseline for variance reduction)
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim // 4, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize weights with small values for stable RL training."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.1)
                nn.init.constant_(m.bias, 0)
        # Start with near-zero mean (no adjustment) and small std
        nn.init.constant_(self.mu_head.weight, 0)
        nn.init.constant_(self.mu_head.bias, 0)
        nn.init.constant_(self.log_std_head.bias, -1.0)  # σ ≈ 0.37

    def forward(self, state):
        """
        Parameters
        ----------
        state : [B, state_dim]

        Returns
        -------
        mu       : [B, 1] mean adjustment
        log_std  : [B, 1] log standard deviation
        value    : [B, 1] state value estimate (baseline)
        """
        features = self.feature_net(state)
        mu = torch.tanh(self.mu_head(features)) * 0.3      # clamp μ ∈ [-0.3, 0.3]
        log_std = torch.clamp(self.log_std_head(features), -3, 0)  # σ ∈ [0.05, 1.0]
        value = self.value_head(features)
        return mu, log_std, value

    def select_action(self, state, deterministic=False):
        """
        Sample an action (adjustment δ) from the policy.

        Returns
        -------
        action   : [B, 1] the adjustment δ
        log_prob : [B, 1] log π(δ|s)
        value    : [B, 1] V(s) baseline estimate
        """
        mu, log_std, value = self.forward(state)

        if deterministic:
            # At test time, use the mean directly
            action = mu
            log_prob = torch.zeros_like(action)
        else:
            std = log_std.exp()
            dist = Normal(mu, std)
            action = dist.rsample()                          # reparameterized sample
            log_prob = dist.log_prob(action)

        # Clamp action to valid range
        action = torch.clamp(action, -0.3, 0.3)
        return action, log_prob, value


class DDIEnvironment:
    """
    RL environment wrapping the trained DDI-GNN model.

    The environment presents drug pair states and rewards the agent
    for correctly calibrating interaction probabilities.
    """

    def __init__(self, model, dataset, device="cpu"):
        """
        Parameters
        ----------
        model   : trained DDIPredictor (frozen)
        dataset : DDIDataset or list of (graph_a, graph_b, label, meta)
        device  : torch device
        """
        self.model = model
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

        self.dataset = dataset
        self.device = device
        self.current_idx = 0
        self.indices = list(range(len(dataset)))

    def reset(self, shuffle=True):
        """Reset the environment for a new episode."""
        if shuffle:
            np.random.shuffle(self.indices)
        self.current_idx = 0

    def _extract_state(self, graph_a, graph_b, label):
        """
        Build the RL state vector from a drug pair.

        State components:
          1. Drug A embedding (embed_dim)
          2. Drug B embedding (embed_dim)
          3. Base model probability (1)
          4. Attention statistics for drug A: mean, std, max, min (4)
          5. Attention statistics for drug B: mean, std, max, min (4)
          6. Embedding cosine similarity (1)
          7. Embedding L2 distance (1)

        Total: embed_dim * 2 + 11
        """
        from torch_geometric.data import Batch

        ba = Batch.from_data_list([graph_a]).to(self.device)
        bb = Batch.from_data_list([graph_b]).to(self.device)

        with torch.no_grad():
            # Get embeddings and attention
            emb_a, attn_a = self.model.mol_gat(
                ba.x, ba.edge_index, ba.edge_attr, ba.batch,
                return_attention=True
            )
            emb_b, attn_b = self.model.mol_gat(
                bb.x, bb.edge_index, bb.edge_attr, bb.batch,
                return_attention=True
            )

            # Base probability
            pair = torch.cat([emb_a, emb_b], dim=1)
            logits = self.model.classifier(pair)
            if self.model.n_classes == 1:
                logits = logits.squeeze(-1)
            base_prob = torch.sigmoid(logits)

        # Attention statistics
        attn_stats_a = torch.tensor([
            attn_a.mean(), attn_a.std() if attn_a.numel() > 1 else 0.0,
            attn_a.max(), attn_a.min()
        ], device=self.device).unsqueeze(0)

        attn_stats_b = torch.tensor([
            attn_b.mean(), attn_b.std() if attn_b.numel() > 1 else 0.0,
            attn_b.max(), attn_b.min()
        ], device=self.device).unsqueeze(0)

        # Embedding similarity metrics
        cos_sim = F.cosine_similarity(emb_a, emb_b, dim=1).unsqueeze(1)
        l2_dist = torch.norm(emb_a - emb_b, dim=1).unsqueeze(1)

        # Compose full state
        state = torch.cat([
            emb_a, emb_b,
            base_prob.unsqueeze(0) if base_prob.dim() == 0 else base_prob.unsqueeze(1),
            attn_stats_a, attn_stats_b,
            cos_sim, l2_dist,
        ], dim=1)

        return state, base_prob

    def step(self, batch_size=1):
        """
        Get a batch of states from the dataset.

        Returns
        -------
        states      : [B, state_dim]
        base_probs  : [B]
        labels      : [B]
        done        : bool (True if epoch is complete)
        """
        states, base_probs, labels = [], [], []

        for _ in range(batch_size):
            if self.current_idx >= len(self.indices):
                break

            idx = self.indices[self.current_idx]
            self.current_idx += 1

            graph_a, graph_b, label, meta = self.dataset[idx]
            state, base_prob = self._extract_state(graph_a, graph_b, label)

            states.append(state)
            base_probs.append(base_prob)
            labels.append(label)

        if not states:
            return None, None, None, True

        states = torch.cat(states, dim=0)
        base_probs = torch.stack(base_probs) if isinstance(base_probs[0], torch.Tensor) else torch.tensor(base_probs)
        base_probs = base_probs.squeeze()
        labels = torch.stack(labels) if isinstance(labels[0], torch.Tensor) else torch.tensor(labels, dtype=torch.float)
        labels = labels.to(self.device)

        done = self.current_idx >= len(self.indices)
        return states, base_probs, labels, done

    @staticmethod
    def compute_reward(base_prob, adjustment, label, alpha=0.5):
        """
        Compute the reward for an RL action.

        Reward structure:
          - Correctness reward: +1 if final prediction matches label, -1 otherwise
          - Confidence bonus:   +α * |0.5 - final_prob| if correct (rewards certainty)
          - Improvement bonus:  +0.5 if RL made the prediction more correct than base
          - Worsening penalty:  -0.5 if RL made the prediction worse than base

        Parameters
        ----------
        base_prob  : base model probability [B]
        adjustment : RL agent's δ [B]
        label      : ground truth [B]
        alpha      : confidence bonus weight
        """
        final_prob = torch.clamp(base_prob + adjustment.squeeze(-1), 0, 1)
        pred = (final_prob >= 0.5).float()
        correct = (pred == label).float()

        # Base correctness
        reward = correct * 2.0 - 1.0  # +1 if correct, -1 if wrong

        # Confidence bonus for correct predictions
        confidence = torch.abs(final_prob - 0.5)
        reward += alpha * confidence * correct

        # Improvement/worsening vs base model
        base_pred = (base_prob >= 0.5).float()
        base_correct = (base_pred == label).float()

        # Bonus if RL fixed a wrong base prediction
        reward += 0.5 * (correct - base_correct)

        # Distance-based component: how close is final_prob to the true label
        distance_to_truth = torch.abs(final_prob - label)
        base_distance = torch.abs(base_prob - label)
        improvement = base_distance - distance_to_truth
        reward += 0.3 * improvement  # positive if closer to truth

        return reward


class RLTrainer:
    """
    REINFORCE trainer with baseline subtraction for the RL calibration agent.
    """

    def __init__(
        self,
        policy: RLPolicyNetwork,
        environment: DDIEnvironment,
        lr: float = 3e-4,
        gamma: float = 0.99,
        entropy_coeff: float = 0.01,
        value_coeff: float = 0.5,
        max_grad_norm: float = 0.5,
        device: str = "cpu",
    ):
        self.policy = policy.to(device)
        self.env = environment
        self.device = device
        self.gamma = gamma
        self.entropy_coeff = entropy_coeff
        self.value_coeff = value_coeff
        self.max_grad_norm = max_grad_norm

        self.optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=50, gamma=0.95
        )

    def train_episode(self, batch_size=32):
        """
        Run one episode (full pass through the dataset) and update the policy.

        Returns
        -------
        dict with episode statistics
        """
        self.policy.train()
        self.env.reset()

        all_log_probs = []
        all_rewards = []
        all_values = []
        all_entropies = []
        total_correct = 0
        total_base_correct = 0
        total_samples = 0
        total_adjustments = []

        done = False
        while not done:
            states, base_probs, labels, done = self.env.step(batch_size)
            if states is None:
                break

            states = states.to(self.device)

            # Policy forward
            actions, log_probs, values = self.policy.select_action(states)

            # Compute entropy for exploration bonus
            mu, log_std, _ = self.policy(states)
            std = log_std.exp()
            entropy = 0.5 * (1 + torch.log(2 * torch.tensor(np.pi) * std ** 2))

            # Compute rewards
            rewards = DDIEnvironment.compute_reward(base_probs, actions, labels)

            all_log_probs.append(log_probs)
            all_rewards.append(rewards)
            all_values.append(values.squeeze(-1))
            all_entropies.append(entropy.mean())

            # Stats
            final_probs = torch.clamp(base_probs + actions.squeeze(-1), 0, 1)
            preds = (final_probs >= 0.5).float()
            base_preds = (base_probs >= 0.5).float()
            total_correct += (preds == labels).sum().item()
            total_base_correct += (base_preds == labels).sum().item()
            total_samples += labels.size(0)
            total_adjustments.extend(actions.detach().cpu().numpy().flatten().tolist())

        if not all_log_probs:
            return {"error": "no samples processed"}

        # Stack everything
        log_probs = torch.cat(all_log_probs)
        rewards = torch.cat(all_rewards)
        values = torch.cat(all_values)

        # Normalize rewards (variance reduction)
        rewards_normalized = rewards
        if rewards.std() > 1e-8:
            rewards_normalized = (rewards - rewards.mean()) / (rewards.std() + 1e-8)

        # Advantage = reward - baseline value
        advantages = rewards_normalized - values.detach()

        # Policy loss: -E[log π(a|s) * A(s,a)]
        policy_loss = -(log_probs.squeeze(-1) * advantages).mean()

        # Value loss: MSE between value and actual reward
        value_loss = F.mse_loss(values, rewards_normalized)

        # Entropy loss: encourage exploration
        entropy_loss = -torch.stack(all_entropies).mean()

        # Total loss
        loss = (
            policy_loss
            + self.value_coeff * value_loss
            + self.entropy_coeff * entropy_loss
        )

        # Backward
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
        self.optimizer.step()
        self.scheduler.step()

        adj_arr = np.array(total_adjustments)
        return {
            "loss": loss.item(),
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "mean_reward": rewards.mean().item(),
            "rl_accuracy": total_correct / max(total_samples, 1),
            "base_accuracy": total_base_correct / max(total_samples, 1),
            "mean_adjustment": adj_arr.mean(),
            "std_adjustment": adj_arr.std(),
            "n_samples": total_samples,
        }

    @torch.no_grad()
    def evaluate(self, batch_size=64):
        """Evaluate the policy in deterministic mode."""
        self.policy.eval()
        self.env.reset(shuffle=False)

        total_correct = 0
        total_base_correct = 0
        total_samples = 0
        total_reward = 0

        done = False
        while not done:
            states, base_probs, labels, done = self.env.step(batch_size)
            if states is None:
                break

            states = states.to(self.device)
            actions, _, _ = self.policy.select_action(states, deterministic=True)
            rewards = DDIEnvironment.compute_reward(base_probs, actions, labels)

            final_probs = torch.clamp(base_probs + actions.squeeze(-1), 0, 1)
            preds = (final_probs >= 0.5).float()
            base_preds = (base_probs >= 0.5).float()

            total_correct += (preds == labels).sum().item()
            total_base_correct += (base_preds == labels).sum().item()
            total_samples += labels.size(0)
            total_reward += rewards.sum().item()

        return {
            "rl_accuracy": total_correct / max(total_samples, 1),
            "base_accuracy": total_base_correct / max(total_samples, 1),
            "mean_reward": total_reward / max(total_samples, 1),
            "n_samples": total_samples,
        }


def get_state_dim(embed_dim: int = 256) -> int:
    """Calculate the RL state dimension based on GNN embed_dim."""
    # emb_a + emb_b + base_prob + attn_stats_a(4) + attn_stats_b(4) + cos_sim + l2_dist
    return embed_dim * 2 + 11


# ── Smoke test ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    state_dim = get_state_dim(256)
    print(f"State dimension: {state_dim}")

    policy = RLPolicyNetwork(state_dim=state_dim)
    print(f"Policy parameters: {sum(p.numel() for p in policy.parameters()):,}")

    # Test forward pass
    dummy_state = torch.randn(4, state_dim)
    action, log_prob, value = policy.select_action(dummy_state)
    print(f"Action shape: {action.shape}")
    print(f"Actions: {action.squeeze().tolist()}")
    print(f"Log probs: {log_prob.squeeze().tolist()}")
    print(f"Values: {value.squeeze().tolist()}")
