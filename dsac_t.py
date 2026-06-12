"""
DSAC-T (Distributed Soft Actor-Critic with Three Refinements) algorithm.

Three Refinements:
1. Distribution truncation: quantile range clipped to [-C_val, C_val]
2. Critic regularization: L2 penalty on critic weights
3. Soft target update: polyak averaging with tau

Key components:
  - ReplayBuffer with prioritized or uniform sampling
  - DSACTAgent: main algorithm class with update() method
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from collections import deque
import random

from config import CONFIG
from networks import Actor, DistributedCritic, Temperature


# =============================================================================
#  Replay Buffer
# =============================================================================

class ReplayBuffer:
    """Fixed-size replay buffer for experience replay."""

    def __init__(self, capacity: int = None):
        self.capacity = capacity or CONFIG["dsac_t"]["buffer_capacity"]
        self.buffer = deque(maxlen=self.capacity)

    def push(self, obs: Dict, action: np.ndarray, reward: float,
             next_obs: Dict, done: bool):
        """Store a transition."""
        self.buffer.append((obs, action, reward, next_obs, done))

    def sample(self, batch_size: int) -> Dict:
        """Sample a batch of transitions. Returns dict of tensors."""
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        batch_size = len(batch)

        # Extract components
        obs_list = [item[0] for item in batch]
        action_list = [item[1] for item in batch]
        reward_list = [item[2] for item in batch]
        next_obs_list = [item[3] for item in batch]
        done_list = [item[4] for item in batch]

        # Convert to tensors
        batch_data = {
            "grid_map": torch.stack([torch.from_numpy(o["grid_map"]).float().unsqueeze(0) for o in obs_list]),
            "pointcloud": torch.stack([torch.from_numpy(o["pointcloud"]).float() for o in obs_list]),
            "self_state": torch.stack([torch.from_numpy(o["self_state"]).float() for o in obs_list]),
            "dynamic_obs": torch.stack([torch.from_numpy(o["dynamic_obs"]).float() for o in obs_list]),
            "action": torch.stack([torch.from_numpy(np.array(a)).float() for a in action_list]),
            "reward": torch.tensor(reward_list, dtype=torch.float32).unsqueeze(-1),
            "next_grid_map": torch.stack([torch.from_numpy(n["grid_map"]).float().unsqueeze(0) for n in next_obs_list]),
            "next_pointcloud": torch.stack([torch.from_numpy(n["pointcloud"]).float() for n in next_obs_list]),
            "next_self_state": torch.stack([torch.from_numpy(n["self_state"]).float() for n in next_obs_list]),
            "next_dynamic_obs": torch.stack([torch.from_numpy(n["dynamic_obs"]).float() for n in next_obs_list]),
            "done": torch.tensor(done_list, dtype=torch.float32).unsqueeze(-1),
        }

        return batch_data

    def __len__(self) -> int:
        return len(self.buffer)


# =============================================================================
#  DSAC-T Agent
# =============================================================================

class DSACTAgent:
    """
    DSAC-T algorithm implementation.

    Key features:
    - Distributed critic with num_quantiles quantile outputs per twin Q
    - Distribution truncation (Refinement 1)
    - Critic L2 regularization (Refinement 2)
    - Soft target update with tau (Refinement 3)
    - Automatic entropy tuning via learnable alpha
    """

    def __init__(self,
                 grid_h: int = None, grid_w: int = None,
                 pc_dim: int = None, state_dim: int = None,
                 dyn_dim: int = None,
                 lr: float = None, gamma: float = None,
                 tau: float = None, num_quantiles: int = None,
                 c_val: float = None, critic_reg: float = None,
                 device: str = None):

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        dsac_cfg = CONFIG["dsac_t"]

        # Hyperparameters
        self.gamma = gamma or dsac_cfg["gamma"]
        self.tau = tau or dsac_cfg["tau"]                     # Refinement 3
        self.c_val = c_val or dsac_cfg["c_val"]               # Refinement 1
        self.critic_reg = critic_reg or dsac_cfg["critic_reg"]  # Refinement 2
        self.num_quantiles = num_quantiles or CONFIG["network"]["num_quantiles"]
        self.batch_size = dsac_cfg["batch_size"]
        self.update_every = dsac_cfg["update_every"]
        self.updates_per_step = dsac_cfg["updates_per_step"]

        # Build networks
        self.actor = Actor(grid_h, grid_w, pc_dim, state_dim, dyn_dim).to(self.device)
        self.critic = DistributedCritic(grid_h, grid_w, pc_dim, state_dim,
                                        dyn_dim, num_quantiles).to(self.device)
        self.critic_target = DistributedCritic(grid_h, grid_w, pc_dim, state_dim,
                                                dyn_dim, num_quantiles).to(self.device)
        self._hard_update_target()

        # Temperature (alpha)
        self.temperature = Temperature().to(self.device)

        # Optimizers
        lr = lr or dsac_cfg["lr"]
        weight_decay = dsac_cfg["weight_decay"]
        self.actor_opt = torch.optim.AdamW(self.actor.parameters(),
                                           lr=lr, weight_decay=weight_decay)
        self.critic_opt = torch.optim.AdamW(self.critic.parameters(),
                                             lr=lr, weight_decay=weight_decay)
        self.temp_opt = torch.optim.AdamW(self.temperature.parameters(),
                                          lr=lr, weight_decay=weight_decay)

        # Target entropy
        self.target_entropy = dsac_cfg["target_entropy"]

        # Replay buffer
        self.buffer = ReplayBuffer()

        # Step counter
        self.step = 0

    def _hard_update_target(self):
        """Copy critic weights to target."""
        self.critic_target.load_state_dict(self.critic.state_dict())

    def _soft_update_target(self):
        """Polyak averaging (Refinement 3)."""
        with torch.no_grad():
            for p, p_target in zip(self.critic.parameters(),
                                   self.critic_target.parameters()):
                p_target.data.mul_(1 - self.tau)
                p_target.data.add_(self.tau * p.data)

    def select_action(self, obs: Dict, deterministic: bool = False) -> np.ndarray:
        """Select action for a single observation (no batch dimension)."""
        with torch.no_grad():
            # Add batch dimension
            grid = torch.from_numpy(obs["grid_map"]).float().unsqueeze(0).unsqueeze(0).to(self.device)
            pc = torch.from_numpy(obs["pointcloud"]).float().unsqueeze(0).to(self.device)
            state = torch.from_numpy(obs["self_state"]).float().unsqueeze(0).to(self.device)
            dyn = torch.from_numpy(obs["dynamic_obs"]).float().unsqueeze(0).to(self.device)

            if deterministic:
                mu, sigma = self.actor(grid, pc, state, dyn)
                action = torch.tanh(mu)
            else:
                action, log_prob, mu, sigma = self.actor.sample(grid, pc, state, dyn)

            return action.cpu().numpy().squeeze(0)

    def _get_quantile_targets(self, batch: Dict) -> torch.Tensor:
        """
        Compute target quantile values using distributional Bellman target.
        Applies distribution truncation (Refinement 1).
        """
        with torch.no_grad():
            # Sample next actions from target policy
            next_action, next_log_prob, _, _ = self.actor.sample(
                batch["next_grid_map"], batch["next_pointcloud"],
                batch["next_self_state"], batch["next_dynamic_obs"]
            )

            # Target Z-values from both target critics
            z1_target, z2_target = self.critic_target(
                batch["next_grid_map"], batch["next_pointcloud"],
                batch["next_self_state"], batch["next_dynamic_obs"],
                next_action
            )

            # Clipped double Q: take minimum of the two
            z_target = torch.min(z1_target, z2_target)

            # Distribution truncation (Refinement 1)
            z_target = torch.clamp(z_target, -self.c_val, self.c_val)

            # Distributional Bellman target
            # reward + gamma * (1 - done) * (Z(s', a') - alpha * log_prob)
            alpha = self.temperature()
            target = batch["reward"] + self.gamma * (1 - batch["done"]) * (
                z_target - alpha * next_log_prob
            )

            # Truncation again on the full target
            target = torch.clamp(target, -self.c_val, self.c_val)

            return target

    def update_critic(self, batch: Dict) -> torch.Tensor:
        """
        Update twin critics using quantile regression.
        Returns: critic loss value.
        """
        # Current Z-values
        z1, z2 = self.critic(
            batch["grid_map"], batch["pointcloud"],
            batch["self_state"], batch["dynamic_obs"],
            batch["action"]
        )

        # Target quantiles
        with torch.no_grad():
            target = self._get_quantile_targets(batch)

        # Quantile Huber loss
        loss1 = self._quantile_huber_loss(z1, target)
        loss2 = self._quantile_huber_loss(z2, target)
        critic_loss = loss1 + loss2

        # Critic L2 regularization (Refinement 2)
        reg_loss = 0
        for param in self.critic.parameters():
            reg_loss += param.pow(2).sum()
        critic_loss = critic_loss + self.critic_reg * reg_loss

        # Update critics
        self.critic_opt.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
        self.critic_opt.step()

        return critic_loss

    def _quantile_huber_loss(self, z: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Quantile Huber loss for distributional RL.
        z: (batch, num_quantiles)
        target: (batch, num_quantiles)
        """
        batch_size = z.size(0)
        num_quantiles = z.size(1)

        # Tau values for each quantile (0 to 1)
        tau = torch.linspace(0, 1, num_quantiles + 1, device=self.device)[:-1]
        tau = tau.unsqueeze(0).expand(batch_size, -1)  # (batch, num_quantiles)

        # Pairwise difference: (batch, num_q, num_q)
        diff = target.unsqueeze(2) - z.unsqueeze(1)  # target - z

        # Huber loss
        huber = F.smooth_l1_loss(z.unsqueeze(1).expand(-1, num_quantiles, -1),
                                  target.unsqueeze(2).expand(-1, -1, num_quantiles),
                                  reduction='none')
        # We need |tau - I(diff < 0)| * huber
        # I(diff < 0): indicator where target < z
        indicator = (diff < 0).float()
        # Quantile weight
        weight = torch.abs(tau.unsqueeze(2) - indicator)  # (batch, num_q, num_q)

        loss = (weight * huber).sum(dim=(1, 2)).mean()

        return loss

    def update_actor(self, batch: Dict) -> torch.Tensor:
        """
        Update actor using the distributed critic.
        Returns: actor loss value.
        """
        # Sample actions from current policy
        action, log_prob, _, _ = self.actor.sample(
            batch["grid_map"], batch["pointcloud"],
            batch["self_state"], batch["dynamic_obs"]
        )

        # Critic value (use min of twin Q for stability)
        z1, z2 = self.critic(
            batch["grid_map"], batch["pointcloud"],
            batch["self_state"], batch["dynamic_obs"],
            action
        )
        z = torch.min(z1, z2).mean(dim=-1, keepdim=True)

        # Temperature
        alpha = self.temperature()

        # Actor loss: maximize Z - alpha * entropy
        actor_loss = (alpha * log_prob - z).mean()

        # Update actor
        self.actor_opt.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
        self.actor_opt.step()

        return actor_loss

    def update_temperature(self, log_prob: torch.Tensor) -> torch.Tensor:
        """Update learnable temperature alpha."""
        alpha_loss = -(self.temperature().log() * (log_prob.detach() + self.target_entropy)).mean()

        self.temp_opt.zero_grad()
        alpha_loss.backward()
        self.temp_opt.step()

        return alpha_loss

    def update(self, batch: Dict) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Perform a single update step (critic + actor + temperature).
        Returns: (critic_loss, actor_loss, alpha_loss)
        """
        # Move batch to device
        batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}

        # Update critic
        critic_loss = self.update_critic(batch)

        # Update actor
        actor_loss = self.update_actor(batch)

        # Update temperature
        with torch.no_grad():
            action, log_prob, _, _ = self.actor.sample(
                batch["grid_map"], batch["pointcloud"],
                batch["self_state"], batch["dynamic_obs"]
            )
        alpha_loss = self.update_temperature(log_prob)

        # Soft target update (Refinement 3)
        self._soft_update_target()

        self.step += 1

        return critic_loss, actor_loss, alpha_loss

    def save_checkpoint(self, path: str):
        """Save complete agent checkpoint."""
        torch.save({
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "critic_target": self.critic_target.state_dict(),
            "temperature": self.temperature.state_dict(),
            "optimizer_actor": self.actor_opt.state_dict(),
            "optimizer_critic": self.critic_opt.state_dict(),
            "optimizer_alpha": self.temp_opt.state_dict(),
            "step": self.step,
        }, path)

    def load_checkpoint(self, path: str):
        """Load agent from checkpoint."""
        ckpt = torch.load(path, map_location=self.device, weights_only=True)
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])
        self.critic_target.load_state_dict(ckpt["critic_target"])
        self.temperature.load_state_dict(ckpt["temperature"])
        self.actor_opt.load_state_dict(ckpt["optimizer_actor"])
        self.critic_opt.load_state_dict(ckpt["optimizer_critic"])
        self.temp_opt.load_state_dict(ckpt["optimizer_alpha"])
        self.step = ckpt["step"]
        return self.step
