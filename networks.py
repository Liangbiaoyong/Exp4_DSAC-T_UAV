"""
Neural network definitions for DSAC-T.

Actor: CNN (grid map) + MLP (point cloud, state, dynamic obs) → Tanh-squashed Gaussian.
Distributed Critic: CNN (grid map) + action concatenation → quantile distribution (32 quantiles).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from config import CONFIG


# =============================================================================
#  CNN Encoder (shared between Actor and Critic)
# =============================================================================

class CNNEncoder(nn.Module):
    """3-layer CNN for occupancy grid encoding."""
    def __init__(self, in_channels: int = 1):
        super().__init__()
        cfg = CONFIG["network"]["cnn_channels"]  # [1, 32, 64, 64]
        self.layers = nn.ModuleList()
        for i in range(len(cfg) - 1):
            self.layers.append(nn.Sequential(
                nn.Conv2d(cfg[i], cfg[i + 1],
                          kernel_size=CONFIG["network"]["cnn_kernel"],
                          stride=CONFIG["network"]["cnn_stride"],
                          padding=CONFIG["network"]["cnn_padding"]),
                nn.ReLU(inplace=True),
            ))

        # Compute flattened size after CNN
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels,
                                CONFIG["obs"]["grid_h"],
                                CONFIG["obs"]["grid_w"])
            for layer in self.layers:
                dummy = layer(dummy)
            self.cnn_feat_dim = dummy.view(1, -1).size(1)

    def forward(self, grid_map: torch.Tensor) -> torch.Tensor:
        """
        grid_map: (batch, 1, H, W)
        returns: (batch, cnn_feat_dim)
        """
        x = grid_map
        for layer in self.layers:
            x = layer(x)
        return x.view(x.size(0), -1)


# =============================================================================
#  Actor Network
# =============================================================================

class Actor(nn.Module):
    """
    Gaussian policy network with Tanh squashing.

    Inputs:
        grid_map: (batch, 1, 160, 160)
        pointcloud: (batch, 60)
        self_state: (batch, 6)
        dynamic_obs: (batch, 25)

    Outputs:
        mu, sigma: (batch, 2) — Gaussian parameters
    """
    def __init__(self, grid_h: int = None, grid_w: int = None,
                 pc_dim: int = None, state_dim: int = None,
                 dyn_dim: int = None):
        super().__init__()
        grid_h = grid_h or CONFIG["obs"]["grid_h"]
        grid_w = grid_w or CONFIG["obs"]["grid_w"]
        pc_dim = pc_dim or CONFIG["obs"]["pointcloud_dim"]
        state_dim = state_dim or CONFIG["obs"]["state_dim"]
        dyn_dim = dyn_dim or CONFIG["obs"]["dyn_obs_dim"]

        # CNN for grid map
        self.cnn = CNNEncoder(1)

        # MLP for vector observations
        mlp_hidden = CONFIG["network"]["mlp_hidden"]  # [128, 128]
        vec_dim = pc_dim + state_dim + dyn_dim
        self.mlp = nn.Sequential(
            nn.Linear(vec_dim, mlp_hidden[0]),
            nn.ReLU(inplace=True),
            nn.Linear(mlp_hidden[0], mlp_hidden[1]),
            nn.ReLU(inplace=True),
        )

        # Combined head
        fc_hidden = CONFIG["network"]["fc_hidden"]  # [256, 256]
        combined_dim = self.cnn.cnn_feat_dim + mlp_hidden[1]
        self.fc = nn.Sequential(
            nn.Linear(combined_dim, fc_hidden[0]),
            nn.ReLU(inplace=True),
            nn.Linear(fc_hidden[0], fc_hidden[1]),
            nn.ReLU(inplace=True),
        )

        # Output heads
        action_dim = CONFIG["network"]["action_dim"]  # 2
        self.mu_head = nn.Linear(fc_hidden[1], action_dim)
        self.sigma_head = nn.Linear(fc_hidden[1], action_dim)

        # Initialize
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                if hasattr(m, 'weight') and m.weight is not None:
                    nn.init.xavier_uniform_(m.weight, gain=0.01)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, grid_map: torch.Tensor, pointcloud: torch.Tensor,
                self_state: torch.Tensor, dynamic_obs: torch.Tensor):
        """
        Returns: (mu, sigma) where mu, sigma are each (batch, action_dim).
        """
        # CNN encoding
        grid_feat = self.cnn(grid_map)

        # MLP encoding
        vec_feat = self.mlp(torch.cat([pointcloud, self_state, dynamic_obs], dim=-1))

        # Combine
        combined = torch.cat([grid_feat, vec_feat], dim=-1)
        h = self.fc(combined)

        mu = self.mu_head(h)
        sigma = F.softplus(self.sigma_head(h)) + 1e-4
        return mu, sigma

    def sample(self, grid_map: torch.Tensor, pointcloud: torch.Tensor,
               self_state: torch.Tensor, dynamic_obs: torch.Tensor) -> tuple:
        """Sample action from Gaussian policy, log_prob, and Tanh-squashed action."""
        mu, sigma = self.forward(grid_map, pointcloud, self_state, dynamic_obs)

        # Reparameterization trick
        normal = torch.distributions.Normal(mu, sigma)
        z = normal.rsample()  # (batch, action_dim)
        action = torch.tanh(z)

        # Log probability with Tanh correction
        log_prob = normal.log_prob(z) - torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)

        return action, log_prob, mu, sigma


# =============================================================================
#  Distributed Critic Network
# =============================================================================

class DistributedCritic(nn.Module):
    """
    Quantile-distribution critic Z(s, a).
    Outputs num_quantiles quantile values per state-action pair.

    Uses twin Q-networks for Clipped Double Q-learning.
    """
    def __init__(self, grid_h: int = None, grid_w: int = None,
                 pc_dim: int = None, state_dim: int = None,
                 dyn_dim: int = None, num_quantiles: int = None):
        super().__init__()
        grid_h = grid_h or CONFIG["obs"]["grid_h"]
        grid_w = grid_w or CONFIG["obs"]["grid_w"]
        pc_dim = pc_dim or CONFIG["obs"]["pointcloud_dim"]
        state_dim = state_dim or CONFIG["obs"]["state_dim"]
        dyn_dim = dyn_dim or CONFIG["obs"]["dyn_obs_dim"]
        self.num_quantiles = num_quantiles or CONFIG["network"]["num_quantiles"]

        # CNN for grid map
        self.cnn1 = CNNEncoder(1)
        self.cnn2 = CNNEncoder(1)

        # MLP for vector observations + action
        mlp_hidden = CONFIG["network"]["mlp_hidden"]
        vec_dim = pc_dim + state_dim + dyn_dim + CONFIG["network"]["action_dim"]
        self.mlp1 = nn.Sequential(
            nn.Linear(vec_dim, mlp_hidden[0]),
            nn.ReLU(inplace=True),
            nn.Linear(mlp_hidden[0], mlp_hidden[1]),
            nn.ReLU(inplace=True),
        )
        self.mlp2 = nn.Sequential(
            nn.Linear(vec_dim, mlp_hidden[0]),
            nn.ReLU(inplace=True),
            nn.Linear(mlp_hidden[0], mlp_hidden[1]),
            nn.ReLU(inplace=True),
        )

        # Combined head
        fc_hidden = CONFIG["network"]["fc_hidden"]
        combined_dim = self.cnn1.cnn_feat_dim + mlp_hidden[1]

        self.fc1 = nn.Sequential(
            nn.Linear(combined_dim, fc_hidden[0]),
            nn.ReLU(inplace=True),
            nn.Linear(fc_hidden[0], fc_hidden[1]),
            nn.ReLU(inplace=True),
            nn.Linear(fc_hidden[1], self.num_quantiles),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(combined_dim, fc_hidden[0]),
            nn.ReLU(inplace=True),
            nn.Linear(fc_hidden[0], fc_hidden[1]),
            nn.ReLU(inplace=True),
            nn.Linear(fc_hidden[1], self.num_quantiles),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                if hasattr(m, 'weight') and m.weight is not None:
                    nn.init.xavier_uniform_(m.weight, gain=1.0)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, grid_map: torch.Tensor, pointcloud: torch.Tensor,
                self_state: torch.Tensor, dynamic_obs: torch.Tensor,
                action: torch.Tensor) -> tuple:
        """
        Returns: (z1, z2) each (batch, num_quantiles) from twin critics.
        """
        vec = torch.cat([pointcloud, self_state, dynamic_obs, action], dim=-1)

        # Critic 1
        grid_feat1 = self.cnn1(grid_map)
        vec_feat1 = self.mlp1(vec)
        z1 = self.fc1(torch.cat([grid_feat1, vec_feat1], dim=-1))

        # Critic 2
        grid_feat2 = self.cnn2(grid_map)
        vec_feat2 = self.mlp2(vec)
        z2 = self.fc2(torch.cat([grid_feat2, vec_feat2], dim=-1))

        return z1, z2


# =============================================================================
#  Temperature (alpha) module
# =============================================================================

class Temperature(nn.Module):
    """Learnable temperature parameter for entropy regularization."""
    def __init__(self, init_value: float = None):
        super().__init__()
        init = init_value or CONFIG["network"]["log_alpha_init"]
        self.log_alpha = nn.Parameter(torch.tensor(init, dtype=torch.float32))

    def forward(self) -> torch.Tensor:
        return self.log_alpha.exp()

    def get_alpha(self) -> float:
        return self.log_alpha.exp().item()
