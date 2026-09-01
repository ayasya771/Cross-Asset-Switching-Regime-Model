"""The VAE recognition component the filter does not replace: an amortized
posterior over the *initial* latent state of each window.

A GRU reads the window backward (values with missing entries zero-filled,
concatenated with the mask so the network knows what is real) and emits the
mean and diagonal covariance of q(z_0 | x_{t_0:t_N}). Every later filtering
distribution comes from the differentiable Kalman filter itself.
"""
from __future__ import annotations

import torch
from torch import nn


class InitialStateEncoder(nn.Module):
    def __init__(self, obs_dim: int, latent_dim: int, hidden: int):
        super().__init__()
        self.gru = nn.GRU(2 * obs_dim, hidden, batch_first=True)
        self.head = nn.Linear(hidden, 2 * latent_dim)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        inp = torch.cat([x * mask, mask], dim=-1).flip(1)
        _, h_n = self.gru(inp)
        out = self.head(h_n.squeeze(0))
        m0, raw = out.chunk(2, dim=-1)
        var0 = nn.functional.softplus(raw) + 1e-4
        return m0, torch.diag_embed(var0)
