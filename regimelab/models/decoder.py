"""Phase 4: emission decoder with a Nelson-Siegel structural head.

The decoder maps latent states to the standardized observation vector. The
yield-curve block is *not* free: latents produce three Nelson-Siegel factors
(level, slope, curvature) and yields follow

    y(tau) = b0 + b1 * (1 - e^{-lam tau}) / (lam tau)
           + b2 * [ (1 - e^{-lam tau}) / (lam tau) - e^{-lam tau} ],

so every reconstructed curve lies on a smooth 3-factor manifold, the
blueprint's structural prior on the emission model (its "no-arbitrage-style
constraint" example). The decay lam (per year) is a learnable positive scalar.

All remaining features come from a free linear head over a shared trunk.
Outputs live in z-score space; the yield head produces percent-unit yields and
standardizes them with the dataset scaler stored as buffers, so the constraint
binds in real units.
"""
from __future__ import annotations

import math

import torch
from torch import nn

from .common import inv_softplus

LOG_2PI = math.log(2.0 * math.pi)


def nelson_siegel_loadings(lam: torch.Tensor, taus: torch.Tensor) -> torch.Tensor:
    """(K, 3) loadings for maturities taus (years) at decay lam (> 0)."""
    x = lam * taus
    ex = torch.exp(-x)
    l1 = (1.0 - ex) / x
    l2 = l1 - ex
    return torch.stack([torch.ones_like(taus), l1, l2], dim=-1)


class EmissionDecoder(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        obs_dim: int,
        hidden: int,
        maturities: list[float],
        yield_mean: list[float],
        yield_std: list[float],
        ns_lambda_init: float = 0.6,
        obs_noise_init: float = 0.5,
    ):
        super().__init__()
        self.n_yield = len(maturities)
        if obs_dim < self.n_yield:
            raise ValueError("obs_dim smaller than the yield block")
        self.trunk = nn.Sequential(nn.Linear(latent_dim, hidden), nn.Tanh())
        self.ns_head = nn.Linear(hidden, 3)
        self.other_head = nn.Linear(hidden, obs_dim - self.n_yield)
        self.raw_lambda = nn.Parameter(torch.tensor(inv_softplus(max(ns_lambda_init - 0.05, 1e-3))))
        self.rho = nn.Parameter(torch.full((obs_dim,), inv_softplus(obs_noise_init)))
        self.register_buffer("taus", torch.tensor(maturities, dtype=torch.get_default_dtype()))
        self.register_buffer("y_mean", torch.tensor(yield_mean, dtype=torch.get_default_dtype()))
        self.register_buffer("y_std", torch.tensor(yield_std, dtype=torch.get_default_dtype()))
        with torch.no_grad():
            self.ns_head.bias.copy_(
                torch.tensor([yield_mean[-1], yield_mean[0] - yield_mean[-1], 0.0])
            )

    @property
    def ns_lambda(self) -> torch.Tensor:
        return nn.functional.softplus(self.raw_lambda) + 0.05

    def ns_factors(self, z: torch.Tensor) -> torch.Tensor:
        """(…, 3) Nelson-Siegel factors in percent units."""
        return self.ns_head(self.trunk(z))

    def yields_percent(self, z: torch.Tensor) -> torch.Tensor:
        beta = self.ns_factors(z)
        L = nelson_siegel_loadings(self.ns_lambda, self.taus)
        return beta @ L.mT if beta.dim() > 1 else L @ beta

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        hcore = self.trunk(z)
        beta = self.ns_head(hcore)
        L = nelson_siegel_loadings(self.ns_lambda, self.taus)
        y_pct = torch.matmul(beta, L.mT)
        y_std = (y_pct - self.y_mean) / self.y_std
        other = self.other_head(hcore)
        return torch.cat([y_std, other], dim=-1)

    def r_diag(self) -> torch.Tensor:
        """Per-dimension observation noise variances (z-score units)."""
        std = nn.functional.softplus(self.rho) + 1e-3
        return std * std
