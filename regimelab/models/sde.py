"""Phase 2: the latent neural SDE prior.

Continuous-time transition dynamics  dz = f_theta(z) dt + g_theta(z) dW.

Stability-first parameterization (blueprint section 4.1):
- drift = Ornstein-Uhlenbeck backbone  -(L L^T + eps I) z + b  plus a
  zero-initialized MLP residual, so training starts from a well-posed linear
  mean-reverting system and learns nonlinearity gradually;
- diffusion = diagonal softplus(c + zero-initialized MLP), i.e. constant at
  initialization, state-dependent once training moves it.

The dynamics are deliberately time-homogeneous (f(z), not f(z, t)): regimes
should be recurring states, not functions of the calendar. The torchsde
wrapper below exposes the (t, y) signature the solver expects.
"""
from __future__ import annotations

import torch
from torch import nn

from .common import MLP, inv_softplus


class OUDrift(nn.Module):
    def __init__(self, latent_dim: int, hidden: int, mean_reversion_init: float = 2.0):
        super().__init__()
        scale = mean_reversion_init**0.5
        L0 = scale * torch.eye(latent_dim) + 0.01 * torch.randn(latent_dim, latent_dim)
        self.L = nn.Parameter(L0)
        self.bias = nn.Parameter(torch.zeros(latent_dim))
        self.res = MLP(latent_dim, hidden, latent_dim, zero_init_last=True)
        self.register_buffer("eps_eye", 1e-3 * torch.eye(latent_dim))

    def decay_matrix(self) -> torch.Tensor:
        """-(L L^T + eps I): symmetric negative definite, guarantees the
        linear part mean-reverts."""
        return -(self.L @ self.L.mT + self.eps_eye)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        A = self.decay_matrix()
        return z @ A.mT + self.bias + self.res(z)


class DiagonalDiffusion(nn.Module):
    def __init__(self, latent_dim: int, hidden: int, init: float = 0.3):
        super().__init__()
        self.c = nn.Parameter(torch.full((latent_dim,), inv_softplus(init)))
        self.res = MLP(latent_dim, hidden, latent_dim, zero_init_last=True)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return nn.functional.softplus(self.c + self.res(z)) + 1e-4


class PriorSDE(nn.Module):
    """torchsde-compatible wrapper around the drift/diffusion networks."""

    noise_type = "diagonal"
    sde_type = "ito"

    def __init__(self, drift: OUDrift, diffusion: DiagonalDiffusion):
        super().__init__()
        self.drift = drift
        self.diffusion = diffusion

    def f(self, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.drift(y)

    def g(self, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.diffusion(y)


@torch.no_grad()
def sample_prior_paths(
    drift: OUDrift,
    diffusion: DiagonalDiffusion,
    z0: torch.Tensor,
    ts: torch.Tensor,
    method: str = "euler",
) -> torch.Tensor:
    """Simulate latent paths under the prior with torchsde (Euler-Maruyama).

    z0: (B, d) initial states; ts: (T,) increasing times in years.
    Returns (T, B, d). Used for model diagnostics and tests, not training:
    training propagates moments through the differentiable Kalman filter.
    """
    import torchsde

    sde = PriorSDE(drift, diffusion)
    dt = float(torch.diff(ts).min().clamp_min(1e-4)) / 2.0
    return torchsde.sdeint(sde, z0, ts, method=method, dt=dt)
