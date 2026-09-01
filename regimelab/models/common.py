"""Small shared building blocks."""
from __future__ import annotations

import math

import torch
from torch import nn


def inv_softplus(y: float) -> float:
    """x such that softplus(x) = y, numerically safe for y > 0."""
    if y > 20:
        return y
    return math.log(math.expm1(y))


class MLP(nn.Module):
    """Two-layer tanh MLP. ``zero_init_last`` makes the whole map start at
    zero, so nonlinear terms switch on gradually during training (the
    blueprint's SDE-stability mitigation)."""

    def __init__(self, d_in: int, hidden: int, d_out: int, zero_init_last: bool = False):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, hidden), nn.Tanh(), nn.Linear(hidden, d_out))
        if zero_init_last:
            last = self.net[-1]
            nn.init.zeros_(last.weight)
            nn.init.zeros_(last.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
