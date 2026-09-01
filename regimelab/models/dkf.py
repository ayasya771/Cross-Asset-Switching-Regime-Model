"""Phase 3: the Differentiable Kalman Filter (continuous-discrete EKF).

The filtering equations of an extended Kalman filter are written directly in
torch ops, so gradients flow through every predict/update into the SDE drift,
the diffusion, the emission decoder, and the observation noise. This is the
"structured inference network" of the blueprint, replacing a black-box RNN
filter.

Predict (between observations, over an interval dt that genuinely varies with
the calendar):
    m   <- m + f(m) dt
    P   <- F P F^T + G G^T dt,   F = I + (df/dz)(m) dt
Update (at an observation with per-dimension mask s in {0,1}^n):
    H rows for missing dims are zeroed and their R entries set to 1, which
    provably reduces to the exact lower-dimensional update (the Kalman gain
    columns for masked dims vanish), so batches with heterogeneous missingness
    keep a fixed shape. Joseph form keeps covariances PSD.

Jacobians come from torch.func.jacfwd under vmap: forward-mode AD, the same
trick the blueprint recommends via jax.jacfwd, efficient because the input
dimension (latent) is small.
"""
from __future__ import annotations

from typing import Callable

import torch

Fn = Callable[[torch.Tensor], torch.Tensor]


def batched_jacobian(fn: Fn, x: torch.Tensor) -> torch.Tensor:
    """Row-wise Jacobian: x (B, d_in) -> (B, d_out, d_in) via vmap(jacfwd)."""

    def single(v: torch.Tensor) -> torch.Tensor:
        return fn(v.unsqueeze(0)).squeeze(0)

    return torch.func.vmap(torch.func.jacfwd(single))(x)


def _sym(P: torch.Tensor) -> torch.Tensor:
    return 0.5 * (P + P.mT)


def ekf_predict(
    f: Fn,
    g: Fn,
    m: torch.Tensor,
    P: torch.Tensor,
    dt: torch.Tensor,
    jitter: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor]:
    B, d = m.shape
    eye = torch.eye(d, dtype=m.dtype, device=m.device)
    dt1 = dt.view(B, 1)
    dt2 = dt.view(B, 1, 1)

    fm = f(m)
    A = batched_jacobian(f, m)
    F = eye + A * dt2
    m_pred = m + fm * dt1
    G = g(m)
    Q = torch.diag_embed(G * G) * dt2
    P_pred = _sym(F @ P @ F.mT + Q) + jitter * eye
    return m_pred, P_pred


def ekf_update(
    h: Fn,
    m_pred: torch.Tensor,
    P_pred: torch.Tensor,
    x: torch.Tensor,
    mask: torch.Tensor,
    R_diag: torch.Tensor,
    jitter: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    B, d = m_pred.shape
    n = x.shape[-1]
    eye_d = torch.eye(d, dtype=m_pred.dtype, device=m_pred.device)
    eye_n = torch.eye(n, dtype=m_pred.dtype, device=m_pred.device)

    xhat = h(m_pred)
    H = batched_jacobian(h, m_pred)
    Hm = H * mask.unsqueeze(-1)
    innov = (x - xhat) * mask
    Rm = R_diag.unsqueeze(0) * mask + (1.0 - mask)

    S = _sym(Hm @ P_pred @ Hm.mT + torch.diag_embed(Rm)) + jitter * eye_n
    Lchol = torch.linalg.cholesky(S)
    K = P_pred @ Hm.mT @ torch.cholesky_inverse(Lchol)

    m_new = m_pred + (K @ innov.unsqueeze(-1)).squeeze(-1)
    ImKH = eye_d - K @ Hm
    P_new = ImKH @ P_pred @ ImKH.mT + K @ torch.diag_embed(Rm) @ K.mT
    P_new = _sym(P_new) + jitter * eye_d
    return m_new, P_new, xhat


def gaussian_kl(
    m0: torch.Tensor,
    P0: torch.Tensor,
    m1: torch.Tensor,
    P1: torch.Tensor,
) -> torch.Tensor:
    """KL( N(m0, P0) || N(m1, P1) ) per batch row, via Cholesky factors."""
    d = m0.shape[-1]
    L1 = torch.linalg.cholesky(P1)
    L0 = torch.linalg.cholesky(P0)
    tr = torch.diagonal(torch.cholesky_solve(P0, L1), dim1=-2, dim2=-1).sum(-1)
    diff = (m1 - m0).unsqueeze(-1)
    quad = (diff * torch.cholesky_solve(diff, L1)).sum(dim=(-2, -1))
    logdet1 = 2.0 * torch.log(torch.diagonal(L1, dim1=-2, dim2=-1)).sum(-1)
    logdet0 = 2.0 * torch.log(torch.diagonal(L0, dim1=-2, dim2=-1)).sum(-1)
    return 0.5 * (tr + quad - d + logdet1 - logdet0)


def filter_sequence(
    f: Fn,
    g: Fn,
    h: Fn,
    R_diag: torch.Tensor,
    x: torch.Tensor,
    mask: torch.Tensor,
    dt: torch.Tensor,
    m0: torch.Tensor,
    P0: torch.Tensor,
    jitter: float = 1e-8,
) -> dict[str, torch.Tensor]:
    """Run the full continuous-discrete filter over a sequence.

    Returns per-step predicted (prior) and filtered (posterior) moments plus
    the pre-update emission means. All differentiable.
    """
    T = x.shape[1]
    m, P = m0, P0
    m_p, P_p, m_f, P_f, xh = [], [], [], [], []
    for t in range(T):
        m, P = ekf_predict(f, g, m, P, dt[:, t], jitter)
        m_p.append(m)
        P_p.append(P)
        m, P, xhat = ekf_update(h, m, P, x[:, t], mask[:, t], R_diag, jitter)
        m_f.append(m)
        P_f.append(P)
        xh.append(xhat)
    return {
        "m_pred": torch.stack(m_p, dim=1),
        "P_pred": torch.stack(P_p, dim=1),
        "m_filt": torch.stack(m_f, dim=1),
        "P_filt": torch.stack(P_f, dim=1),
        "xhat_pred": torch.stack(xh, dim=1),
    }
