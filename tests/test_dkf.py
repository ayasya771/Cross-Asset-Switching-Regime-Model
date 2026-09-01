"""The differentiable Kalman filter must agree with a hand-written linear
Kalman filter whenever drift and emission are linear, and its masking scheme
must reduce exactly to the lower-dimensional update."""
import numpy as np
import torch

from regimelab.models.dkf import ekf_predict, ekf_update, filter_sequence, gaussian_kl

D, N = 3, 5
JIT = 0.0


def _setup(seed=0):
    rng = np.random.default_rng(seed)
    A = -np.eye(D) * 0.8 + 0.1 * rng.standard_normal((D, D))
    b = rng.standard_normal(D) * 0.1
    G = np.abs(rng.standard_normal(D)) * 0.4 + 0.1
    H = rng.standard_normal((N, D))
    c = rng.standard_normal(N) * 0.2
    R = np.abs(rng.standard_normal(N)) * 0.3 + 0.05
    m = rng.standard_normal(D)
    S = rng.standard_normal((D, D))
    P = S @ S.T + 0.5 * np.eye(D)
    x = rng.standard_normal(N)
    return A, b, G, H, c, R, m, P, x


def _torch_fns(A, b, G, H, c):
    tA, tb = torch.tensor(A), torch.tensor(b)
    tG = torch.tensor(G)
    tH, tc = torch.tensor(H), torch.tensor(c)
    f = lambda z: z @ tA.mT + tb
    g = lambda z: tG.expand_as(z)
    h = lambda z: z @ tH.mT + tc
    return f, g, h


def test_predict_matches_linear_kf():
    A, b, G, H, c, R, m, P, x = _setup()
    f, g, _ = _torch_fns(A, b, G, H, c)
    dt = 0.02
    m_t, P_t = ekf_predict(
        f, g, torch.tensor(m).unsqueeze(0), torch.tensor(P).unsqueeze(0),
        torch.tensor([dt]), jitter=JIT,
    )
    F = np.eye(D) + A * dt
    m_ref = m + (A @ m + b) * dt
    P_ref = F @ P @ F.T + np.diag(G**2) * dt
    np.testing.assert_allclose(m_t.squeeze(0).numpy(), m_ref, atol=1e-12)
    np.testing.assert_allclose(P_t.squeeze(0).numpy(), P_ref, atol=1e-12)


def test_update_matches_linear_kf_full_mask():
    A, b, G, H, c, R, m, P, x = _setup()
    _, _, h = _torch_fns(A, b, G, H, c)
    mask = torch.ones(1, N)
    m_t, P_t, xhat = ekf_update(
        h, torch.tensor(m).unsqueeze(0), torch.tensor(P).unsqueeze(0),
        torch.tensor(x).unsqueeze(0), mask, torch.tensor(R), jitter=JIT,
    )
    S = H @ P @ H.T + np.diag(R)
    K = P @ H.T @ np.linalg.inv(S)
    innov = x - (H @ m + c)
    m_ref = m + K @ innov
    ImKH = np.eye(D) - K @ H
    P_ref = ImKH @ P @ ImKH.T + K @ np.diag(R) @ K.T
    np.testing.assert_allclose(m_t.squeeze(0).numpy(), m_ref, atol=1e-10)
    np.testing.assert_allclose(P_t.squeeze(0).numpy(), P_ref, atol=1e-10)
    np.testing.assert_allclose(xhat.squeeze(0).numpy(), H @ m + c, atol=1e-12)


def test_masked_update_equals_reduced_dimension_update():
    A, b, G, H, c, R, m, P, x = _setup(seed=3)
    _, _, h = _torch_fns(A, b, G, H, c)
    obs = np.array([0, 2, 4])
    mask = torch.zeros(1, N)
    mask[0, obs] = 1.0
    m_t, P_t, _ = ekf_update(
        h, torch.tensor(m).unsqueeze(0), torch.tensor(P).unsqueeze(0),
        torch.tensor(x).unsqueeze(0), mask, torch.tensor(R), jitter=JIT,
    )
    Ho, Ro, xo, co = H[obs], R[obs], x[obs], c[obs]
    S = Ho @ P @ Ho.T + np.diag(Ro)
    K = P @ Ho.T @ np.linalg.inv(S)
    m_ref = m + K @ (xo - (Ho @ m + co))
    ImKH = np.eye(D) - K @ Ho
    P_ref = ImKH @ P @ ImKH.T + K @ np.diag(Ro) @ K.T
    np.testing.assert_allclose(m_t.squeeze(0).numpy(), m_ref, atol=1e-10)
    np.testing.assert_allclose(P_t.squeeze(0).numpy(), P_ref, atol=1e-10)


def test_fully_masked_step_is_identity():
    A, b, G, H, c, R, m, P, x = _setup(seed=5)
    _, _, h = _torch_fns(A, b, G, H, c)
    mask = torch.zeros(1, N)
    m_t, P_t, _ = ekf_update(
        h, torch.tensor(m).unsqueeze(0), torch.tensor(P).unsqueeze(0),
        torch.tensor(x).unsqueeze(0), mask, torch.tensor(R), jitter=JIT,
    )
    np.testing.assert_allclose(m_t.squeeze(0).numpy(), m, atol=1e-12)
    np.testing.assert_allclose(P_t.squeeze(0).numpy(), P, atol=1e-12)


def test_filter_keeps_covariance_spd():
    A, b, G, H, c, R, *_ = _setup(seed=7)
    f, g, h = _torch_fns(A, b, G, H, c)
    B, T = 4, 30
    torch.manual_seed(1)
    x = torch.randn(B, T, N)
    mask = (torch.rand(B, T, N) > 0.4).double()
    dt = torch.full((B, T), 1 / 252.0)
    dt[:, 0] = 0.0
    m0 = torch.zeros(B, D)
    P0 = torch.eye(D).expand(B, D, D).clone()
    out = filter_sequence(f, g, h, torch.tensor(R), x, mask, dt, m0, P0, jitter=1e-10)
    P = out["P_filt"].reshape(-1, D, D)
    asym = (P - P.mT).abs().max()
    assert float(asym) < 1e-9
    eig = torch.linalg.eigvalsh(P)
    assert float(eig.min()) > 0


def test_gaussian_kl_properties():
    S = torch.randn(2, D, D) * 0.3
    P = S @ S.mT + torch.eye(D)
    m = torch.randn(2, D)
    kl_same = gaussian_kl(m, P, m, P)
    assert float(kl_same.abs().max()) < 1e-10
    kl = gaussian_kl(m, P, m + 1.0, P * 2.0)
    assert float(kl.min()) > 0
