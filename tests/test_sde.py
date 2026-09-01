"""The neural SDE must start as a stable linear OU system (blueprint's
training-stability mitigation) and integrate with torchsde."""
import torch

from regimelab.models.sde import DiagonalDiffusion, OUDrift, sample_prior_paths

D = 4


def test_drift_is_linear_at_init():
    torch.manual_seed(0)
    drift = OUDrift(D, hidden=16).double()
    z = torch.randn(8, D)
    A = drift.decay_matrix()
    expected = z @ A.mT + drift.bias
    torch.testing.assert_close(drift(z), expected, atol=1e-12, rtol=0)


def test_decay_matrix_is_negative_definite():
    torch.manual_seed(0)
    drift = OUDrift(D, hidden=16).double()
    eig = torch.linalg.eigvalsh(drift.decay_matrix())
    assert float(eig.max()) < 0


def test_diffusion_positive_and_constant_at_init():
    torch.manual_seed(0)
    diff = DiagonalDiffusion(D, hidden=16, init=0.3).double()
    z1, z2 = torch.randn(5, D), torch.randn(5, D)
    g1, g2 = diff(z1), diff(z2)
    assert float(g1.min()) > 0
    torch.testing.assert_close(g1, g2, atol=1e-12, rtol=0)
    torch.testing.assert_close(
        g1[0], torch.full((D,), 0.3, dtype=torch.float64) + 1e-4, atol=1e-6, rtol=0
    )


def test_mean_reversion_shrinks_state():
    torch.manual_seed(0)
    drift = OUDrift(D, hidden=16).double()
    with torch.no_grad():
        drift.bias.zero_()
    z = torch.randn(6, D) * 3.0
    dt = 0.01
    z_next = z + drift(z) * dt
    assert float(z_next.norm(dim=-1).max()) < float(z.norm(dim=-1).max())


def test_torchsde_prior_simulation():
    torch.manual_seed(0)
    drift = OUDrift(D, hidden=16).double()
    diff = DiagonalDiffusion(D, hidden=16).double()
    z0 = torch.randn(3, D)
    ts = torch.linspace(0.0, 0.1, 12).double()
    paths = sample_prior_paths(drift, diff, z0, ts)
    assert paths.shape == (12, 3, D)
    assert bool(torch.isfinite(paths).all())
