"""The emission decoder's yield block must lie exactly on the Nelson-Siegel
manifold and standardize with the dataset scaler."""
import torch

from regimelab.models.decoder import EmissionDecoder, nelson_siegel_loadings

D, N = 4, 12
MATS = [0.25, 2.0, 5.0, 10.0, 30.0]
YM = [3.0, 3.2, 3.5, 3.8, 4.2]
YS = [1.5, 1.4, 1.2, 1.1, 1.0]


def _decoder():
    torch.manual_seed(0)
    return EmissionDecoder(D, N, hidden=32, maturities=MATS, yield_mean=YM, yield_std=YS).double()


def test_yield_block_is_nelson_siegel():
    dec = _decoder()
    z = torch.randn(7, D)
    out = dec(z)
    beta = dec.ns_factors(z)
    L = nelson_siegel_loadings(dec.ns_lambda, dec.taus)
    y_pct = beta @ L.mT
    y_std = (y_pct - dec.y_mean) / dec.y_std
    torch.testing.assert_close(out[:, : len(MATS)], y_std, atol=1e-12, rtol=0)


def test_ns_formula_manual():
    dec = _decoder()
    lam = dec.ns_lambda
    tau = dec.taus
    L = nelson_siegel_loadings(lam, tau)
    x = lam * tau
    l1 = (1 - torch.exp(-x)) / x
    l2 = l1 - torch.exp(-x)
    torch.testing.assert_close(L[:, 0], torch.ones_like(tau))
    torch.testing.assert_close(L[:, 1], l1)
    torch.testing.assert_close(L[:, 2], l2)
    assert float(lam) > 0


def test_bias_init_matches_mean_curve_shape():
    dec = _decoder()
    b = dec.ns_head.bias.detach()
    assert abs(float(b[0]) - YM[-1]) < 1e-9
    assert abs(float(b[1]) - (YM[0] - YM[-1])) < 1e-9


def test_r_diag_positive_and_shape():
    dec = _decoder()
    r = dec.r_diag()
    assert r.shape == (N,)
    assert float(r.min()) > 0


def test_handles_time_batched_input():
    dec = _decoder()
    z = torch.randn(3, 11, D)
    out = dec(z)
    assert out.shape == (3, 11, N)
