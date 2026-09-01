"""End-to-end differentiability: the ELBO must be finite, every parameter must
receive gradients through the filter, and a few optimizer steps must reduce
the loss on a small synthetic problem."""
import torch

from regimelab.config import ModelConfig
from regimelab.models.model import LatentRegimeModel

YM = [3.0, 3.2, 3.5, 3.8, 4.2]
YS = [1.5, 1.4, 1.2, 1.1, 1.0]


def _model(n=10, d=3):
    torch.manual_seed(0)
    cfg = ModelConfig(obs_dim=n, latent_dim=d, hidden=24, enc_hidden=24)
    return LatentRegimeModel(cfg, YM, YS).double()


def _batch(B=4, T=16, n=10):
    torch.manual_seed(1)
    t = torch.linspace(0, 3.0, T)
    base = torch.stack([torch.sin(t + k) for k in range(n)], dim=-1)
    x = base.unsqueeze(0) + 0.1 * torch.randn(B, T, n)
    m = (torch.rand(B, T, n) > 0.25).double()
    dt = torch.full((B, T), 1 / 252.0)
    dt[:, 0] = 0.0
    return x.double(), m, dt.double()


def test_elbo_finite_and_grads_flow():
    model = _model()
    x, m, dt = _batch()
    out = model.elbo(x, m, dt, beta=1.0)
    assert bool(torch.isfinite(out["loss"]))
    out["loss"].backward()
    missing = [
        name
        for name, p in model.named_parameters()
        if p.grad is None or not bool(torch.isfinite(p.grad).all())
    ]
    assert missing == []
    dead = [
        name
        for name, p in model.named_parameters()
        if float(p.grad.abs().sum()) == 0.0 and "res.net.0" not in name
    ]
    assert dead == []


def test_kl_positive_and_beta_scales_loss():
    model = _model()
    x, m, dt = _batch()
    with torch.no_grad():
        out0 = model.elbo(x, m, dt, beta=0.0, sample=False)
        out1 = model.elbo(x, m, dt, beta=1.0, sample=False)
    assert float(out1["kl"]) > 0
    assert float(out1["loss"]) > float(out0["loss"])


def test_few_steps_reduce_loss():
    model = _model()
    x, m, dt = _batch()
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    first, last = None, None
    for i in range(30):
        out = model.elbo(x, m, dt, beta=0.1)
        opt.zero_grad()
        out["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        v = float(out["loss"])
        if i < 3:
            first = v if first is None else max(first, v)
        last = v
    assert last < first


def test_fully_masked_sequence_is_safe():
    model = _model()
    x, m, dt = _batch()
    m.zero_()
    out = model.elbo(x, m, dt, beta=1.0)
    assert bool(torch.isfinite(out["loss"]))
