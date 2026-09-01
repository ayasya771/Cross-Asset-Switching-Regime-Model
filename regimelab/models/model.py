"""The assembled latent state-space model and its ELBO (Phase 4/5).

Loss (per batch, normalized by the number of *observed* entries):

    L = E_q[ -log p(x | z) ] + beta * ( sum_t KL(q_t || p_t) + KL(q_0 || N(0, I)) )

where q_t = N(m_t^+, P_t^+) is the filtered posterior produced by the DKF and
p_t = N(m_t^-, P_t^-) the SDE-propagated prediction. Both are Gaussian, so the
path KL is closed form, the filtering-based counterpart of the Girsanov path
KL used by torchsde's latent SDE. beta implements KL annealing / beta-VAE
(blueprint's mode-collapse mitigation). Reconstruction uses a reparameterized
sample from q_t through the decoder, so gradients reach every component:
encoder (initial state), drift/diffusion (through predict), decoder and
observation noise (through update, reconstruction, and the Kalman gain).
"""
from __future__ import annotations

import torch
from torch import nn

from ..config import YIELD_MATURITIES, ModelConfig
from .decoder import LOG_2PI, EmissionDecoder
from .dkf import filter_sequence, gaussian_kl
from .encoder import InitialStateEncoder
from .sde import DiagonalDiffusion, OUDrift


class LatentRegimeModel(nn.Module):
    def __init__(
        self,
        cfg: ModelConfig,
        yield_mean: list[float],
        yield_std: list[float],
        maturities: list[float] | None = None,
    ):
        super().__init__()
        self.cfg = cfg
        maturities = maturities or list(YIELD_MATURITIES.values())
        self.encoder = InitialStateEncoder(cfg.obs_dim, cfg.latent_dim, cfg.enc_hidden)
        self.drift = OUDrift(cfg.latent_dim, cfg.hidden, cfg.mean_reversion_init)
        self.diffusion = DiagonalDiffusion(cfg.latent_dim, cfg.hidden, cfg.diffusion_init)
        self.decoder = EmissionDecoder(
            cfg.latent_dim,
            cfg.obs_dim,
            cfg.hidden,
            maturities,
            yield_mean,
            yield_std,
            cfg.ns_lambda_init,
            cfg.obs_noise_init,
        )

    def filter(self, x: torch.Tensor, m: torch.Tensor, dt: torch.Tensor) -> dict:
        m0, P0 = self.encoder(x, m)
        out = filter_sequence(
            self.drift,
            self.diffusion,
            self.decoder,
            self.decoder.r_diag(),
            x,
            m,
            dt,
            m0,
            P0,
            self.cfg.jitter,
        )
        out["m0"], out["P0"] = m0, P0
        return out

    def elbo(
        self,
        x: torch.Tensor,
        m: torch.Tensor,
        dt: torch.Tensor,
        beta: float = 1.0,
        sample: bool = True,
    ) -> dict:
        B, T, n = x.shape
        d = self.cfg.latent_dim
        out = self.filter(x, m, dt)
        m_f = out["m_filt"]
        P_f = out["P_filt"]

        kl_t = gaussian_kl(
            m_f.reshape(B * T, d),
            P_f.reshape(B * T, d, d),
            out["m_pred"].reshape(B * T, d),
            out["P_pred"].reshape(B * T, d, d),
        ).reshape(B, T).sum(-1)

        zeros = torch.zeros(B, d, dtype=x.dtype, device=x.device)
        eye = torch.eye(d, dtype=x.dtype, device=x.device).expand(B, d, d)
        kl0 = gaussian_kl(out["m0"], out["P0"], zeros, eye)

        if sample:
            Lf = torch.linalg.cholesky(P_f.reshape(B * T, d, d))
            eps = torch.randn(B * T, d, dtype=x.dtype, device=x.device)
            z = (m_f.reshape(B * T, d) + (Lf @ eps.unsqueeze(-1)).squeeze(-1)).reshape(B, T, d)
        else:
            z = m_f
        xhat = self.decoder(z)
        R = self.decoder.r_diag()
        nll_elem = 0.5 * ((x - xhat) ** 2 / R + torch.log(R) + LOG_2PI)
        nll = (nll_elem * m).sum(dim=(1, 2))

        n_entries = m.sum().clamp_min(1.0)
        total_nll = nll.sum()
        total_kl = kl_t.sum() + kl0.sum()
        loss = (total_nll + beta * total_kl) / n_entries
        return {
            "loss": loss,
            "nll": (total_nll / n_entries).detach(),
            "kl": (total_kl / n_entries).detach(),
            "xhat": xhat.detach(),
            "m_filt": m_f.detach(),
        }

    @torch.no_grad()
    def filter_states(self, x: torch.Tensor, m: torch.Tensor, dt: torch.Tensor) -> dict:
        """Frozen-model filtering pass (Phase 6). Returns posterior means,
        covariances and one-step-ahead emission means."""
        out = self.filter(x, m, dt)
        return {
            "m_filt": out["m_filt"],
            "P_filt": out["P_filt"],
            "xhat_pred": out["xhat_pred"],
        }


def build_model(cfg: ModelConfig, meta: dict) -> LatentRegimeModel:
    """Construct the model against a processed dataset's meta.json."""
    import dataclasses

    cfg = dataclasses.replace(cfg, obs_dim=len(meta["features"]))
    model = LatentRegimeModel(cfg, meta["yield_mean"], meta["yield_std"])
    return model.double()
