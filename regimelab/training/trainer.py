"""Phase 5: end-to-end training.

Gradients backpropagate through the EKF recursion and the SDE moment
propagation (discretize-then-optimize). Blueprint mitigations wired in:
gradient clipping, weight decay, KL annealing, near-linear initialization of
the dynamics, and a non-finite-loss guard.
"""
from __future__ import annotations

import dataclasses
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..config import CHECKPOINTS, Config
from ..data.dataset import PanelWindows, load_meta
from ..models.model import build_model


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def kl_beta(step: int, total_steps: int, frac: float, beta_max: float) -> float:
    ramp = max(1, int(total_steps * frac))
    return beta_max * min(1.0, step / ramp)


@torch.no_grad()
def evaluate(model, loader, beta: float) -> dict:
    model.eval()
    agg = {"loss": 0.0, "nll": 0.0, "kl": 0.0}
    for batch in loader:
        out = model.elbo(batch["x"], batch["m"], batch["dt"], beta=beta, sample=False)
        for k in agg:
            agg[k] += float(out[k])
    n = max(len(loader), 1)
    return {k: v / n for k, v in agg.items()}


def train(cfg: Config, processed_dir: Path, out_dir: Path = CHECKPOINTS) -> dict:
    t_cfg = cfg.train
    set_seed(t_cfg.seed)
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(max(torch.get_num_threads(), 2))
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "metrics.jsonl"

    meta = load_meta(processed_dir)
    train_ds = PanelWindows(processed_dir, t_cfg.window, t_cfg.stride, "train", t_cfg.val_frac)
    val_ds = PanelWindows(processed_dir, t_cfg.window, t_cfg.stride, "val", t_cfg.val_frac)
    train_loader = DataLoader(train_ds, batch_size=t_cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=t_cfg.batch_size, shuffle=False)
    print(f"train windows: {len(train_ds)}  val windows: {len(val_ds)}  obs_dim: {len(meta['features'])}", flush=True)

    model = build_model(cfg.model, meta)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"parameters: {n_params}", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=t_cfg.lr, weight_decay=t_cfg.weight_decay)
    total_steps = max(1, t_cfg.epochs * len(train_loader))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_steps)

    best_val = float("inf")
    bad_epochs = 0
    history: list[dict] = []
    step = 0

    def save(name: str, epoch: int, val_loss: float) -> None:
        torch.save(
            {
                "model_state": model.state_dict(),
                "model_cfg": dataclasses.asdict(
                    dataclasses.replace(cfg.model, obs_dim=len(meta["features"]))
                ),
                "meta": meta,
                "epoch": epoch,
                "val_loss": val_loss,
            },
            out_dir / name,
        )

    for epoch in range(t_cfg.epochs):
        model.train()
        t0 = time.time()
        ep_loss, ep_nll, ep_kl, n_batches, skipped = 0.0, 0.0, 0.0, 0, 0
        for batch in train_loader:
            beta = kl_beta(step, total_steps, t_cfg.kl_anneal_frac, t_cfg.beta_max)
            out = model.elbo(batch["x"], batch["m"], batch["dt"], beta=beta, sample=True)
            loss = out["loss"]
            opt.zero_grad(set_to_none=True)
            if not torch.isfinite(loss):
                skipped += 1
                step += 1
                sched.step()
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), t_cfg.grad_clip)
            opt.step()
            sched.step()
            ep_loss += float(loss)
            ep_nll += float(out["nll"])
            ep_kl += float(out["kl"])
            n_batches += 1
            step += 1
        if skipped > max(2, len(train_loader) // 5):
            raise RuntimeError(
                f"epoch {epoch}: {skipped} non-finite losses; lower lr or diffusion_init"
            )

        val = evaluate(model, val_loader, beta=t_cfg.beta_max)
        rec = {
            "epoch": epoch,
            "train_loss": ep_loss / max(n_batches, 1),
            "train_nll": ep_nll / max(n_batches, 1),
            "train_kl": ep_kl / max(n_batches, 1),
            "val_loss": val["loss"],
            "val_nll": val["nll"],
            "val_kl": val["kl"],
            "beta": kl_beta(step, total_steps, t_cfg.kl_anneal_frac, t_cfg.beta_max),
            "lr": sched.get_last_lr()[0],
            "skipped": skipped,
            "seconds": round(time.time() - t0, 1),
        }
        history.append(rec)
        with log_path.open("a") as fh:
            fh.write(json.dumps(rec) + "\n")
        print(
            f"epoch {epoch:02d}  train {rec['train_loss']:.4f}  val {rec['val_loss']:.4f}  "
            f"nll {rec['val_nll']:.4f}  kl {rec['val_kl']:.4f}  beta {rec['beta']:.2f}  "
            f"({rec['seconds']}s)",
            flush=True,
        )

        save("last.pt", epoch, val["loss"])
        if val["loss"] < best_val - 1e-5:
            best_val = val["loss"]
            bad_epochs = 0
            save("best.pt", epoch, val["loss"])
        else:
            bad_epochs += 1
            if bad_epochs >= t_cfg.patience:
                print(f"early stop at epoch {epoch} (best val {best_val:.4f})", flush=True)
                break

    return {"best_val": best_val, "history": history}


def load_checkpoint(path: Path):
    """Rebuild a trained model from a checkpoint file."""
    from ..config import ModelConfig

    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    mcfg = ModelConfig(**ckpt["model_cfg"])
    model = build_model(mcfg, ckpt["meta"])
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, ckpt["meta"], ckpt
