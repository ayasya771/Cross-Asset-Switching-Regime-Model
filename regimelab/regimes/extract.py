"""Phase 6: regime extraction and analysis.

Freeze the trained model, run the differentiable Kalman filter over the whole
history, cluster the filtered latent trajectory with a Gaussian mixture (k
chosen by BIC), and characterize each regime in observable terms. Labels are
smoothed with a small modal window so one-day flickers do not fragment the
timeline. When the synthetic generator's ground-truth chain is present, the
extraction is scored with the adjusted Rand index.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
import polars as pl
import torch
from sklearn.metrics import adjusted_rand_score
from sklearn.mixture import GaussianMixture

from ..config import RegimeConfig
from ..data.dataset import full_sequence
from ..training.trainer import load_checkpoint


def modal_smooth(labels: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return labels.copy()
    half = window // 2
    out = labels.copy()
    for i in range(len(labels)):
        lo, hi = max(0, i - half), min(len(labels), i + half + 1)
        out[i] = Counter(labels[lo:hi].tolist()).most_common(1)[0][0]
    return out


def _regime_names(rows: list[dict]) -> list[str]:
    """Heuristic labels from observable characteristics; rows ordered by
    ascending average volatility."""
    k = len(rows)
    hy_med = float(np.median([r["hy_oas"] for r in rows]))
    names = [""] * k
    names[0] = "Calm Risk-On"
    if k > 1:
        names[-1] = "Crisis / Stress"
    for i in range(1, k - 1):
        r = rows[i]
        if r["slope"] < 0.3:
            names[i] = "Late-Cycle / Flat Curve"
        elif r["ann_ret"] < 0:
            names[i] = "Risk-Off / Drawdown"
        elif r["hy_oas"] > hy_med:
            names[i] = "Recovery / Wide Credit"
        else:
            names[i] = "Mid-Cycle"
    seen: dict[str, int] = {}
    for i, n in enumerate(names):
        seen[n] = seen.get(n, 0) + 1
        if seen[n] > 1:
            names[i] = f"{n} {seen[n]}"
    return names


def extract(
    cfg: RegimeConfig,
    processed_dir: Path,
    checkpoint: Path,
    raw_dir: Path | None = None,
    out_dir: Path | None = None,
) -> dict:
    torch.set_default_dtype(torch.float64)
    out_dir = Path(out_dir) if out_dir else processed_dir.parent.parent / "outputs" / "regimes"
    out_dir.mkdir(parents=True, exist_ok=True)

    model, meta, ckpt = load_checkpoint(checkpoint)
    seq = full_sequence(processed_dir)
    fs = model.filter_states(seq["x"], seq["m"], seq["dt"])
    m_f = fs["m_filt"][0].numpy()
    P_diag = np.diagonal(fs["P_filt"][0].numpy(), axis1=-2, axis2=-1)
    dates = seq["dates"].astype("datetime64[D]")
    T = len(dates)

    z = (m_f - m_f.mean(0)) / np.maximum(m_f.std(0), 1e-9)

    bic_rows = []
    best = None
    for k in range(cfg.k_min, cfg.k_max + 1):
        gmm = GaussianMixture(
            n_components=k,
            covariance_type="full",
            n_init=cfg.gmm_n_init,
            random_state=cfg.seed,
            reg_covar=1e-6,
        ).fit(z)
        bic = float(gmm.bic(z))
        bic_rows.append({"k": k, "bic": bic})
        if best is None or bic < best[1]:
            best = (k, bic, gmm)
    k, _, gmm = best
    raw_labels = gmm.predict(z)
    probs = gmm.predict_proba(z)

    panel = pl.read_parquet(processed_dir / "panel_raw.parquet")
    vix = panel["vix"].to_numpy()
    order = np.argsort([np.nanmean(vix[raw_labels == j]) for j in range(k)])
    remap = np.empty(k, dtype=int)
    remap[order] = np.arange(k)
    labels_raw = remap[raw_labels]
    probs = probs[:, order]
    labels = modal_smooth(labels_raw, cfg.smooth_window)

    spx_ret = panel["spx_ret"].to_numpy()
    hy = panel["hy_oas"].to_numpy()
    slope = panel["y_10y"].to_numpy() - panel["y_3m"].to_numpy()
    rows = []
    for j in range(k):
        sel = labels == j
        runs = [len(list(g)) for lab, g in _groupby_runs(labels) if lab == j]
        rows.append({
            "regime": j,
            "days": int(sel.sum()),
            "occupancy": 100.0 * float(sel.mean()),
            "vix": float(np.nanmean(vix[sel])),
            "hy_oas": float(np.nanmean(hy[sel])),
            "slope": float(np.nanmean(slope[sel])),
            "ann_ret": 100.0 * 252.0 * float(np.nanmean(spx_ret[sel])),
            "ann_vol": 100.0 * float(np.sqrt(252.0) * np.nanstd(spx_ret[sel])),
            "avg_duration_days": float(np.mean(runs)) if runs else 0.0,
        })
    names = _regime_names(rows)
    for r, n in zip(rows, names):
        r["name"] = n

    P = np.zeros((k, k))
    for a, b in zip(labels[:-1], labels[1:]):
        P[a, b] += 1
    P = P / np.maximum(P.sum(1, keepdims=True), 1)
    durations = [1.0 / max(1.0 - P[j, j], 1e-6) for j in range(k)]

    ari = None
    ari_true_k = None
    true_k = None
    truth = None
    truth_names: list[str] = []
    if raw_dir is not None and (Path(raw_dir) / "ground_truth_regimes.csv").exists():
        gt = pl.read_csv(Path(raw_dir) / "ground_truth_regimes.csv")
        gt = gt.with_columns(pl.col("date").str.to_date())
        gt_map = dict(zip(gt["date"].to_numpy().astype("datetime64[D]"), gt["regime"].to_numpy()))
        aligned = np.array([gt_map.get(d, -1) for d in dates])
        ok = aligned >= 0
        ari = float(adjusted_rand_score(aligned[ok], labels[ok]))
        truth = aligned
        truth_names = (
            gt.unique(subset=["regime"], keep="first")
            .sort("regime")["regime_name"]
            .to_list()
        )
        true_k = len(truth_names)
        if true_k != k:
            gmm_t = GaussianMixture(
                n_components=true_k,
                covariance_type="full",
                n_init=cfg.gmm_n_init,
                random_state=cfg.seed,
                reg_covar=1e-6,
            ).fit(z)
            lab_t = modal_smooth(gmm_t.predict(z), cfg.smooth_window)
            ari_true_k = float(adjusted_rand_score(aligned[ok], lab_t[ok]))
        else:
            ari_true_k = ari

    lab_df = pl.DataFrame({
        "date": dates.astype("datetime64[us]"),
        "label_raw": labels_raw,
        "label": labels,
        "regime_name": [names[j] for j in labels],
    })
    for j in range(k):
        lab_df = lab_df.with_columns(pl.Series(f"prob_{j}", probs[:, j]))
    lab_df.write_parquet(out_dir / "labels.parquet")
    np.savez_compressed(out_dir / "latent.npz", m_filt=m_f, P_diag=P_diag,
                        z=z, dates=seq["dates"])

    summary = {
        "k": k,
        "bic": bic_rows,
        "names": names,
        "stats": rows,
        "transition_matrix": P.tolist(),
        "expected_duration_days": durations,
        "ari_vs_ground_truth": ari,
        "true_k": true_k,
        "ari_at_true_k": ari_true_k,
        "checkpoint_epoch": ckpt.get("epoch"),
        "checkpoint_val_loss": ckpt.get("val_loss"),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    return {
        "summary": summary,
        "labels": labels,
        "labels_raw": labels_raw,
        "probs": probs,
        "dates": dates,
        "m_filt": m_f,
        "truth": truth,
        "truth_names": truth_names,
        "panel": panel,
        "meta": meta,
        "model": model,
    }


def _groupby_runs(labels: np.ndarray):
    """Yield (label, iterator) per contiguous run, itertools.groupby-style."""
    from itertools import groupby

    return groupby(labels.tolist())
