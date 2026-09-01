"""Phase 1: raw series -> aligned, masked, standardized observation arrays.

Handles the blueprint's asynchronous-sampling requirement explicitly:

- The time grid is the union of all trading dates on which at least one daily
  market series printed. Nothing is forward-filled.
- Monthly and quarterly macro series are attached to the first grid date at or
  after their reference date; every other day they are simply *missing* and the
  mask says so. The continuous-discrete filter consumes (value, mask, dt)
  triples, so mixed frequencies need no resampling.
- dt is measured in calendar years between consecutive grid dates, so weekends
  and holidays genuinely stretch the latent SDE's integration interval.
"""
from __future__ import annotations

import json

import numpy as np
import polars as pl

from ..config import (
    FEATURES,
    LOG_FEATURES,
    YIELD_FEATURES,
    DataConfig,
)
from .sources import load_raw

GRID_SERIES = ["y_10y", "vix", "spx", "ndx", "hy_oas"]
MACRO_SPEC = [
    ("cpi", "cpi_yoy", 12),
    ("indpro", "indpro_yoy", 12),
    ("unrate", "unrate", None),
    ("gdp", "gdp_yoy", 4),
]


def _yoy(df: pl.DataFrame, periods: int) -> pl.DataFrame:
    return df.sort("date").with_columns(
        (100.0 * (pl.col("value") / pl.col("value").shift(periods) - 1.0)).alias("value")
    ).drop_nulls(subset=["value"])


def _attach_sparse(grid: np.ndarray, obs: pl.DataFrame) -> np.ndarray:
    """Place each sparse observation on the first grid date >= its own date."""
    out = np.full(len(grid), np.nan)
    if obs.height == 0:
        return out
    dates = obs["date"].to_numpy().astype("datetime64[D]")
    vals = obs["value"].to_numpy().astype(np.float64)
    idx = np.searchsorted(grid, dates, side="left")
    for i, v in zip(idx, vals):
        if i < len(grid):
            out[i] = v
    return out


def build_panel(cfg: DataConfig, split_frac: float = 0.85) -> dict:
    """Build the aligned panel and model arrays; returns the meta dict."""
    cfg.processed_dir.mkdir(parents=True, exist_ok=True)
    raw = {name: load_raw(cfg, name) for name in [
        "y_3m", "y_2y", "y_5y", "y_10y", "y_30y",
        "hy_oas", "ig_oas", "vix", "ted", "usd_idx", "wti",
        "spx", "ndx", "cpi", "indpro", "unrate", "gdp",
    ]}

    start = np.datetime64(cfg.start, "D")
    end = np.datetime64(cfg.end, "D") if cfg.end else np.datetime64("today", "D")

    grid_dates: set[np.datetime64] = set()
    for name in GRID_SERIES:
        df = raw[name]
        if df.height:
            d = df.drop_nulls(subset=["value"])["date"].to_numpy().astype("datetime64[D]")
            grid_dates.update(d[(d >= start) & (d <= end)])
    if not grid_dates:
        raise RuntimeError("no grid dates: every grid series failed to download")
    grid = np.array(sorted(grid_dates), dtype="datetime64[D]")
    T = len(grid)

    def on_grid(name: str) -> np.ndarray:
        df = raw[name].drop_nulls(subset=["value"])
        out = np.full(T, np.nan)
        if df.height == 0:
            return out
        d = df["date"].to_numpy().astype("datetime64[D]")
        v = df["value"].to_numpy().astype(np.float64)
        idx = np.searchsorted(grid, d)
        ok = (idx < T) & (grid[np.minimum(idx, T - 1)] == d)
        out[idx[ok]] = v[ok]
        return out

    cols: dict[str, np.ndarray] = {}
    for f in YIELD_FEATURES + ["hy_oas", "ig_oas", "vix", "ted"]:
        cols[f] = on_grid(f)

    def logdiff(x: np.ndarray) -> np.ndarray:
        lx = np.where(x > 0, np.log(np.maximum(x, 1e-12)), np.nan)
        out = np.full_like(lx, np.nan)
        out[1:] = lx[1:] - lx[:-1]
        return out

    usd = on_grid("usd_idx")
    cols["usd_ret"] = logdiff(usd)

    wti = on_grid("wti")
    oil_ret = np.full(T, np.nan)
    prev = wti[:-1]
    cur = wti[1:]
    with np.errstate(invalid="ignore", divide="ignore"):
        r = (cur - prev) / np.maximum(np.abs(prev), 5.0)
    oil_ret[1:] = np.clip(r, -cfg.oil_ret_clip, cfg.oil_ret_clip)
    cols["oil_ret"] = oil_ret

    spx = on_grid("spx")
    ndx = on_grid("ndx")
    cols["spx_ret"] = logdiff(spx)
    cols["ndx_ret"] = logdiff(ndx)

    r2 = np.square(cols["spx_ret"])
    rv = np.full(T, np.nan)
    w = cfg.rv_window
    valid = ~np.isnan(r2)
    csum = np.cumsum(np.where(valid, r2, 0.0))
    ccnt = np.cumsum(valid.astype(np.int64))
    for t in range(w, T):
        n = ccnt[t] - ccnt[t - w]
        if n >= w - 2:
            s = csum[t] - csum[t - w]
            rv[t] = 100.0 * np.sqrt(252.0 * s / max(n, 1))
    cols["spx_rv21"] = rv

    for raw_name, feat, periods in MACRO_SPEC:
        series = raw[raw_name]
        if periods is not None:
            series = _yoy(series, periods)
        cols[feat] = _attach_sparse(grid, series)

    panel = pl.DataFrame({"date": grid.astype("datetime64[us]")})
    for f in FEATURES:
        panel = panel.with_columns(pl.Series(f, cols[f]))
    for extra, arr in [("spx_close", spx), ("ndx_close", ndx), ("usd_idx", usd), ("wti", wti)]:
        panel = panel.with_columns(pl.Series(extra, arr))
    panel.write_parquet(cfg.processed_dir / "panel_raw.parquet")

    X_raw = np.column_stack([cols[f] for f in FEATURES])
    X = X_raw.copy()
    for f in LOG_FEATURES:
        j = FEATURES.index(f)
        X[:, j] = np.where(np.isnan(X[:, j]), np.nan, np.log(np.maximum(X[:, j], 1e-4)))

    M = (~np.isnan(X)).astype(np.uint8)
    split_idx = int(T * split_frac)

    kept, dropped = [], []
    for j, f in enumerate(FEATURES):
        n_train = int(M[:split_idx, j].sum())
        if n_train < 50:
            dropped.append(f)
        else:
            kept.append(f)
    missing_yields = [f for f in YIELD_FEATURES if f in dropped]
    if missing_yields:
        raise RuntimeError(
            f"yield features {missing_yields} lack data; the Nelson-Siegel head "
            "needs the full curve. Check data/raw/manifest.json."
        )

    keep_idx = [FEATURES.index(f) for f in kept]
    X, M = X[:, keep_idx], M[:, keep_idx]

    mean = np.zeros(len(kept))
    std = np.zeros(len(kept))
    for j in range(len(kept)):
        v = X[:split_idx, j][M[:split_idx, j] == 1]
        mean[j] = v.mean()
        std[j] = max(v.std(), 1e-8)
    Xs = (X - mean) / std
    Xs[M == 0] = 0.0

    dt = np.zeros(T)
    dt[1:] = (grid[1:] - grid[:-1]).astype("timedelta64[D]").astype(np.float64) / 365.25

    np.savez_compressed(
        cfg.processed_dir / "arrays.npz",
        X=Xs.astype(np.float64),
        M=M,
        dt=dt.astype(np.float64),
        dates=grid.astype("datetime64[D]").astype(np.int64),
    )

    ymean = [float(mean[kept.index(f)]) for f in YIELD_FEATURES]
    ystd = [float(std[kept.index(f)]) for f in YIELD_FEATURES]
    meta = {
        "features": kept,
        "dropped": dropped,
        "log_features": [f for f in LOG_FEATURES if f in kept],
        "mean": mean.tolist(),
        "std": std.tolist(),
        "yield_mean": ymean,
        "yield_std": ystd,
        "n_rows": T,
        "split_idx": split_idx,
        "start": str(grid[0]),
        "end": str(grid[-1]),
        "coverage": {f: float(M[:, kept.index(f)].mean()) for f in kept},
    }
    (cfg.processed_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    return meta
