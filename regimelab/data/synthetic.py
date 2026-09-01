"""Calibrated synthetic market generator with known regime ground truth.

Purpose: (1) prove the full pipeline end to end in environments where the free
data endpoints are unreachable, and (2) validate the architecture. The
generator embeds a hidden 4-state Markov regime chain, so extracted regimes
can be scored against the truth (adjusted Rand index) instead of eyeballed.

It writes exactly the same per-series CSVs as ``sources.download_all`` (plus
``ground_truth_regimes.csv``), so preprocessing, training, and extraction run
unchanged on synthetic or real data.

Regimes:
    0  Calm expansion:      low vol, tight spreads, upward curve, +drift
    1  Late-cycle squeeze:  flat/inverted curve, rising short rates
    2  Crisis:              vol/spread spikes, curve bull-steepens, -drift
    3  Reflation/recovery:  steep curve, falling spreads, strong +drift
"""
from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import polars as pl

from ..config import DataConfig

REGIME_NAMES = ["calm_expansion", "late_cycle", "crisis", "reflation_recovery"]

TRANSITION = np.array(
    [
        [0.9990, 0.0007, 0.0002, 0.0001],
        [0.0010, 0.9975, 0.0013, 0.0002],
        [0.0002, 0.0003, 0.9950, 0.0045],
        [0.0025, 0.0002, 0.0003, 0.9970],
    ]
)

REGIME_PARAMS = {
    0: dict(lvl=4.2, slp=-1.6, crv=0.4, lvix=np.log(14), hy=3.5, ig=1.1, mu=0.12, vol=0.11, usd=0.00),
    1: dict(lvl=5.0, slp=0.6, crv=-0.6, lvix=np.log(19), hy=4.6, ig=1.5, mu=0.04, vol=0.16, usd=0.03),
    2: dict(lvl=3.2, slp=-2.6, crv=1.0, lvix=np.log(38), hy=12.0, ig=3.4, mu=-0.55, vol=0.42, usd=0.06),
    3: dict(lvl=3.6, slp=-3.0, crv=0.8, lvix=np.log(21), hy=6.0, ig=1.8, mu=0.30, vol=0.20, usd=-0.04),
}


def _business_days(start: date, end: date) -> list[date]:
    out, d = [], start
    while d <= end:
        if d.weekday() < 5:
            out.append(d)
        d += timedelta(days=1)
    return out


def generate(cfg: DataConfig, start: str = "1997-01-01", end: str = "2026-08-21",
             seed: int = 11) -> dict:
    """Write synthetic raw CSVs into cfg.raw_dir; returns a summary dict."""
    rng = np.random.default_rng(seed)
    days = _business_days(date.fromisoformat(start), date.fromisoformat(end))
    T = len(days)

    reg = np.zeros(T, dtype=int)
    for t in range(1, T):
        reg[t] = rng.choice(4, p=TRANSITION[reg[t - 1]])

    dt = 1.0 / 252.0
    sq = np.sqrt(dt)

    lvl = np.zeros(T); slp = np.zeros(T); crv = np.zeros(T)
    lvix = np.zeros(T); hy = np.zeros(T); ig = np.zeros(T)
    lvl[0], slp[0], crv[0] = 4.5, -1.5, 0.3
    lvix[0], hy[0], ig[0] = np.log(16), 4.0, 1.2
    spx = np.zeros(T); ndx = np.zeros(T); usd = np.zeros(T); oil = np.zeros(T)
    spx[0], ndx[0], usd[0], oil[0] = np.log(800.0), np.log(1000.0), np.log(100.0), np.log(25.0)

    for t in range(1, T):
        p = REGIME_PARAMS[reg[t]]
        z = rng.standard_normal(9)
        lvl[t] = lvl[t-1] + 1.5 * (p["lvl"] - lvl[t-1]) * dt + 0.55 * sq * z[0]
        slp[t] = slp[t-1] + 3.0 * (p["slp"] - slp[t-1]) * dt + 0.80 * sq * z[1]
        crv[t] = crv[t-1] + 4.0 * (p["crv"] - crv[t-1]) * dt + 1.00 * sq * z[2]
        lvix[t] = lvix[t-1] + 12.0 * (p["lvix"] - lvix[t-1]) * dt + 1.10 * sq * z[3]
        hy[t] = max(1.5, hy[t-1] + 6.0 * (p["hy"] - hy[t-1]) * dt + 2.2 * sq * z[4])
        ig[t] = max(0.4, ig[t-1] + 6.0 * (p["ig"] - ig[t-1]) * dt + 0.55 * sq * z[5])
        vol_t = p["vol"] * (0.7 + 0.3 * np.exp(lvix[t] - np.log(20)) / 1.5)
        eq_shock = -0.7 * z[3] + np.sqrt(1 - 0.7**2) * z[6]
        r_spx = p["mu"] * dt + vol_t * sq * eq_shock
        spx[t] = spx[t-1] + r_spx
        ndx[t] = ndx[t-1] + 1.15 * r_spx + 0.006 * sq * z[7]
        usd[t] = usd[t-1] + p["usd"] * dt + 0.07 * sq * z[8]
        oil[t] = oil[t-1] + (0.02 if reg[t] in (0, 3) else -0.10) * dt \
            + 0.35 * sq * (0.3 * eq_shock + 0.95 * rng.standard_normal())

    lam = 0.6
    taus = np.array([0.25, 2.0, 5.0, 10.0, 30.0])
    x = lam * taus
    l1 = (1 - np.exp(-x)) / x
    l2 = l1 - np.exp(-x)
    curves = (
        lvl[:, None]
        + slp[:, None] * l1[None, :]
        + crv[:, None] * l2[None, :]
        + 0.02 * rng.standard_normal((T, 5))
    )
    curves = np.maximum(curves, 0.01)

    ted = np.clip(0.25 + 0.35 * (hy - 3.5) / 8.0 + 0.05 * rng.standard_normal(T), 0.05, None)
    vix = np.exp(lvix)

    cfg.raw_dir.mkdir(parents=True, exist_ok=True)

    def w(name: str, ds: list[date], vals: np.ndarray) -> None:
        pl.DataFrame({"date": ds, "value": vals}).write_csv(cfg.raw_dir / f"{name}.csv")

    names = ["y_3m", "y_2y", "y_5y", "y_10y", "y_30y"]
    for j, nm in enumerate(names):
        w(nm, days, np.round(curves[:, j], 2))
    w("hy_oas", days, np.round(hy, 2))
    w("ig_oas", days, np.round(ig, 2))
    w("vix", days, np.round(vix, 2))
    w("spx", days, np.round(np.exp(spx), 2))
    w("ndx", days, np.round(np.exp(ndx), 2))
    w("wti", days, np.round(np.exp(oil), 2))

    ted_mask = np.array([d < date(2022, 1, 28) for d in days])
    w("ted", [d for d, k in zip(days, ted_mask) if k], np.round(ted[ted_mask], 2))
    usd_mask = np.array([d >= date(2006, 1, 2) for d in days])
    w("usd_idx", [d for d, k in zip(days, usd_mask) if k], np.round(np.exp(usd[usd_mask]), 4))

    month_firsts, seen = [], set()
    for d in days:
        key = (d.year, d.month)
        if key not in seen:
            seen.add(key)
            month_firsts.append(d)
    n_m = len(month_firsts)
    day_ix = {d: i for i, d in enumerate(days)}
    m_ix = np.array([day_ix[d] for d in month_firsts])
    cpi_yoy = np.zeros(n_m); ind_yoy = np.zeros(n_m); unr = np.zeros(n_m)
    cpi_yoy[0], ind_yoy[0], unr[0] = 2.5, 2.0, 5.0
    for i in range(1, n_m):
        r = reg[m_ix[i]]
        cpi_t = {0: 2.3, 1: 3.8, 2: 1.0, 3: 3.0}[r]
        ind_t = {0: 3.0, 1: 1.5, 2: -6.0, 3: 5.0}[r]
        unr_t = {0: 4.2, 1: 3.8, 2: 8.5, 3: 6.0}[r]
        cpi_yoy[i] = cpi_yoy[i-1] + 0.10 * (cpi_t - cpi_yoy[i-1]) + 0.15 * rng.standard_normal()
        ind_yoy[i] = ind_yoy[i-1] + 0.25 * (ind_t - ind_yoy[i-1]) + 0.40 * rng.standard_normal()
        unr[i] = max(2.5, unr[i-1] + 0.15 * (unr_t - unr[i-1]) + 0.10 * rng.standard_normal())
    cpi_level = 100.0 * np.cumprod((1 + cpi_yoy / 100.0) ** (1 / 12))
    ind_level = 100.0 * np.cumprod((1 + ind_yoy / 100.0) ** (1 / 12))
    w("cpi", month_firsts, np.round(cpi_level, 3))
    w("indpro", month_firsts, np.round(ind_level, 3))
    w("unrate", month_firsts, np.round(unr, 1))

    q_firsts = [d for d in month_firsts if d.month in (1, 4, 7, 10)]
    q_ix = np.array([day_ix[d] for d in q_firsts])
    gdp_yoy = np.zeros(len(q_firsts)); gdp_yoy[0] = 2.5
    for i in range(1, len(q_firsts)):
        tgt = {0: 2.8, 1: 1.8, 2: -3.0, 3: 4.0}[reg[q_ix[i]]]
        gdp_yoy[i] = gdp_yoy[i-1] + 0.35 * (tgt - gdp_yoy[i-1]) + 0.30 * rng.standard_normal()
    gdp_level = 10000.0 * np.cumprod((1 + gdp_yoy / 100.0) ** (1 / 4))
    w("gdp", q_firsts, np.round(gdp_level, 1))

    pl.DataFrame(
        {"date": days, "regime": reg, "regime_name": [REGIME_NAMES[r] for r in reg]}
    ).write_csv(cfg.raw_dir / "ground_truth_regimes.csv")

    counts = {REGIME_NAMES[k]: int((reg == k).sum()) for k in range(4)}
    manifest = {nm: {"source": "synthetic", "status": "ok"} for nm in
                names + ["hy_oas", "ig_oas", "vix", "ted", "usd_idx", "wti",
                         "spx", "ndx", "cpi", "indpro", "unrate", "gdp"]}
    import json
    (cfg.raw_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return {"days": T, "regime_days": counts, "start": start, "end": end}
