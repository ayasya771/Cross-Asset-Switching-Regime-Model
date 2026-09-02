"""Alignment, masking, transforms, and dt computation on synthetic raw files."""
from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest

from regimelab.config import DataConfig, YIELD_FEATURES
from regimelab.data.preprocess import build_panel


def _bdays(start: date, n: int) -> list[date]:
    out, d = [], start
    while len(out) < n:
        if d.weekday() < 5:
            out.append(d)
        d += timedelta(days=1)
    return out


def _write(dirpath, name, dates, values):
    pl.DataFrame({"date": dates, "value": values}).write_csv(dirpath / f"{name}.csv")


@pytest.fixture()
def raw_dir(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    days = _bdays(date(2020, 1, 6), 220)
    n = len(days)
    rng = np.random.default_rng(0)

    for i, f in enumerate(YIELD_FEATURES):
        _write(raw, f, days, 2.0 + 0.1 * i + 0.05 * rng.standard_normal(n))
    _write(raw, "hy_oas", days, 4.0 + 0.2 * rng.standard_normal(n))
    _write(raw, "ig_oas", days, 1.2 + 0.05 * rng.standard_normal(n))
    _write(raw, "vix", days, 18.0 + np.abs(rng.standard_normal(n)) * 4)
    _write(raw, "ted", days[: n // 2], 0.3 + 0.05 * rng.standard_normal(n // 2))
    _write(raw, "usd_idx", days, 110.0 + np.cumsum(0.1 * rng.standard_normal(n)))
    _write(raw, "wti", days, 60.0 + np.cumsum(0.5 * rng.standard_normal(n)))
    _write(raw, "spx", days, 3200.0 * np.exp(np.cumsum(0.01 * rng.standard_normal(n))))
    _write(raw, "ndx", days, 8800.0 * np.exp(np.cumsum(0.012 * rng.standard_normal(n))))

    months = sorted(set(date(2018 + m // 12, m % 12 + 1, 1) for m in range(36)))
    _write(raw, "cpi", months, np.linspace(250, 270, len(months)))
    _write(raw, "indpro", months, np.linspace(100, 104, len(months)))
    _write(raw, "unrate", months, np.linspace(3.5, 4.4, len(months)))
    quarters = sorted(set(date(2018 + q // 4, (q % 4) * 3 + 1, 1) for q in range(10)))
    _write(raw, "gdp", quarters, np.linspace(18000, 19000, len(quarters)))
    return tmp_path


def test_build_panel(raw_dir):
    cfg = DataConfig(
        start="2020-01-01",
        end="2020-12-31",
        raw_dir=raw_dir / "raw",
        processed_dir=raw_dir / "processed",
    )
    meta = build_panel(cfg, split_frac=0.85)
    z = np.load(raw_dir / "processed" / "arrays.npz")
    X, M, dt = z["X"], z["M"], z["dt"]
    feats = meta["features"]

    assert feats[:5] == YIELD_FEATURES
    for f in ("cpi_yoy", "gdp_yoy"):
        assert f in meta["dropped"]

    T = meta["n_rows"]
    assert X.shape == (T, len(feats)) and M.shape == X.shape

    j_ted = feats.index("ted")
    assert M[:20, j_ted].mean() > 0.8
    assert M[-20:, j_ted].sum() == 0
    assert np.all(X[M == 0] == 0.0)

    j_ret = feats.index("spx_ret")
    assert M[0, j_ret] == 0 and M[1, j_ret] == 1

    j_rv = feats.index("spx_rv21")
    assert M[: cfg.rv_window, j_rv].sum() == 0
    assert M[cfg.rv_window + 2, j_rv] == 1

    assert dt[0] == 0.0
    uniq = np.unique(np.round(dt[1:] * 365.25).astype(int))
    assert set(uniq.tolist()) <= {1, 2, 3, 4}
    assert 3 in uniq

    j_vix = feats.index("vix")
    cut = meta["split_idx"]
    v = X[:cut, j_vix][M[:cut, j_vix] == 1]
    assert abs(v.mean()) < 1e-8
    assert abs(v.std() - 1.0) < 1e-6

    panel = pl.read_parquet(raw_dir / "processed" / "panel_raw.parquet")
    raw_vix = panel["vix"].to_numpy()
    mean = meta["mean"][j_vix]
    std = meta["std"][j_vix]
    k = 10
    assert M[k, j_vix] == 1
    assert abs(np.exp(X[k, j_vix] * std + mean) - raw_vix[k]) < 1e-8
