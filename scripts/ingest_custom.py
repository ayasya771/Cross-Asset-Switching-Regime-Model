#!/usr/bin/env python3
"""Ingest a user-supplied wide CSV into the processed panel.

Accepts the same flexible layout as the browser upload (a ``date`` column
plus any of the 18 model features, raw closes, or macro levels under common
aliases), derives returns / realized-vol / YoY exactly like
``regimelab.data.preprocess``, and writes ``panel_raw.parquet``,
``arrays.npz`` and ``meta.json`` into ``data/processed``.

The 18-feature schema is always preserved (absent series stay masked), so the
shipped checkpoint can filter the custom panel without retraining. Run
``train.py`` afterwards only if you want a model fitted to the new data.

Usage:
    python scripts/ingest_custom.py path/to/my_data.csv [--start ...] [--end ...]
    python scripts/ingest_custom.py --template   # print the minimal template
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from regimelab.config import (
    DATA_PROCESSED,
    DATA_RAW,
    FEATURES,
    LOG_FEATURES,
    YIELD_FEATURES,
    DataConfig,
    ensure_dirs,
)

ALIASES: dict[str, list[str]] = {
    "date": ["date", "ds", "time", "timestamp", "datetime", "day", "trading_date", "obs_date"],
    "y_3m": ["y_3m", "dgs3mo", "3m", "3mo", "us3m", "y3m"],
    "y_2y": ["y_2y", "dgs2", "2y", "2yr", "us2y", "y2y"],
    "y_5y": ["y_5y", "dgs5", "5y", "5yr", "us5y", "y5y"],
    "y_10y": ["y_10y", "dgs10", "10y", "10yr", "us10y", "y10y"],
    "y_30y": ["y_30y", "dgs30", "30y", "30yr", "us30y", "y30y"],
    "hy_oas": ["hy_oas", "hy", "bamlh0a0hym2", "high_yield", "hy_spread", "hyoas"],
    "ig_oas": ["ig_oas", "ig", "bamlc0a0cm", "ig_spread", "igoas", "investment_grade"],
    "vix": ["vix", "vixcls", "volatility"],
    "ted": ["ted", "tedrate", "ted_spread", "tedspread"],
    "usd_ret": ["usd_ret", "dollar_ret", "dxy_ret"],
    "oil_ret": ["oil_ret", "wti_ret", "crude_ret"],
    "spx_ret": ["spx_ret", "sp500_ret", "equity_ret"],
    "spx_rv21": ["spx_rv21", "rv21", "rv", "realized_vol", "realised_vol", "spx_rv"],
    "ndx_ret": ["ndx_ret", "nasdaq_ret", "ndxret"],
    "cpi_yoy": ["cpi_yoy"],
    "indpro_yoy": ["indpro_yoy"],
    "unrate": ["unrate", "unemployment", "unemp"],
    "gdp_yoy": ["gdp_yoy"],
    "spx_close": ["spx_close", "spx", "^spx", "^gspc", "sp500", "sp500_close",
                  "close_spx", "spxclose", "equity_close"],
    "ndx_close": ["ndx_close", "ndx", "^ndx", "nasdaq", "nasdaq_close", "ndxclose"],
    "usd_idx": ["usd_idx", "usd", "dxy", "dtwexbgs", "dollar_idx", "broad_usd"],
    "wti": ["wti", "oil", "crude", "dcoilwtico", "wti_close", "oil_close"],
    "cpi": ["cpi", "cpiaucsl", "inflation"],
    "indpro": ["indpro", "industrial_production"],
    "gdp": ["gdp", "gdpc1"],
}
CANON = {a.lower(): c for c, names in ALIASES.items() for a in names}
MISSING = {"", "nan", "na", "n/a", "nat", "null", "none", "nil", ".", "-",
           "--", "---", "#n/a", "#na", "?"}


def normalize_header(name: str) -> str | None:
    k = (name or "").strip().lower()
    if k in CANON:
        return CANON[k]
    cleaned = k.replace("fred:", "").replace("stooq:", "").replace("yahoo:", "")
    cleaned = "".join(ch for ch in cleaned if ch.isalnum() or ch == "_")
    return CANON.get(cleaned)


def parse_num(cell) -> float:
    if cell is None:
        return np.nan
    t = str(cell).strip().lower().replace(",", "")
    if t in MISSING:
        return np.nan
    try:
        v = float(t)
    except ValueError:
        return np.nan
    return v if np.isfinite(v) else np.nan


def parse_date(cell) -> date | None:
    import re

    if cell is None:
        return None
    t = str(cell).strip()
    if not t:
        return None
    m = re.match(r"^(\d{4})-(\d{1,2})-(\d{1,2})", t)
    if m:
        return date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
    m = re.match(r"^(\d{1,2})/(\d{1,2})/(\d{2,4})", t)
    if m:
        yy = int(m.group(3))
        yy += 2000 if yy < 50 else (1900 if yy < 100 else 0)
        return date(yy, int(m.group(1)), int(m.group(2)))
    m = re.match(r"^(\d{4})/(\d{1,2})/(\d{1,2})", t)
    if m:
        return date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
    try:
        from datetime import datetime
        return datetime.fromisoformat(t[:10]).date()
    except ValueError:
        return None


def _logdiff(levels: np.ndarray) -> np.ndarray:
    out = np.full_like(levels, np.nan, dtype=np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        lx = np.where(levels > 0, np.log(np.maximum(levels, 1e-12)), np.nan)
    out[1:] = lx[1:] - lx[:-1]
    return out


def _oil_ret(wti: np.ndarray, clip: float = 0.5) -> np.ndarray:
    out = np.full_like(wti, np.nan, dtype=np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        r = (wti[1:] - wti[:-1]) / np.maximum(np.abs(wti[:-1]), 5.0)
    out[1:] = np.clip(r, -clip, clip)
    return out


def _rv21(rets: np.ndarray, window: int = 21) -> np.ndarray:
    out = np.full_like(rets, np.nan, dtype=np.float64)
    r2 = np.square(rets)
    valid = ~np.isnan(r2)
    csum = np.cumsum(np.where(valid, r2, 0.0))
    ccnt = np.cumsum(valid.astype(np.int64))
    for t in range(window, len(rets)):
        n = ccnt[t] - ccnt[t - window]
        if n >= window - 2:
            out[t] = 100.0 * np.sqrt(252.0 * (csum[t] - csum[t - window]) / max(n, 1))
    return out


def _yoy_from_levels(dates: np.ndarray, levels: np.ndarray) -> np.ndarray:
    out = np.full(len(dates), np.nan)
    day = np.timedelta64(1, "D")
    for t in range(len(dates)):
        if not np.isfinite(levels[t]):
            continue
        best, best_err = -1, np.inf
        for k in range(t - 1, -1, -1):
            if not np.isfinite(levels[k]):
                continue
            gap = (dates[t] - dates[k]) / day
            err = abs(gap - 365.0)
            if err < best_err:
                best_err, best = err, k
            if gap > 425:
                break
        if best >= 0 and best_err <= 60 and levels[best] != 0:
            out[t] = 100.0 * (levels[t] / levels[best] - 1.0)
    return out


TEMPLATE = (
    "date,y_3m,y_2y,y_5y,y_10y,y_30y,hy_oas,ig_oas,vix,ted,"
    "usd_ret,oil_ret,spx_ret,spx_rv21,ndx_ret,cpi_yoy,indpro_yoy,unrate,gdp_yoy,"
    "spx_close,ndx_close,usd_idx,wti\n"
    "2024-01-02,0.15,0.45,0.60,0.85,1.10,4.20,1.35,16.5,0.28,"
    "0.0008,0.0042,0.0006,15.2,0.0009,3.1,1.8,3.7,2.4,"
    "4783.83,16833.38,101.2,73.5\n"
    "2024-01-03,0.16,0.47,0.62,0.87,1.12,4.25,1.37,17.1,0.29,"
    "-0.0004,-0.0061,-0.0012,15.4,-0.0018,,,,,"
    "4778.12,16806.45,101.1,72.9\n"
)


def read_wide_csv(path: Path) -> tuple[list[date], dict[str, np.ndarray], list[str]]:
    """Parse the CSV into sorted, deduped dates + canonical float columns."""
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        head = next(reader)
        rows = [r for r in reader if any((c or "").strip() for c in r)]
    canon = [normalize_header(h) for h in head]
    if "date" not in canon:
        raise SystemExit("no date column found: name one column 'date' (YYYY-MM-DD)")
    date_ix = canon.index("date")
    recs = []
    for r in rows:
        d = parse_date(r[date_ix] if date_ix < len(r) else None)
        if d is None:
            continue
        recs.append((d, r))
    if not recs:
        raise SystemExit("no parseable dates: use YYYY-MM-DD, one row per observation")
    recs.sort(key=lambda t: t[0])
    dedup: list[tuple[date, list[str]]] = []
    for d, r in recs:
        if dedup and dedup[-1][0] == d:
            dedup[-1] = (d, r)
        else:
            dedup.append((d, r))
    dates = [d for d, _ in dedup]
    cols: dict[str, np.ndarray] = {}
    seen: set[str] = set()
    for i, c in enumerate(canon):
        if not c or c == "date" or c in seen:
            continue
        seen.add(c)
        cols[c] = np.array(
            [parse_num(r[i] if i < len(r) else None) for _, r in dedup],
            dtype=np.float64,
        )
    return dates, cols, [str(d) for d in dates]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("csv", nargs="?", help="wide CSV to ingest")
    ap.add_argument("--template", action="store_true", help="print the CSV template")
    ap.add_argument("--start", default=None)
    ap.add_argument("--end", default=None)
    ap.add_argument("--split-frac", type=float, default=0.85)
    ap.add_argument("--scaler", choices=["recompute", "shipped"], default="recompute",
                    help="recompute normalization from the upload (default) or reuse shipped stats")
    args = ap.parse_args()

    if args.template or not args.csv:
        if not args.csv:
            print(TEMPLATE, end="")
            return
        print(TEMPLATE, end="")
        return

    DATA_RAW.mkdir(parents=True, exist_ok=True)
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    cfg = DataConfig(raw_dir=DATA_RAW, processed_dir=DATA_PROCESSED)
    dates, cols, _ = read_wide_csv(Path(args.csv))
    grid = np.array(dates, dtype="datetime64[D]")
    if args.start:
        grid = grid[grid >= np.datetime64(args.start, "D")]
    if args.end:
        grid = grid[grid <= np.datetime64(args.end, "D")]
    keep = np.isin(np.array(dates, dtype="datetime64[D]"), grid)
    T = int(keep.sum())
    if T < 60:
        raise SystemExit(f"only {T} dated rows after filtering; need at least 60")
    cols = {k: v[keep] for k, v in cols.items()}
    dates = np.array(dates, dtype="datetime64[D]")[keep]

    feats: dict[str, np.ndarray] = {
        f: cols.get(f, np.full(T, np.nan)).astype(np.float64) for f in FEATURES
    }
    closes = {k: cols.get(k, np.full(T, np.nan)) for k in
              ("spx_close", "ndx_close", "usd_idx", "wti")}

    def fill_missing(feat: str, derived: np.ndarray, label: str) -> None:
        miss = ~np.isfinite(feats[feat])
        use = miss & np.isfinite(derived)
        if use.any():
            feats[feat][use] = derived[use]
            print(f"  derived {feat} from {label}: {int(use.sum())} rows")

    fill_missing("spx_ret", _logdiff(closes["spx_close"]), "equity closes")
    fill_missing("ndx_ret", _logdiff(closes["ndx_close"]), "Nasdaq closes")
    fill_missing("usd_ret", _logdiff(closes["usd_idx"]), "USD index levels")
    fill_missing("oil_ret", _oil_ret(closes["wti"], cfg.oil_ret_clip), "WTI levels")
    fill_missing("spx_rv21", _rv21(feats["spx_ret"], cfg.rv_window), "equity returns")
    for yoy, lvl in (("cpi_yoy", "cpi"), ("indpro_yoy", "indpro"), ("gdp_yoy", "gdp")):
        if lvl in cols:
            fill_missing(yoy, _yoy_from_levels(dates, cols[lvl]), "uploaded levels")

    observed = [f for f in FEATURES if np.isfinite(feats[f]).any()]
    if len(observed) < 3:
        raise SystemExit(
            f"only {len(observed)} of the 18 model series recognized; "
            "see --template for accepted column names")
    for f in FEATURES:
        if not np.isfinite(feats[f]).any():
            print(f"  warning: {f} not observed; masked throughout")

    shipped_mean, shipped_std = None, None
    meta_path = cfg.processed_dir / "meta.json"
    if meta_path.exists():
        old = json.loads(meta_path.read_text())
        if old.get("features") == FEATURES and len(old.get("mean", [])) == len(FEATURES):
            shipped_mean = np.array(old["mean"], dtype=np.float64)
            shipped_std = np.array(old["std"], dtype=np.float64)

    X_raw = np.column_stack([feats[f] for f in FEATURES])
    X = X_raw.copy()
    for f in LOG_FEATURES:
        j = FEATURES.index(f)
        X[:, j] = np.where(np.isnan(X[:, j]), np.nan,
                           np.log(np.maximum(X[:, j], 1e-4)))
    M = (~np.isnan(X)).astype(np.uint8)
    split_idx = max(1, int(T * args.split_frac))
    mean = np.zeros(len(FEATURES))
    std = np.zeros(len(FEATURES))
    for j in range(len(FEATURES)):
        v = X[:split_idx, j][M[:split_idx, j] == 1]
        if args.scaler == "shipped" or len(v) < 50:
            if shipped_mean is not None:
                mean[j], std[j] = float(shipped_mean[j]), float(shipped_std[j])
                if args.scaler != "shipped" and len(v) > 0:
                    print(f"  warning: {FEATURES[j]} has {len(v)} train rows; "
                          "using shipped normalization")
            elif len(v):
                mean[j], std[j] = float(v.mean()), max(float(v.std()), 1e-8)
            else:
                mean[j], std[j] = 0.0, 1.0
        else:
            mean[j], std[j] = float(v.mean()), max(float(v.std()), 1e-8)
    Xs = (X - mean) / std
    Xs[M == 0] = 0.0

    dt = np.zeros(T)
    dt[1:] = (dates[1:] - dates[:-1]).astype("timedelta64[D]").astype(np.float64) / 365.25

    panel = pl.DataFrame({"date": dates.astype("datetime64[us]")})
    for f in FEATURES:
        panel = panel.with_columns(pl.Series(f, feats[f]))
    for extra, arr in [("spx_close", closes["spx_close"]), ("ndx_close", closes["ndx_close"]),
                       ("usd_idx", closes["usd_idx"]), ("wti", closes["wti"])]:
        panel = panel.with_columns(pl.Series(extra, arr))
    panel.write_parquet(cfg.processed_dir / "panel_raw.parquet")
    np.savez_compressed(cfg.processed_dir / "arrays.npz",
                        X=Xs.astype(np.float64), M=M,
                        dt=dt.astype(np.float64),
                        dates=dates.astype("datetime64[D]").astype(np.int64))
    ymean = [float(mean[FEATURES.index(f)]) for f in YIELD_FEATURES]
    ystd = [float(std[FEATURES.index(f)]) for f in YIELD_FEATURES]
    meta = {
        "features": FEATURES,
        "dropped": [],
        "log_features": [f for f in LOG_FEATURES if f in FEATURES],
        "mean": mean.tolist(),
        "std": std.tolist(),
        "yield_mean": ymean,
        "yield_std": ystd,
        "n_rows": T,
        "split_idx": split_idx,
        "start": str(dates[0]),
        "end": str(dates[-1]),
        "coverage": {f: float(M[:, FEATURES.index(f)].mean()) for f in FEATURES},
        "source": f"custom:{Path(args.csv).name}",
    }
    (cfg.processed_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    (cfg.processed_dir / "custom_source.json").write_text(json.dumps(
        {"file": str(Path(args.csv).name), "rows": T,
         "start": str(dates[0]), "end": str(dates[-1]),
         "observed": observed}, indent=2))
    print(f"panel: {T} rows  {dates[0]} .. {dates[-1]}  ({len(observed)}/18 series)")
    print("next: python scripts/train.py  (retrain)  or  python scripts/extract_regimes.py  "
          "(filter with the shipped checkpoint)")


if __name__ == "__main__":
    main()
