"""Clients for free/open data sources: FRED, Stooq, Yahoo Finance.

Every fetch returns a two-column polars DataFrame (date, value), sorted by
date, with genuine gaps kept as nulls. ``download_all`` writes one CSV per
series into ``data/raw`` plus a ``manifest.json`` recording provenance, so a
partially blocked network degrades gracefully instead of failing the run.

Sources (all free, no API key):
- FRED CSV endpoint: https://fred.stlouisfed.org/graph/fredgraph.csv?id=<ID>
- Stooq daily CSV:   https://stooq.com/q/d/l/?s=<sym>&i=d
- Yahoo chart API:   https://query1.finance.yahoo.com/v8/finance/chart/<sym>
"""
from __future__ import annotations

import io
import json
import time
from pathlib import Path

import polars as pl
import requests

from ..config import DataConfig

USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)

FRED_SERIES: dict[str, str] = {
    "y_3m": "DGS3MO",
    "y_2y": "DGS2",
    "y_5y": "DGS5",
    "y_10y": "DGS10",
    "y_30y": "DGS30",
    "hy_oas": "BAMLH0A0HYM2",
    "ig_oas": "BAMLC0A0CM",
    "vix": "VIXCLS",
    "ted": "TEDRATE",
    "usd_idx": "DTWEXBGS",
    "wti": "DCOILWTICO",
    "cpi": "CPIAUCSL",
    "indpro": "INDPRO",
    "unrate": "UNRATE",
    "gdp": "GDPC1",
}

EQUITY_SOURCES: dict[str, list[str]] = {
    "spx": ["stooq:^spx", "yahoo:^GSPC", "fred:SP500"],
    "ndx": ["stooq:^ndx", "yahoo:^NDX", "fred:NASDAQCOM"],
}


class FetchError(RuntimeError):
    pass


def _get(url: str, retries: int = 3, timeout: int = 30) -> bytes:
    last: Exception | None = None
    for attempt in range(retries):
        try:
            r = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=timeout)
            if r.status_code == 200 and r.content:
                return r.content
            last = FetchError(f"HTTP {r.status_code} for {url}")
        except requests.RequestException as e:
            last = e
        time.sleep(1.0 + attempt)
    raise FetchError(f"failed after {retries} attempts: {url}") from last


def _two_col(df: pl.DataFrame) -> pl.DataFrame:
    """Normalize any (date-ish, value-ish) frame to (date: pl.Date, value: f64)."""
    df = df.select(df.columns[:2])
    df.columns = ["date", "value"]
    if df.schema["date"] != pl.Date:
        df = df.with_columns(pl.col("date").cast(pl.Utf8).str.slice(0, 10).str.to_date())
    df = df.with_columns(pl.col("value").cast(pl.Float64, strict=False))
    return df.drop_nulls(subset=["date"]).sort("date")


def fetch_fred(series_id: str) -> pl.DataFrame:
    raw = _get(f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}")
    text = raw.decode("utf-8", errors="replace")
    if text.lstrip().startswith("<"):
        raise FetchError(f"FRED returned HTML for {series_id}")
    df = pl.read_csv(io.StringIO(text), null_values=["."], infer_schema_length=0)
    return _two_col(df)


def fetch_stooq(symbol: str) -> pl.DataFrame:
    raw = _get(f"https://stooq.com/q/d/l/?s={symbol}&i=d")
    text = raw.decode("utf-8", errors="replace")
    if text.lstrip().startswith("<") or "Exceeded" in text[:200]:
        raise FetchError(f"Stooq refused {symbol}")
    df = pl.read_csv(io.StringIO(text), infer_schema_length=0)
    if "Close" not in df.columns or df.height < 100:
        raise FetchError(f"Stooq payload malformed for {symbol}")
    return _two_col(df.select(["Date", "Close"]))


def fetch_yahoo(symbol: str) -> pl.DataFrame:
    import urllib.parse

    sym = urllib.parse.quote(symbol)
    url = (
        f"https://query1.finance.yahoo.com/v8/finance/chart/{sym}"
        f"?period1=473385600&period2={int(time.time())}&interval=1d&events=history"
    )
    payload = json.loads(_get(url).decode("utf-8", errors="replace"))
    result = (payload.get("chart") or {}).get("result") or []
    if not result:
        raise FetchError(f"Yahoo returned no result for {symbol}")
    ts = result[0].get("timestamp") or []
    quote = ((result[0].get("indicators") or {}).get("quote") or [{}])[0]
    close = quote.get("close") or []
    if not ts or not close or len(ts) != len(close):
        raise FetchError(f"Yahoo payload malformed for {symbol}")
    df = pl.DataFrame({"epoch": ts, "value": close}).with_columns(
        pl.from_epoch("epoch", time_unit="s").dt.date().alias("date")
    )
    return _two_col(df.select(["date", "value"]))


def _fetch_equity(chain: list[str]) -> tuple[pl.DataFrame, str]:
    errors: list[str] = []
    for spec in chain:
        provider, sym = spec.split(":", 1)
        try:
            if provider == "stooq":
                return fetch_stooq(sym), spec
            if provider == "yahoo":
                return fetch_yahoo(sym), spec
            if provider == "fred":
                return fetch_fred(sym), spec
        except Exception as e:
            errors.append(f"{spec}: {e}")
        time.sleep(0.3)
    raise FetchError(" | ".join(errors))


def download_all(cfg: DataConfig) -> dict:
    """Fetch every series to data/raw/<name>.csv; returns the manifest."""
    cfg.raw_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, dict] = {}

    for name, fred_id in FRED_SERIES.items():
        try:
            df = fetch_fred(fred_id)
            df.write_csv(cfg.raw_dir / f"{name}.csv")
            manifest[name] = {
                "source": f"fred:{fred_id}",
                "rows": df.height,
                "start": str(df["date"].min()),
                "end": str(df["date"].max()),
                "status": "ok",
            }
        except Exception as e:
            manifest[name] = {"source": f"fred:{fred_id}", "status": f"failed: {e}"}
        time.sleep(0.3)

    for name, chain in EQUITY_SOURCES.items():
        try:
            df, used = _fetch_equity(chain)
            df.write_csv(cfg.raw_dir / f"{name}.csv")
            manifest[name] = {
                "source": used,
                "rows": df.height,
                "start": str(df["date"].min()),
                "end": str(df["date"].max()),
                "status": "ok",
            }
        except Exception as e:
            manifest[name] = {"source": "|".join(chain), "status": f"failed: {e}"}

    (cfg.raw_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def load_raw(cfg: DataConfig, name: str) -> pl.DataFrame:
    """Read a previously downloaded series; empty frame if absent."""
    path = cfg.raw_dir / f"{name}.csv"
    if not path.exists():
        return pl.DataFrame({"date": [], "value": []}, schema={"date": pl.Date, "value": pl.Float64})
    return _two_col(pl.read_csv(path, infer_schema_length=0))
