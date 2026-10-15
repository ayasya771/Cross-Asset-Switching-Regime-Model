"""Custom-CSV ingestion: the upload pipeline applied to user files."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.ingest_custom import read_wide_csv  # noqa: E402
from scripts import ingest_custom as ing  # noqa: E402


def _write_csv(path: Path, header: list[str], rows: list[list]) -> None:
    with open(path, "w", newline="") as f:
        w = __import__("csv").writer(f)
        w.writerow(header)
        w.writerows(rows)


def test_aliases_and_missing_tokens(tmp_path):
    p = tmp_path / "u.csv"
    _write_csv(p,
               ["Date", "DGS10", "VIXCLS", "^GSPC", "CPIAUCSL", "TEDRATE"],
               [["2024-01-02", "4.0", "15.0", "4700", "3.2", "0.25"],
                ["2024-01-03", ".", "NA", "4710", "null", "-"],
                ["2024-01-04", "4.1", "16.0", "4690", "3.3", "0.26"]])
    dates, cols, _ = read_wide_csv(p)
    assert [str(d) for d in dates] == ["2024-01-02", "2024-01-03", "2024-01-04"]
    assert cols["y_10y"][0] == 4.0 and np.isnan(cols["y_10y"][1])
    assert np.isnan(cols["vix"][1]) and np.isnan(cols["ted"][1])
    assert cols["spx_close"][2] == 4690
    assert cols["cpi"][0] == 3.2 and np.isnan(cols["cpi"][1])


def test_date_formats_and_dedupe(tmp_path):
    p = tmp_path / "u.csv"
    _write_csv(p, ["date", "vix"],
               [["01/02/2024", "15"], ["2024-01-02", "16"], ["2024/01/03", "17"]])
    dates, cols, _ = read_wide_csv(p)
    # last row wins for the duplicated 2024-01-02
    assert [str(d) for d in dates] == ["2024-01-02", "2024-01-03"]
    assert cols["vix"][0] == 16.0


def test_end_to_end_ingest_builds_panel(tmp_path, monkeypatch):
    from regimelab.config import FEATURES
    import scripts.ingest_custom as m

    dates = [f"2023-01-{d:02d}" for d in range(2, 32)]  # 30 days is too few...
    # ...so build 120 business-day-like rows instead.
    import datetime
    start = datetime.date(2023, 1, 2)
    days, d = [], start
    while len(days) < 120:
        if d.weekday() < 5:
            days.append(d.isoformat())
        d += datetime.timedelta(days=1)
    rng = np.random.default_rng(0)
    spx = 4700 + np.cumsum(rng.standard_normal(len(days)) * 20)
    vix = 15 + rng.standard_normal(len(days))
    p = tmp_path / "panel.csv"
    _write_csv(p, ["date", "y_10y", "vix", "spx_close"],
               [[days[i], 4.0, round(float(vix[i]), 2), round(float(spx[i]), 2)]
                for i in range(len(days))])
    monkeypatch.setattr(m, "DATA_PROCESSED", tmp_path / "processed")
    monkeypatch.setattr(m, "DATA_RAW", tmp_path / "raw")
    (tmp_path / "processed").mkdir()
    (tmp_path / "raw").mkdir()
    sys.argv = ["ingest_custom.py", str(p)]
    m.main()
    meta = json.loads((tmp_path / "processed" / "meta.json").read_text())
    assert meta["features"] == FEATURES and meta["n_rows"] == len(days)
    assert meta["source"].startswith("custom:")
    arrs = np.load(tmp_path / "processed" / "arrays.npz")
    assert arrs["X"].shape == (len(days), len(FEATURES))
    # spx_ret derived from closes: row 0 masked, row 1 observed
    j = FEATURES.index("spx_ret")
    assert arrs["M"][0, j] == 0 and arrs["M"][1, j] == 1
    panel = pl.read_parquet(tmp_path / "processed" / "panel_raw.parquet")
    assert panel.height == len(days)
