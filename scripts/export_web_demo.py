#!/usr/bin/env python3
"""Export the trained model, the dataset and the reference results into
``docs/data/`` so the static GitHub Pages demo can run the whole inference
pipeline client-side.

Three artifacts:
- ``panel.csv``      the raw observation panel, empty cells where a series did
                     not print. The browser applies the documented transforms
                     itself, so the demo exercises the real preprocessing path.
- ``scaler.json``    feature order, log-transform list, train-region mean and
                     standard deviation, yield maturities.
- ``model.json``     every weight and buffer as base64 float64, plus shapes.
- ``reference.csv``  labels from the Python run, so the page can report how
                     closely the browser filter reproduces them.
"""
from __future__ import annotations

import base64
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import polars as pl
import torch

from regimelab.config import (
    CHECKPOINTS,
    DATA_PROCESSED,
    LOG_FEATURES,
    PROJECT_ROOT,
    REGIMES,
    YIELD_MATURITIES,
)

WEB_DATA = PROJECT_ROOT / "docs" / "data"
DISPLAY_COLUMNS = ["spx_close", "ndx_close", "usd_idx", "wti"]


def b64(arr: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(arr, dtype=np.float64).tobytes()).decode()


def main() -> None:
    WEB_DATA.mkdir(parents=True, exist_ok=True)
    meta = json.loads((DATA_PROCESSED / "meta.json").read_text())
    summary = json.loads((REGIMES / "summary.json").read_text())
    panel = pl.read_parquet(DATA_PROCESSED / "panel_raw.parquet").with_columns(
        pl.col("date").cast(pl.Date)
    )

    keep = ["date"] + meta["features"] + [c for c in DISPLAY_COLUMNS if c in panel.columns]
    panel.select(keep).write_csv(WEB_DATA / "panel.csv")

    (WEB_DATA / "scaler.json").write_text(json.dumps({
        "features": meta["features"],
        "log_features": meta["log_features"],
        "mean": meta["mean"],
        "std": meta["std"],
        "yield_features": list(YIELD_MATURITIES),
        "maturities": list(YIELD_MATURITIES.values()),
        "n_rows": meta["n_rows"],
        "start": meta["start"],
        "end": meta["end"],
        "coverage": meta["coverage"],
        "display_columns": [c for c in DISPLAY_COLUMNS if c in panel.columns],
    }, indent=2))

    ckpt = torch.load(CHECKPOINTS / "best.pt", map_location="cpu", weights_only=True)
    state = ckpt["model_state"]
    tensors = {
        k: {"shape": list(v.shape), "data": b64(v.detach().numpy())}
        for k, v in state.items()
    }
    (WEB_DATA / "model.json").write_text(json.dumps({
        "config": ckpt["model_cfg"],
        "epoch": ckpt["epoch"],
        "val_loss": ckpt["val_loss"],
        "tensors": tensors,
    }))

    labels = pl.read_parquet(REGIMES / "labels.parquet").with_columns(
        pl.col("date").cast(pl.Date)
    )
    labels.select(["date", "label", "regime_name"]).write_csv(WEB_DATA / "reference.csv")
    shutil.copy(REGIMES / "summary.json", WEB_DATA / "summary.json")

    latent = np.load(REGIMES / "latent.npz")["m_filt"]
    np.savetxt(WEB_DATA / "reference_latent.csv", latent, delimiter=",", fmt="%.6f")

    for f in sorted(WEB_DATA.iterdir()):
        print(f"  {f.name:24s} {f.stat().st_size / 1024:8.1f} KB")


if __name__ == "__main__":
    main()
