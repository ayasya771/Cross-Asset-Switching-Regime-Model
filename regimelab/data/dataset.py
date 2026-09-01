"""Windowed PyTorch dataset over the processed arrays.

Each item is a dict of:
- ``x``  (W, n) standardized observations, zeros where missing
- ``m``  (W, n) observation mask (1 = observed)
- ``dt`` (W,)   calendar-year gap since the previous grid date; dt[0] = 0 so
                the filter starts each window from the encoder's initial state
- ``start``     index of the window's first row in the full panel
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


def load_meta(processed_dir: Path) -> dict:
    return json.loads((Path(processed_dir) / "meta.json").read_text())


def load_arrays(processed_dir: Path) -> dict[str, np.ndarray]:
    z = np.load(Path(processed_dir) / "arrays.npz")
    return {k: z[k] for k in z.files}


class PanelWindows(Dataset):
    def __init__(
        self,
        processed_dir: Path,
        window: int,
        stride: int,
        split: str = "train",
        val_frac: float = 0.15,
    ):
        arrs = load_arrays(processed_dir)
        self.X = arrs["X"]
        self.M = arrs["M"].astype(np.float64)
        self.dt = arrs["dt"]
        T = len(self.X)
        if window > T:
            raise ValueError(f"window {window} exceeds panel length {T}")
        cut = int(T * (1.0 - val_frac))
        starts = list(range(0, T - window + 1, stride))
        if split == "train":
            starts = [s for s in starts if s + window <= cut]
        elif split == "val":
            starts = [s for s in starts if s >= cut]
        elif split != "all":
            raise ValueError(f"unknown split {split!r}")
        if not starts:
            raise ValueError(f"split {split!r} produced zero windows (T={T}, window={window})")
        self.starts = starts
        self.window = window

    def __len__(self) -> int:
        return len(self.starts)

    def __getitem__(self, i: int) -> dict:
        s = self.starts[i]
        w = self.window
        dt = self.dt[s : s + w].copy()
        dt[0] = 0.0
        return {
            "x": torch.from_numpy(self.X[s : s + w]).double(),
            "m": torch.from_numpy(self.M[s : s + w]).double(),
            "dt": torch.from_numpy(dt).double(),
            "start": s,
        }


def full_sequence(processed_dir: Path) -> dict:
    """The entire panel as one batch-1 sequence, for regime extraction."""
    arrs = load_arrays(processed_dir)
    dt = arrs["dt"].copy()
    dt[0] = 0.0
    return {
        "x": torch.from_numpy(arrs["X"]).double().unsqueeze(0),
        "m": torch.from_numpy(arrs["M"].astype(np.float64)).double().unsqueeze(0),
        "dt": torch.from_numpy(dt).double().unsqueeze(0),
        "dates": arrs["dates"],
    }
