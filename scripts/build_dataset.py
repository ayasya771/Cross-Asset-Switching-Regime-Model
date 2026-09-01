#!/usr/bin/env python3
"""Phase 1b: raw series -> aligned, masked, standardized model arrays."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from regimelab.config import DataConfig, ensure_dirs
from regimelab.data.preprocess import build_panel


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="1997-01-01")
    ap.add_argument("--end", default=None)
    ap.add_argument("--split-frac", type=float, default=0.85,
                    help="fraction of rows whose stats define the scaler (train region)")
    args = ap.parse_args()

    ensure_dirs()
    cfg = DataConfig(start=args.start, end=args.end)
    meta = build_panel(cfg, split_frac=args.split_frac)
    print(f"panel: {meta['n_rows']} rows  {meta['start']} .. {meta['end']}")
    print(f"features kept ({len(meta['features'])}):")
    for f in meta["features"]:
        print(f"  {f:12s} coverage {meta['coverage'][f]:6.1%}")
    if meta["dropped"]:
        print(f"dropped (insufficient data): {meta['dropped']}")


if __name__ == "__main__":
    main()
