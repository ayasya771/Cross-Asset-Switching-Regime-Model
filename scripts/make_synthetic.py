#!/usr/bin/env python3
"""Generate the synthetic raw dataset (used when the free data endpoints are
unreachable, and for architecture validation against known regimes)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from regimelab.config import DataConfig, ensure_dirs
from regimelab.data.synthetic import generate


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="1997-01-01")
    ap.add_argument("--end", default="2026-08-21")
    ap.add_argument("--seed", type=int, default=11)
    args = ap.parse_args()
    ensure_dirs()
    info = generate(DataConfig(), start=args.start, end=args.end, seed=args.seed)
    print(f"synthetic panel: {info['days']} business days  {info['start']} .. {info['end']}")
    for k, v in info["regime_days"].items():
        print(f"  {k:20s} {v:>6} days ({v / info['days']:.1%})")


if __name__ == "__main__":
    main()
