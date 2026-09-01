#!/usr/bin/env python3
"""Phase 1a: download every raw series (FRED / Stooq / Yahoo) into data/raw."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from regimelab.config import DataConfig, ensure_dirs
from regimelab.data.sources import download_all


def main() -> None:
    ensure_dirs()
    manifest = download_all(DataConfig())
    ok = 0
    for name, info in sorted(manifest.items()):
        status = info["status"]
        if status == "ok":
            ok += 1
            print(f"  {name:12s} {info['source']:22s} {info['rows']:>7} rows  "
                  f"{info['start']} .. {info['end']}")
        else:
            print(f"  {name:12s} {info['source']:22s} {status}")
    print(f"{ok}/{len(manifest)} series downloaded")
    if ok == 0:
        raise SystemExit("every download failed: check network access")


if __name__ == "__main__":
    main()
