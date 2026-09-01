#!/usr/bin/env python3
"""End-to-end pipeline: download -> build dataset -> train -> extract regimes."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent


def run(script: str, *extra: str) -> None:
    cmd = [sys.executable, str(SCRIPTS / script), *extra]
    print(f"\n=== {' '.join(cmd)} ===", flush=True)
    subprocess.run(cmd, check=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-download", action="store_true",
                    help="reuse whatever is already in data/raw")
    ap.add_argument("--epochs", type=int, default=None)
    args = ap.parse_args()

    if not args.skip_download:
        run("download_data.py")
    run("build_dataset.py")
    train_args = ["--epochs", str(args.epochs)] if args.epochs else []
    run("train.py", *train_args)
    run("extract_regimes.py")


if __name__ == "__main__":
    main()
