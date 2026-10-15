#!/usr/bin/env python3
"""End-to-end pipeline: acquire data, build the panel, train, extract regimes,
render both reports, and refresh the payload the web demo loads.

Data acquisition has three modes. ``--synthetic`` generates the calibrated panel
with known regime ground truth and needs no network. ``--custom CSV`` ingests a
user-supplied wide CSV (see docs/CUSTOM_DATA.md). The default downloads the
real series from the free endpoints; if that fails the run stops rather than
silently substituting synthetic data, since the two are not interchangeable
for any claim about real markets.
"""
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
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--synthetic", action="store_true",
                     help="generate the synthetic panel instead of downloading")
    src.add_argument("--skip-download", action="store_true",
                     help="reuse whatever is already in data/raw")
    src.add_argument("--custom", metavar="CSV",
                     help="ingest your own wide CSV instead of downloading")
    ap.add_argument("--epochs", type=int, default=None)
    args = ap.parse_args()

    if args.custom:
        run("ingest_custom.py", args.custom)
    elif args.synthetic:
        run("make_synthetic.py")
    elif not args.skip_download:
        try:
            run("download_data.py")
        except subprocess.CalledProcessError:
            print(
                "\ndownload failed. Re-run with --synthetic to use the offline "
                "validation panel, or with --skip-download to reuse data/raw.",
                file=sys.stderr,
            )
            raise SystemExit(1)

    if not args.custom:
        run("build_dataset.py")
    run("train.py", *(["--epochs", str(args.epochs)] if args.epochs else []))
    run("extract_regimes.py")
    run("build_report_page.py")
    run("export_web_demo.py")


if __name__ == "__main__":
    main()
