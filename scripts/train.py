#!/usr/bin/env python3
"""Phase 5: train the model end to end; checkpoints land in outputs/checkpoints."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from regimelab.config import CHECKPOINTS, DATA_PROCESSED, Config, ensure_dirs
from regimelab.training.trainer import train


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--window", type=int, default=None)
    ap.add_argument("--stride", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--latent-dim", type=int, default=None)
    ap.add_argument("--hidden", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--beta-max", type=float, default=None)
    args = ap.parse_args()

    cfg = Config()
    for name in ("epochs", "window", "stride", "batch_size", "lr", "seed", "beta_max"):
        v = getattr(args, name)
        if v is not None:
            setattr(cfg.train, name, v)
    for name in ("latent_dim", "hidden"):
        v = getattr(args, name)
        if v is not None:
            setattr(cfg.model, name, v)

    ensure_dirs()
    cfg.dump(CHECKPOINTS / "run_config.json")
    result = train(cfg, DATA_PROCESSED, CHECKPOINTS)
    print(f"done, best val loss {result['best_val']:.4f}")


if __name__ == "__main__":
    main()
