"""Central configuration.

Typed dataclasses instead of YAML: every script imports these defaults and
exposes the interesting ones as CLI flags, so runs stay reproducible without
extra dependencies.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
OUTPUTS = PROJECT_ROOT / "outputs"
CHECKPOINTS = OUTPUTS / "checkpoints"
FIGURES = OUTPUTS / "figures"
REGIMES = OUTPUTS / "regimes"

YIELD_MATURITIES: dict[str, float] = {
    "y_3m": 0.25,
    "y_2y": 2.0,
    "y_5y": 5.0,
    "y_10y": 10.0,
    "y_30y": 30.0,
}
YIELD_FEATURES: list[str] = list(YIELD_MATURITIES)

MARKET_FEATURES: list[str] = [
    "hy_oas",
    "ig_oas",
    "vix",
    "ted",
    "usd_ret",
    "oil_ret",
    "spx_ret",
    "spx_rv21",
    "ndx_ret",
]
MACRO_FEATURES: list[str] = [
    "cpi_yoy",
    "indpro_yoy",
    "unrate",
    "gdp_yoy",
]
FEATURES: list[str] = YIELD_FEATURES + MARKET_FEATURES + MACRO_FEATURES
N_OBS: int = len(FEATURES)

LOG_FEATURES: list[str] = ["vix", "spx_rv21"]


@dataclass
class DataConfig:
    start: str = "1997-01-01"
    end: str | None = None
    raw_dir: Path = DATA_RAW
    processed_dir: Path = DATA_PROCESSED
    oil_ret_clip: float = 0.5
    rv_window: int = 21


@dataclass
class ModelConfig:
    latent_dim: int = 6
    obs_dim: int = N_OBS
    hidden: int = 64
    enc_hidden: int = 64
    ns_lambda_init: float = 0.6
    obs_noise_init: float = 0.5
    diffusion_init: float = 0.3
    mean_reversion_init: float = 2.0
    jitter: float = 1e-8


@dataclass
class TrainConfig:
    window: int = 100
    stride: int = 10
    batch_size: int = 24
    epochs: int = 15
    lr: float = 3e-3
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    kl_anneal_frac: float = 0.4
    beta_max: float = 1.0
    val_frac: float = 0.15
    patience: int = 5
    seed: int = 7
    device: str = "cpu"


@dataclass
class RegimeConfig:
    k_min: int = 2
    k_max: int = 6
    gmm_n_init: int = 5
    smooth_window: int = 5
    seed: int = 7


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    regime: RegimeConfig = field(default_factory=RegimeConfig)

    def dump(self, path: str | Path) -> None:
        d = asdict(self)
        for section in d.values():
            for k, v in section.items():
                if isinstance(v, Path):
                    try:
                        section[k] = v.resolve().relative_to(PROJECT_ROOT).as_posix()
                    except ValueError:
                        section[k] = str(v)
        Path(path).write_text(json.dumps(d, indent=2))


def ensure_dirs() -> None:
    for p in (DATA_RAW, DATA_PROCESSED, CHECKPOINTS, FIGURES, REGIMES):
        p.mkdir(parents=True, exist_ok=True)
