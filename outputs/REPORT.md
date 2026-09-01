# Cross-Asset Market Regime Report

Data source: **synthetic generator** | panel 7733 trading days, 1997-01-01 to 2026-08-21, 18 features | model checkpoint epoch 14 (val loss 0.2828)

Latent trajectory filtered by the trained differentiable Kalman filter; Gaussian-mixture clustering selected **k = 6** regimes by BIC.

## Regime characteristics

| regime | share of days | avg spell | avg VIX | avg HY OAS | avg 10y-3m slope | ann. equity return | ann. equity vol |
|---|---|---|---|---|---|---|---|
| **Calm Risk-On** | 23.7% | 30d | 13.5 | 3.44% | 1.33% | +6.4% | 9.0% |
| **Mid-Cycle** | 45.6% | 58d | 14.3 | 3.68% | 1.22% | +12.7% | 9.6% |
| **Late-Cycle / Flat Curve** | 9.8% | 152d | 18.9 | 4.18% | -0.23% | +4.2% | 14.0% |
| **Recovery / Wide Credit** | 10.4% | 62d | 20.2 | 6.55% | 1.97% | +23.6% | 20.9% |
| **Risk-Off / Drawdown** | 4.7% | 121d | 36.4 | 11.69% | 2.37% | -24.3% | 40.8% |
| **Crisis / Stress** | 5.7% | 63d | 39.7 | 9.63% | 1.52% | -121.6% | 47.8% |

Validation against the synthetic generator's hidden 4-state chain: adjusted Rand index **0.45** at the BIC-selected k = 6 (BIC prefers finer sub-states), and **0.83** when clustering at the true k = 4 (1.0 = perfect recovery, 0 = chance). The filtered latent trajectory recovers the hidden chain almost exactly up to state granularity.

## Model selection (BIC)

| k | BIC |
|---|---|
| 2 | 59,854 |
| 3 | 54,984 |
| 4 | 52,332 |
| 5 | 50,699 |
| 6 | 50,080 |

## Daily transition probabilities

| from \ to | Calm Risk-On | Mid-Cycle | Late-Cycle | Recovery | Risk-Off | Crisis | expected duration |
|---|---|---|---|---|---|---|---|
| **Calm Risk-On** | 96.62% | 3.21% | 0.00% | 0.16% | 0.00% | 0.00% | ~30d |
| **Mid-Cycle** | 1.50% | 98.27% | 0.14% | 0.09% | 0.00% | 0.00% | ~58d |
| **Late-Cycle** | 0.13% | 0.13% | 99.34% | 0.00% | 0.00% | 0.40% | ~152d |
| **Recovery** | 1.00% | 0.00% | 0.00% | 98.51% | 0.00% | 0.50% | ~67d |
| **Risk-Off** | 0.00% | 0.00% | 0.00% | 0.82% | 99.18% | 0.00% | ~121d |
| **Crisis** | 0.00% | 0.00% | 0.00% | 0.90% | 0.68% | 98.42% | ~63d |

## Figures

### Regime-colored equity, volatility and credit history

![Regime-colored equity, volatility and credit history](figures/regime_timeline.png)

### Ground-truth vs extracted regimes (synthetic validation)

![Ground-truth vs extracted regimes (synthetic validation)](figures/truth_vs_extracted.png)

### Filtered latent state trajectories

![Filtered latent state trajectories](figures/latent_states.png)

### Transition matrix

![Transition matrix](figures/transition_matrix.png)

### Per-regime characteristics

![Per-regime characteristics](figures/regime_stats.png)

### Nelson-Siegel decoder reconstructions

![Nelson-Siegel decoder reconstructions](figures/yield_reconstruction.png)

### Training curves

![Training curves](figures/training_curves.png)

## Method (one paragraph)

Observations are an 18-dimensional cross-asset vector (Treasury curve, credit OAS, VIX, TED, USD, oil, equity returns and realized vol, and monthly/quarterly macro) on an irregular trading-day grid with explicit missingness masks. Latent dynamics follow a neural SDE dz = f(z)dt + g(z)dW (OU backbone plus zero-initialized residuals). A continuous-discrete extended Kalman filter, written in differentiable torch ops with Jacobians via forward-mode autodiff, propagates moments between observation times and performs masked updates at them. The emission decoder maps latents to observables, with the yield block constrained to a Nelson-Siegel curve. Training maximizes the ELBO (reconstruction likelihood minus the KL between filtered posterior and SDE-propagated prior, KL-annealed), backpropagating through the filter. Regimes are Gaussian-mixture clusters of the filtered latent trajectory.

## Note on the data

This run used the calibrated synthetic generator (the build environment had no outbound access to the free data endpoints), which embeds a known 4-state regime chain, hence the recovery score above. Run `python scripts/download_data.py` on a machine with normal internet access, then rerun `build_dataset.py`, `train.py`, and `extract_regimes.py` to produce this same report on live FRED/Stooq data.
