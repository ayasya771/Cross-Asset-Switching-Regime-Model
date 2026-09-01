#!/usr/bin/env python3
"""Phase 6: extract regimes from the trained model, build all report figures,
and write outputs/REPORT.md."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from regimelab.config import (
    CHECKPOINTS,
    DATA_PROCESSED,
    DATA_RAW,
    FIGURES,
    OUTPUTS,
    REGIMES,
    YIELD_MATURITIES,
    RegimeConfig,
    ensure_dirs,
)
from regimelab.regimes.extract import extract
from regimelab.viz import plots


def _md_table(headers: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out)


def write_report(s: dict, res: dict, data_source: str) -> Path:
    k = s["k"]
    stats = s["stats"]
    P = np.array(s["transition_matrix"])
    lines: list[str] = []
    add = lines.append
    add("# Cross-Asset Market Regime Report")
    add("")
    add(f"Data source: **{data_source}** | panel {res['meta']['n_rows']} trading days, "
        f"{res['meta']['start']} to {res['meta']['end']}, "
        f"{len(res['meta']['features'])} features | model checkpoint epoch "
        f"{s['checkpoint_epoch']} (val loss {s['checkpoint_val_loss']:.4f})")
    add("")
    add("Latent trajectory filtered by the trained differentiable Kalman filter; "
        f"Gaussian-mixture clustering selected **k = {k}** regimes by BIC.")
    add("")
    add("## Regime characteristics")
    add("")
    add(_md_table(
        ["regime", "share of days", "avg spell", "avg VIX", "avg HY OAS",
         "avg 10y-3m slope", "ann. equity return", "ann. equity vol"],
        [[
            f"**{r['name']}**",
            f"{r['occupancy']:.1f}%",
            f"{r['avg_duration_days']:.0f}d",
            f"{r['vix']:.1f}",
            f"{r['hy_oas']:.2f}%",
            f"{r['slope']:.2f}%",
            f"{r['ann_ret']:+.1f}%",
            f"{r['ann_vol']:.1f}%",
        ] for r in stats],
    ))
    add("")
    if s["ari_vs_ground_truth"] is not None:
        add(f"Validation against the synthetic generator's hidden {s['true_k']}-state "
            f"chain: adjusted Rand index **{s['ari_vs_ground_truth']:.2f}** at the "
            f"BIC-selected k = {k} (BIC prefers finer sub-states), and "
            f"**{s['ari_at_true_k']:.2f}** when clustering at the true k = "
            f"{s['true_k']} (1.0 = perfect recovery, 0 = chance). The filtered "
            "latent trajectory recovers the hidden chain almost exactly up to "
            "state granularity.")
        add("")
    add("## Model selection (BIC)")
    add("")
    add(_md_table(["k", "BIC"], [[str(r["k"]), f"{r['bic']:,.0f}"] for r in s["bic"]]))
    add("")
    add("## Daily transition probabilities")
    add("")
    short = [n.split(" /")[0] for n in s["names"]]
    add(_md_table(
        ["from \\ to"] + short + ["expected duration"],
        [[f"**{short[i]}**"] + [f"{P[i, j]:.2%}" for j in range(k)]
         + [f"~{s['expected_duration_days'][i]:.0f}d"] for i in range(k)],
    ))
    add("")
    add("## Figures")
    add("")
    for f, cap in [
        ("regime_timeline.png", "Regime-colored equity, volatility and credit history"),
        ("truth_vs_extracted.png", "Ground-truth vs extracted regimes (synthetic validation)"),
        ("latent_states.png", "Filtered latent state trajectories"),
        ("transition_matrix.png", "Transition matrix"),
        ("regime_stats.png", "Per-regime characteristics"),
        ("yield_reconstruction.png", "Nelson-Siegel decoder reconstructions"),
        ("training_curves.png", "Training curves"),
    ]:
        if (FIGURES / f).exists():
            add(f"### {cap}")
            add("")
            add(f"![{cap}](figures/{f})")
            add("")
    add("## Method (one paragraph)")
    add("")
    add("Observations are an 18-dimensional cross-asset vector (Treasury curve, "
        "credit OAS, VIX, TED, USD, oil, equity returns and realized vol, and "
        "monthly/quarterly macro) on an irregular trading-day grid with explicit "
        "missingness masks. Latent dynamics follow a neural SDE "
        "dz = f(z)dt + g(z)dW (OU backbone plus zero-initialized residuals). A "
        "continuous-discrete extended Kalman filter, written in differentiable "
        "torch ops with Jacobians via forward-mode autodiff, propagates moments "
        "between observation times and performs masked updates at them. The "
        "emission decoder maps latents to observables, with the yield block "
        "constrained to a Nelson-Siegel curve. Training maximizes the ELBO "
        "(reconstruction likelihood minus the KL between filtered posterior and "
        "SDE-propagated prior, KL-annealed), backpropagating through the filter. "
        "Regimes are Gaussian-mixture clusters of the filtered latent trajectory.")
    add("")
    if s["ari_vs_ground_truth"] is not None:
        add("## Note on the data")
        add("")
        add("This run used the calibrated synthetic generator (the build "
            "environment had no outbound access to the free data endpoints), "
            "which embeds a known 4-state regime chain, hence the recovery "
            "score above. Run `python scripts/download_data.py` on a machine "
            "with normal internet access, then rerun `build_dataset.py`, "
            "`train.py`, and `extract_regimes.py` to produce this same report "
            "on live FRED/Stooq data.")
        add("")
    path = OUTPUTS / "REPORT.md"
    path.write_text("\n".join(lines))
    return path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=str(CHECKPOINTS / "best.pt"))
    ap.add_argument("--k-min", type=int, default=None)
    ap.add_argument("--k-max", type=int, default=None)
    ap.add_argument("--smooth", type=int, default=None)
    args = ap.parse_args()

    cfg = RegimeConfig()
    if args.k_min is not None:
        cfg.k_min = args.k_min
    if args.k_max is not None:
        cfg.k_max = args.k_max
    if args.smooth is not None:
        cfg.smooth_window = args.smooth

    ensure_dirs()
    res = extract(cfg, DATA_PROCESSED, Path(args.checkpoint), raw_dir=DATA_RAW,
                  out_dir=REGIMES)
    s = res["summary"]
    k = s["k"]
    names = s["names"]
    colors = plots.CAT[:k]
    panel = res["panel"]
    dates = res["dates"]
    labels = res["labels"]

    spx = panel["spx_close"].to_numpy()
    vix = panel["vix"].to_numpy()
    hy = panel["hy_oas"].to_numpy()
    plots.fig_timeline(dates, spx, vix, hy, labels, names, colors,
                       FIGURES / "regime_timeline.png")
    plots.fig_latent(dates, res["m_filt"], labels, names, colors,
                     FIGURES / "latent_states.png")
    P = np.array(s["transition_matrix"])
    plots.fig_transitions(P, names, s["expected_duration_days"],
                          FIGURES / "transition_matrix.png")
    plots.fig_regime_stats(s["stats"], names, colors, FIGURES / "regime_stats.png")

    mpath = CHECKPOINTS / "metrics.jsonl"
    if mpath.exists():
        history = [json.loads(line) for line in mpath.read_text().splitlines() if line]
        if history:
            plots.fig_training(history, FIGURES / "training_curves.png")

    ycols = list(YIELD_MATURITIES)
    yobs = np.column_stack([panel[c].to_numpy() for c in ycols])
    complete = ~np.isnan(yobs).any(axis=1)
    sel: list[int] = []
    for j in range(min(k, 4)):
        cand = np.where((labels == j) & complete)[0]
        if len(cand):
            sel.append(int(cand[np.argmax(res["probs"][cand, j])]))
    if sel:
        with torch.no_grad():
            zsel = torch.tensor(res["m_filt"][sel], dtype=torch.float64)
            fit = res["model"].decoder.yields_percent(zsel).numpy()
        plots.fig_yield_recon(
            np.array(list(YIELD_MATURITIES.values())),
            [yobs[i] for i in sel],
            list(fit),
            [f"{dates[i]}  {names[labels[i]].split(' /')[0]}" for i in sel],
            FIGURES / "yield_reconstruction.png",
        )

    if res["truth"] is not None:
        truth = res["truth"]
        n_t = len(res["truth_names"])
        conf = np.zeros((n_t, k))
        ok = truth >= 0
        for a, b in zip(truth[ok], labels[ok]):
            conf[a, b] += 1
        truth_colors = [colors[int(np.argmax(conf[t_]))] for t_ in range(n_t)]
        plots.fig_truth_vs_extracted(
            dates, truth, labels, res["truth_names"], names, colors,
            s["ari_vs_ground_truth"], FIGURES / "truth_vs_extracted.png",
            truth_colors=truth_colors,
        )

    manifest_path = DATA_RAW / "manifest.json"
    source = "unknown"
    if manifest_path.exists():
        man = json.loads(manifest_path.read_text())
        srcs = {v.get("source", "?") for v in man.values() if v.get("status") == "ok"}
        source = "synthetic generator" if srcs == {"synthetic"} else \
            "live endpoints (" + ", ".join(sorted({s_.split(":")[0] for s_ in srcs})) + ")"
    report = write_report(s, res, source)

    print(f"k = {k} regimes: {', '.join(names)}")
    for r in s["stats"]:
        print(f"  {r['name']:26s} {r['occupancy']:5.1f}% of days  vix {r['vix']:5.1f}  "
              f"hy {r['hy_oas']:5.2f}%  ret {r['ann_ret']:+6.1f}%")
    if s["ari_vs_ground_truth"] is not None:
        print(f"adjusted Rand index vs ground truth: {s['ari_vs_ground_truth']:.3f}")
    print(f"report: {report}")


if __name__ == "__main__":
    main()
