"""Report figures. Light-surface, validated reference palette (dataviz method):
categorical slots assigned to regimes in fixed stress order, sequential blue
ramp for the transition heatmap, ink/grid tokens for chrome. One axis per
panel; regime identity always carried by legend + labels, never color alone.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
SEQ = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
       "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]

plt.rcParams.update({
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "font.family": "sans-serif",
    "font.size": 9.5,
    "axes.edgecolor": BASELINE,
    "axes.labelcolor": INK2,
    "axes.titlecolor": INK,
    "axes.titlesize": 10.5,
    "axes.titleweight": "semibold",
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "text.color": INK,
})


def _style(ax, grid_axis: str = "y") -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(BASELINE)
        ax.spines[side].set_linewidth(0.8)
    if grid_axis:
        ax.grid(True, axis=grid_axis, color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)


def _runs(labels: np.ndarray) -> list[tuple[int, int, int]]:
    """Contiguous (start, end_exclusive, label) runs."""
    out, s = [], 0
    for i in range(1, len(labels) + 1):
        if i == len(labels) or labels[i] != labels[s]:
            out.append((s, i, int(labels[s])))
            s = i
    return out


def regime_spans(ax, dates, labels, colors, alpha: float = 0.30) -> None:
    for s, e, lab in _runs(labels):
        ax.axvspan(dates[s], dates[min(e, len(dates) - 1)], color=colors[lab],
                   alpha=alpha, linewidth=0)


def _regime_legend(fig, names, colors, extra=()):
    handles = [Patch(facecolor=colors[i], alpha=0.55, label=n) for i, n in enumerate(names)]
    handles += list(extra)
    fig.legend(handles=handles, loc="upper center", ncol=min(len(handles), 4),
               frameon=False, bbox_to_anchor=(0.5, 1.0), fontsize=9)


def fig_timeline(dates, spx, vix, hy, labels, names, colors, path) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(11, 7.6), sharex=True,
                             gridspec_kw={"height_ratios": [2.2, 1, 1]})
    panels = [
        (spx, "Equity index (log scale)", True),
        (vix, "Volatility index", False),
        (hy, "High-yield credit spread (%)", False),
    ]
    for ax, (series, title, logscale) in zip(axes, panels):
        regime_spans(ax, dates, labels, colors)
        ok = ~np.isnan(series)
        ax.plot(dates[ok], series[ok], color=INK, linewidth=1.1)
        if logscale:
            ax.set_yscale("log")
        ax.set_title(title, loc="left")
        _style(ax)
    axes[-1].xaxis.set_major_locator(mdates.YearLocator(2))
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    _regime_legend(fig, names, colors)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_latent(dates, m_filt, labels, names, colors, path, n_dims: int = 4) -> None:
    n_dims = min(n_dims, m_filt.shape[1])
    fig, axes = plt.subplots(n_dims, 1, figsize=(11, 1.9 * n_dims), sharex=True)
    axes = np.atleast_1d(axes)
    for k, ax in enumerate(axes):
        regime_spans(ax, dates, labels, colors, alpha=0.22)
        ax.plot(dates, m_filt[:, k], color=INK, linewidth=1.0)
        ax.set_title(f"latent state z{k + 1} (filtered mean)", loc="left")
        _style(ax)
    axes[-1].xaxis.set_major_locator(mdates.YearLocator(2))
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    _regime_legend(fig, names, colors)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_transitions(P, names, durations, path) -> None:
    k = P.shape[0]
    fig, ax = plt.subplots(figsize=(1.5 * k + 3.2, 1.1 * k + 2.2))
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("seq_blue", SEQ)
    ax.imshow(P, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    for i in range(k):
        for j in range(k):
            v = P[i, j]
            ink = "#ffffff" if v > 0.55 else INK
            ax.text(j, i, f"{v:.1%}" if v >= 0.0005 else "<0.1%",
                    ha="center", va="center", fontsize=9, color=ink)
    short = [n.split(" /")[0] for n in names]
    ax.set_xticks(range(k), short, fontsize=8.5, color=INK2)
    ax.set_yticks(range(k), [f"{n}\n(~{d:.0f}d)" for n, d in zip(short, durations)],
                  fontsize=8.5, color=INK2)
    ax.set_title("Daily transition probabilities (rows: from, ~expected duration)",
                 loc="left")
    for side in ax.spines.values():
        side.set_visible(False)
    ax.tick_params(length=0)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_regime_stats(stats_rows: list[dict], names, colors, path) -> None:
    """Small multiples, one metric per panel; regime keeps its color everywhere."""
    metrics = [
        ("vix", "avg volatility index"),
        ("hy_oas", "avg HY spread (%)"),
        ("slope", "avg curve slope 10y-3m (%)"),
        ("ann_ret", "annualized equity return (%)"),
        ("ann_vol", "annualized equity vol (%)"),
        ("occupancy", "share of days (%)"),
    ]
    k = len(names)
    fig, axes = plt.subplots(2, 3, figsize=(11.5, 5.4))
    for (key, title), ax in zip(metrics, axes.ravel()):
        vals = [r[key] for r in stats_rows]
        y = np.arange(k)[::-1]
        ax.barh(y, vals, height=0.55, color=[colors[i] for i in range(k)])
        for yi, v in zip(y, vals):
            ax.text(v + (0.01 * (max(vals) - min(min(vals), 0)) + 1e-9) * np.sign(v or 1),
                    yi, f"{v:,.1f}", va="center", fontsize=8, color=INK2)
        ax.set_yticks(y, [n.split(" /")[0] for n in names], fontsize=8, color=INK2)
        ax.set_title(title, loc="left", fontsize=9.5)
        ax.axvline(0, color=BASELINE, linewidth=0.8)
        _style(ax, grid_axis="x")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_training(history: list[dict], path) -> None:
    ep = [h["epoch"] for h in history]
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.2))
    series = [
        [("train_loss", "train", CAT[0]), ("val_loss", "validation", CAT[1])],
        [("train_nll", "train", CAT[0]), ("val_nll", "validation", CAT[1])],
        [("train_kl", "train", CAT[0]), ("val_kl", "validation", CAT[1])],
    ]
    titles = ["ELBO loss / observed entry", "reconstruction NLL", "KL (filter vs SDE prior)"]
    for ax, group, title in zip(axes, series, titles):
        for key, label, color in group:
            ax.plot(ep, [h[key] for h in history], color=color,
                    linewidth=1.8, label=label)
        ax.set_title(title, loc="left")
        ax.set_xlabel("epoch", color=MUTED)
        _style(ax)
        ax.legend(frameon=False, fontsize=8.5)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_yield_recon(taus, obs_list, fit_list, date_labels, path) -> None:
    n = len(obs_list)
    fig, axes = plt.subplots(1, n, figsize=(2.9 * n, 3.0), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, obs, fit, lab in zip(axes, obs_list, fit_list, date_labels):
        ax.plot(taus, fit, color=CAT[0], linewidth=1.8, label="Nelson-Siegel decode")
        ax.scatter(taus, obs, color=INK, s=18, zorder=3, label="observed")
        ax.set_title(str(lab), loc="left", fontsize=9)
        ax.set_xlabel("maturity (years)", color=MUTED)
        _style(ax)
    axes[0].set_ylabel("yield (%)", color=INK2)
    handles, labels_ = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels_, loc="upper right", frameon=False, fontsize=8.5)
    fig.suptitle("Yield curve reconstruction through the structural decoder head",
                 x=0.01, ha="left", fontsize=10.5, fontweight="semibold")
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_truth_vs_extracted(dates, truth, extracted, truth_names, names, colors,
                           ari, path, truth_colors=None) -> None:
    """Synthetic-data validation: ground-truth chain vs extracted regimes.
    ``truth_colors`` should map each truth state to the hue of its
    majority-matched extracted regime, so aligned segments share color."""
    fig, axes = plt.subplots(2, 1, figsize=(11, 3.4), sharex=True)
    t_colors = truth_colors or [CAT[i % len(CAT)] for i in range(len(truth_names))]
    for ax, lab, cols, title in [
        (axes[0], truth, t_colors, "ground-truth regime chain (synthetic generator)"),
        (axes[1], extracted, colors, f"extracted regimes (adjusted Rand index {ari:.2f})"),
    ]:
        regime_spans(ax, dates, lab, cols, alpha=0.85)
        ax.set_yticks([])
        ax.set_title(title, loc="left")
        _style(ax, grid_axis="")
    axes[-1].xaxis.set_major_locator(mdates.YearLocator(2))
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    handles = [Patch(facecolor=t_colors[i], alpha=0.9, label=n)
               for i, n in enumerate(truth_names)]
    handles += [Patch(facecolor=colors[i], alpha=0.9, label=names[i].split(" /")[0])
                for i in range(len(names))]
    fig.legend(handles=handles, loc="upper center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, 1.02), fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.savefig(path, dpi=150)
    plt.close(fig)
