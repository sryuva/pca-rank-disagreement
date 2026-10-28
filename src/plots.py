"""Publication-quality figures for the technical note."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RES, FIG = ROOT / "results", ROOT / "figures"
FIG.mkdir(exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "axes.linewidth": 0.7,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "legend.fontsize": 7.5,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

df = pd.read_csv(RES / "results.csv")
order = ["small", "medium", "large"]
gap_labels = ["Small (1.5)", "Medium (3)", "Large (10)"]
NS = sorted(df.n.unique())


def _save(fig, stem):
    fig.savefig(FIG / f"{stem}.png", dpi=450, bbox_inches="tight", pad_inches=0.04)
    fig.savefig(FIG / f"{stem}.pdf", bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def heat(metric, legend, stem, cmap, vmin, vmax, fmt, norm=None):
    agg = df.groupby(["n", "separation", "signal_pc"])[metric].mean().reset_index()
    fig, axes = plt.subplots(2, 2, figsize=(7.05, 5.25), sharex=True, sharey=True,
                             constrained_layout=True)
    axes = axes.ravel()
    im = None
    for ax, n in zip(axes, NS):
        m = (agg[agg.n == n]
             .pivot(index="separation", columns="signal_pc", values=metric)
             .loc[order])
        im = ax.imshow(m.values, cmap=cmap, vmin=None if norm else vmin,
                       vmax=None if norm else vmax, norm=norm, aspect="auto")
        for (i, j), value in np.ndenumerate(m.values):
            rgba = im.cmap(im.norm(value))
            luminance = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
            ax.text(j, i, fmt.format(value), ha="center", va="center",
                    fontsize=7.4, color="white" if luminance < 0.48 else "#111111")
        ax.axvline(2.5, color="#222222", lw=1.0, ls=(0, (3, 2)))
        ax.set_title(f"Sample size  n = {n}", pad=5, fontweight="semibold")
        ax.set_xticks(range(6), range(1, 7))
        ax.set_yticks(range(3), gap_labels)
        ax.tick_params(length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
    for ax in axes[2:]:
        ax.set_xlabel("PC carrying the signal")
    for ax in axes[::2]:
        ax.set_ylabel("Eigengap  $\\lambda_3/\\lambda_4$")
    cbar = fig.colorbar(im, ax=axes, shrink=0.82, pad=0.025, aspect=28)
    cbar.set_label(legend, rotation=270, labelpad=13)
    cbar.outline.set_linewidth(0.6)
    _save(fig, stem)


heat("disagree", "Fraction of runs", "fig1_disagreement_rate",
     "Blues", 0, 1, "{:.2f}")
delta_norm = TwoSlopeNorm(vmin=-0.10, vcenter=0.0, vmax=1.05)
heat("delta_mse", "Mean excess test MSE", "fig2_test_error_cost",
     "RdBu_r", -0.10, 1.05, "{:.2f}", norm=delta_norm)

sd = pd.read_csv(RES / "subspace_distance.csv")
fig, axes = plt.subplots(1, 3, figsize=(7.05, 2.75), sharex=True, sharey=True,
                         constrained_layout=True)
colors = {"empirical": "#111111", "predicted": "#2468a2", "bound": "#b24a3b"}
for ax, sep, title in zip(axes, order, gap_labels):
    s = sd[sd.separation == sep]
    ax.plot(s.n, s.empirical, "o-", color=colors["empirical"], ms=3.5, lw=1.2,
            label="Measured")
    ax.plot(s.n, s.predicted, "-", color=colors["predicted"], lw=1.35,
            label="First-order")
    ax.plot(s.n, s.davis_kahan_bound, "--", color=colors["bound"], lw=1.2,
            label="Davis–Kahan")
    ax.axhline(np.sqrt(6), color="#888888", ls=":", lw=0.8)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(which="major", color="#d9d9d9", lw=0.5)
    ax.grid(which="minor", visible=False)
    ax.set_title(title, fontweight="semibold", pad=5)
    ax.set_xlabel("Sample size  n")
    ax.spines[["top", "right"]].set_visible(False)
axes[0].set_ylabel(r"RMS projector distance  $\|\hat P-P\|_F$")
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.08),
           ncol=3, frameon=False, handlelength=2.2, columnspacing=1.6)
_save(fig, "fig3_subspace_distance")
