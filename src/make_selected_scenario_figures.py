# -*- coding: utf-8 -*-
"""Publication exports for the selected transition-window scenario experiment."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parents[1]
OUT = BASE / "results" / "scenario"
PRED = OUT / "scenario_predictions_long.csv"
FINAL = OUT / "scenario_final_outcomes.csv"

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "font.size": 8,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "legend.frameon": False,
})
COLORS = {
    "continued_work": "#4C566A",
    "rest": "#2A9D8F",
    "rest_environment": "#457B9D",
    "workload_reduction": "#E09F3E",
}
LABELS = {
    "continued_work": "Continued work",
    "rest": "Rest",
    "rest_environment": "Rest + environment",
    "workload_reduction": "Workload reduction",
}
ORDER = list(LABELS)


def save(fig, stem):
    fig.savefig(OUT / f"{stem}.png", dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(OUT / f"{stem}.svg", bbox_inches="tight", facecolor="white")
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight", facecolor="white")
    fig.savefig(OUT / f"{stem}.tiff", dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def fig10():
    df = pd.read_csv(PRED)
    df = df[df["cohort"] == "primary"].copy()
    ids = sorted(df["sample_id"].unique())
    variables = [("predicted_HR", "HR (bpm)"), ("predicted_Tskin", "Skin temperature (°C)"), ("predicted_Tcore", "Core temperature (°C)")]
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.55), constrained_layout=True)
    x = np.arange(1, 16)
    for ax, (column, ylabel) in zip(axes, variables):
        for scenario in ORDER:
            g = df[df["scenario"] == scenario]
            mat = g.pivot(index="sample_id", columns="minute", values=column).loc[ids].to_numpy(float)
            q25, med, q75 = np.quantile(mat, [0.25, 0.5, 0.75], axis=0)
            ax.fill_between(x, q25, q75, color=COLORS[scenario], alpha=0.13, linewidth=0)
            ax.plot(x, med, color=COLORS[scenario], lw=2.0, label=LABELS[scenario], zorder=4)
            for row in mat:
                ax.plot(x, row, color=COLORS[scenario], alpha=0.24, lw=0.7, zorder=2)
        ax.set_xlabel("Future minute")
        ax.set_ylabel(ylabel)
        ax.set_xlim(1, 15)
        ax.set_xticks([1, 5, 10, 15])
        ax.grid(axis="y", color="#D9DDE3", lw=0.55, alpha=0.8)
    axes[0].set_title("(a) Heart rate")
    axes[1].set_title("(b) Skin temperature")
    axes[2].set_title("(c) Core temperature")
    axes[-1].legend(loc="upper right", fontsize=6.4, handlelength=1.8)
    save(fig, "Fig10_primary_scenario_trajectories")
    # Keep an intuitive PNG alias used in the earlier output directory.
    import shutil
    shutil.copyfile(OUT / "Fig10_primary_scenario_trajectories.png", OUT / "primary_predicted_trajectories.png")


def fig11():
    df = pd.read_csv(FINAL)
    df = df[df["cohort"] == "primary"].copy()
    ids = sorted(df["sample_id"].unique())
    wide = df.pivot(index="sample_id", columns="scenario", values="final_mPSI").loc[ids, ORDER]
    fig, ax = plt.subplots(figsize=(5.5, 3.15), constrained_layout=True)
    xpos = np.arange(len(ORDER))
    # Pale individual paired lines make the small-sample structure explicit.
    for _, row in wide.iterrows():
        ax.plot(xpos, row.to_numpy(float), color="#9AA3AF", lw=0.85, alpha=0.72, zorder=1)
        ax.scatter(xpos, row.to_numpy(float), s=18, color=[COLORS[s] for s in ORDER], edgecolor="white", linewidth=0.35, zorder=3)
    med = wide.median(axis=0).to_numpy(float)
    ax.plot(xpos, med, color="#111827", lw=2.3, marker="o", ms=5.0, markerfacecolor="white", markeredgewidth=1.2, label="Median", zorder=5)
    for x, y in zip(xpos, med):
        ax.text(x, y + 0.28, f"{y:.2f}", ha="center", va="bottom", fontsize=7, color="#111827")
    ax.set_xticks(xpos, [LABELS[s] for s in ORDER], rotation=18, ha="right")
    ax.set_ylabel("Predicted final mPSI")
    ax.set_title("Primary transition blocks (n = 6)")
    ax.grid(axis="y", color="#D9DDE3", lw=0.55, alpha=0.8)
    ax.legend(loc="upper right", fontsize=7)
    save(fig, "Fig11_primary_mPSI_paired_slope")
    import shutil
    shutil.copyfile(OUT / "Fig11_primary_mPSI_paired_slope.png", OUT / "primary_delta_mPSI_paired_slope.png")


if __name__ == "__main__":
    fig10()
    fig11()
    print("Saved publication figure exports to", OUT)
