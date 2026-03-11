#!/usr/bin/env python3
"""
Generate figures from real MuJoCo experiment data.
"""

import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT_DIR = Path("reports/figures")


def set_style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.grid": True,
        "grid.alpha": 0.3,
    })


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    set_style()

    with open("reports/mujoco_full_experiment.json") as f:
        data = json.load(f)

    morphologies = ["MyoLeg (80)", "MyoLeg+Abs (86)", "MyoLeg+Back (290)"]
    muscles = [80, 86, 290]

    # Extract metrics
    random_coord = []
    dep_coord = []
    random_disp = []
    dep_disp = []
    random_actvol = []
    dep_actvol = []
    random_actdiv = []
    dep_actdiv = []

    for m in morphologies:
        rk = f"{m}_Random"
        dk = f"{m}_DEP"
        if rk in data and dk in data:
            random_coord.append(data[rk]["summary"]["action_coordination"]["mean"])
            dep_coord.append(data[dk]["summary"]["action_coordination"]["mean"])
            random_disp.append(data[rk]["summary"]["displacement"]["mean"])
            dep_disp.append(data[dk]["summary"]["displacement"]["mean"])
            random_actvol.append(data[rk]["summary"]["activation_volume"]["mean"])
            dep_actvol.append(data[dk]["summary"]["activation_volume"]["mean"])
            random_actdiv.append(data[rk]["summary"]["action_diversity"]["mean"])
            dep_actdiv.append(data[dk]["summary"]["action_diversity"]["mean"])

    morph_labels = ["MyoLeg\n(80)", "MyoLeg+Abs\n(86)", "MyoLeg+Back\n(290)"]

    # Fig 1: Action Coordination comparison
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.2))
    x = np.arange(len(morphologies))
    width = 0.32

    metrics_data = [
        ("Action Coordination", random_coord, dep_coord),
        ("Displacement (m)", random_disp, dep_disp),
        ("Action Diversity", random_actdiv, dep_actdiv),
        ("Activation Volume", random_actvol, dep_actvol),
    ]

    for ax, (title, rand_vals, dep_vals) in zip(axes, metrics_data):
        ax.bar(x - width/2, rand_vals, width, label="Random", color="#d62728", alpha=0.8)
        ax.bar(x + width/2, dep_vals, width, label="DEP", color="#1f77b4", alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(morph_labels, fontsize=8)
        ax.set_title(title)
        ax.legend(fontsize=7)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "real_metrics_comparison.pdf")
    plt.close(fig)
    print("Saved real_metrics_comparison.pdf")

    # Fig 2: Coordination ratio (DEP/Random)
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ratios = [d/r for d, r in zip(dep_coord, random_coord)]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    bars = ax.bar(morph_labels, ratios, color=colors, alpha=0.85)
    for bar, ratio in zip(bars, ratios):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f"{ratio:.1f}x", ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.set_ylabel("DEP / Random Ratio")
    ax.set_title("Action Coordination Improvement (DEP vs Random)")
    ax.axhline(y=1, color="gray", linestyle="--", alpha=0.5)
    ax.set_ylim(0, max(ratios) * 1.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "coordination_improvement.pdf")
    plt.close(fig)
    print("Saved coordination_improvement.pdf")

    # Fig 3: DEP self-organization from validation data
    val_path = Path("reports/dep_validation.json")
    if val_path.exists():
        with open(val_path) as f:
            vdata = json.load(f)
        c_norms = vdata["self_org_c_norms"]
        action_std = vdata["self_org_action_std"]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.5))
        ax1.plot(c_norms, color="#2ca02c", linewidth=1.5)
        ax1.set_xlabel("Timestep")
        ax1.set_ylabel("||C||")
        ax1.set_title("Controller Matrix Evolution")
        ax2.plot(action_std, color="#ff7f0e", linewidth=1.5)
        ax2.set_xlabel("Timestep")
        ax2.set_ylabel("Action Std")
        ax2.set_title("Action Diversity Over Time")
        fig.tight_layout()
        fig.savefig(OUT_DIR / "dep_self_organization.pdf")
        plt.close(fig)
        print("Saved dep_self_organization.pdf (updated)")

    print("\nAll real figures generated.")


if __name__ == "__main__":
    main()
