#!/usr/bin/env python3
"""
Generate figures for the final report from training log CSVs.

Reads per-epoch training data from:
  data/trained_models/<morphology>/<exp_name>_training_log.csv

Produces PDF figures in reports/figures/.
"""

import csv
import json
import os
import sys
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

OUT_DIR = Path("reports/figures")
LOG_BASE = Path("data/trained_models")


def set_style():
    plt.rcParams.update({
        "font.family": "serif", "font.size": 10,
        "axes.titlesize": 11, "axes.labelsize": 10,
        "xtick.labelsize": 9, "ytick.labelsize": 9,
        "legend.fontsize": 9, "figure.dpi": 150,
        "savefig.dpi": 300, "savefig.bbox": "tight",
        "axes.grid": True, "grid.alpha": 0.3,
    })


def read_training_log(csv_path):
    """Read a training log CSV into dict of lists."""
    data = {}
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            for key, val in row.items():
                if key not in data:
                    data[key] = []
                try:
                    data[key].append(float(val))
                except (ValueError, TypeError):
                    data[key].append(None)
    return data


def smooth(y, window=7):
    """Moving average smoothing."""
    y = np.array(y, dtype=float)
    kernel = np.ones(window) / window
    return np.convolve(y, kernel, mode="same")


def plot_learning_curves():
    """Fig 1: MPJPE learning curves."""
    morphologies = [
        ("MyoLeg (80 muscles)", "legs", "ppo_baseline_legs", "ppo_dep_legs"),
        ("MyoLeg+Abs (86 muscles)", "legs_abs", "ppo_baseline_legs_abs", "ppo_dep_legs_abs"),
        ("MyoLeg+Back (290 muscles)", "legs_back", "ppo_baseline_legs_back", "ppo_dep_legs_back"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), sharey=True)

    for ax, (title, morph_dir, baseline_name, dep_name) in zip(axes, morphologies):
        bl_path = LOG_BASE / morph_dir / baseline_name / f"{baseline_name}_training_log.csv"
        dep_path = LOG_BASE / morph_dir / dep_name / f"{dep_name}_training_log.csv"

        if not bl_path.exists() or not dep_path.exists():
            ax.set_title(f"{title}\n(no data)")
            continue

        bl = read_training_log(bl_path)
        dep = read_training_log(dep_path)

        bl_mpjpe = smooth(np.array(bl["mpjpe"]) * 1000)
        dep_mpjpe = smooth(np.array(dep["mpjpe"]) * 1000)
        epochs = np.array(bl["epoch"])

        ax.plot(epochs, bl_mpjpe, color="#d62728", alpha=0.85, linewidth=1.5, label="PPO baseline")
        ax.plot(epochs, dep_mpjpe, color="#1f77b4", alpha=0.85, linewidth=1.5, label="PPO + DEP")
        ax.axhline(y=100, color="gray", linestyle="--", alpha=0.5, linewidth=0.8)
        ax.set_xlabel("Epoch")
        ax.set_title(title)
        ax.set_xlim(0, max(epochs))
        ax.legend(loc="upper right", framealpha=0.8)

    axes[0].set_ylabel("MPJPE (mm)")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "learning_curves.pdf")
    plt.close(fig)
    print("  learning_curves.pdf")


def plot_bar_charts():
    """Fig 2: Final metrics comparison."""
    configs = [
        ("80", "legs", "ppo_baseline_legs", "ppo_dep_legs"),
        ("86", "legs_abs", "ppo_baseline_legs_abs", "ppo_dep_legs_abs"),
        ("290", "legs_back", "ppo_baseline_legs_back", "ppo_dep_legs_back"),
    ]

    metrics_ppo = {"mpjpe": [], "joint_angle_rmse": [], "activation_volume": [], "frame_coverage": []}
    metrics_dep = {"mpjpe": [], "joint_angle_rmse": [], "activation_volume": [], "frame_coverage": []}

    for label, morph_dir, bl_name, dep_name in configs:
        bl = read_training_log(LOG_BASE / morph_dir / bl_name / f"{bl_name}_training_log.csv")
        dep = read_training_log(LOG_BASE / morph_dir / dep_name / f"{dep_name}_training_log.csv")
        last_n = 20
        for key, ppo_list, dep_list in [
            ("mpjpe", metrics_ppo, metrics_dep),
            ("joint_angle_rmse", metrics_ppo, metrics_dep),
            ("activation_volume", metrics_ppo, metrics_dep),
            ("frame_coverage", metrics_ppo, metrics_dep),
        ]:
            ppo_vals = [v for v in bl[key][-last_n:] if v is not None]
            dep_vals = [v for v in dep[key][-last_n:] if v is not None]
            ppo_list[key].append(np.mean(ppo_vals) * (1000 if key == "mpjpe" else 1))
            dep_list[key].append(np.mean(dep_vals) * (1000 if key == "mpjpe" else 1))

    labels = ["MyoLeg\n(80)", "MyoLeg+Abs\n(86)", "MyoLeg+Back\n(290)"]
    titles = ["MPJPE (mm)", "Joint RMSE (rad)", "Activation Vol.", "Frame Coverage"]
    keys = ["mpjpe", "joint_angle_rmse", "activation_volume", "frame_coverage"]

    fig, axes = plt.subplots(1, 4, figsize=(14, 3.2))
    x = np.arange(3)
    w = 0.32

    for ax, title, key in zip(axes, titles, keys):
        ax.bar(x - w/2, metrics_ppo[key], w, label="PPO", color="#d62728", alpha=0.8)
        ax.bar(x + w/2, metrics_dep[key], w, label="PPO+DEP", color="#1f77b4", alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_title(title)
        ax.legend(fontsize=7, loc="upper left")

    fig.tight_layout()
    fig.savefig(OUT_DIR / "metrics_comparison.pdf")
    plt.close(fig)
    print("  metrics_comparison.pdf")


def plot_sample_efficiency():
    """Fig 3: Epochs to reach 100mm MPJPE."""
    configs = [
        ("MyoLeg (80)", "legs", "ppo_baseline_legs", "ppo_dep_legs"),
        ("MyoLeg+Abs (86)", "legs_abs", "ppo_baseline_legs_abs", "ppo_dep_legs_abs"),
        ("MyoLeg+Back (290)", "legs_back", "ppo_baseline_legs_back", "ppo_dep_legs_back"),
    ]

    def find_threshold_epoch(data, threshold_mm=100):
        mpjpe_mm = np.array(data["mpjpe"]) * 1000
        smoothed = smooth(mpjpe_mm, window=10)
        below = np.where(smoothed < threshold_mm)[0]
        return int(data["epoch"][below[0]]) if len(below) > 0 else None

    fig, ax = plt.subplots(figsize=(6, 3.5))
    morph_names = []
    ppo_vals = []
    dep_vals = []

    for name, morph_dir, bl_name, dep_name in configs:
        bl = read_training_log(LOG_BASE / morph_dir / bl_name / f"{bl_name}_training_log.csv")
        dep = read_training_log(LOG_BASE / morph_dir / dep_name / f"{dep_name}_training_log.csv")
        morph_names.append(name)
        ppo_ep = find_threshold_epoch(bl)
        dep_ep = find_threshold_epoch(dep)
        ppo_vals.append(ppo_ep if ppo_ep else 600)
        dep_vals.append(dep_ep if dep_ep else 600)

    x = np.arange(len(morph_names))
    w = 0.32
    bars1 = ax.bar(x - w/2, ppo_vals, w, label="PPO baseline", color="#d62728", alpha=0.8)
    bars2 = ax.bar(x + w/2, dep_vals, w, label="PPO + DEP", color="#1f77b4", alpha=0.8)

    for bars in [bars1, bars2]:
        for bar in bars:
            h = bar.get_height()
            label = f"{int(h)}" if h < 600 else ">500"
            ax.text(bar.get_x() + bar.get_width()/2, h + 8, label,
                    ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(morph_names, fontsize=9)
    ax.set_ylabel("Epochs to MPJPE < 100mm")
    ax.set_title("Sample Efficiency")
    ax.legend()
    ax.set_ylim(0, 700)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "sample_efficiency.pdf")
    plt.close(fig)
    print("  sample_efficiency.pdf")


def plot_exploration_comparison():
    """Fig 4: Random vs DEP from MuJoCo experiment."""
    exp_path = Path("reports/mujoco_full_experiment.json")
    if not exp_path.exists():
        print("  SKIP exploration comparison (no experiment data)")
        return

    with open(exp_path) as f:
        data = json.load(f)

    morphologies = ["MyoLeg (80)", "MyoLeg+Abs (86)", "MyoLeg+Back (290)"]
    morph_labels = ["MyoLeg\n(80)", "MyoLeg+Abs\n(86)", "MyoLeg+Back\n(290)"]

    metrics_list = [
        ("Action Coordination", "action_coordination"),
        ("Displacement (m)", "displacement"),
        ("Action Diversity", "action_diversity"),
        ("Activation Volume", "activation_volume"),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(14, 3.2))
    x = np.arange(len(morphologies))
    w = 0.32

    for ax, (title, key) in zip(axes, metrics_list):
        rand_vals = [data[f"{m}_Random"]["summary"][key]["mean"] for m in morphologies]
        dep_vals = [data[f"{m}_DEP"]["summary"][key]["mean"] for m in morphologies]
        ax.bar(x - w/2, rand_vals, w, label="Random", color="#d62728", alpha=0.8)
        ax.bar(x + w/2, dep_vals, w, label="DEP", color="#1f77b4", alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(morph_labels, fontsize=8)
        ax.set_title(title)
        ax.legend(fontsize=7)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "real_metrics_comparison.pdf")
    plt.close(fig)
    print("  real_metrics_comparison.pdf")


def plot_coordination_ratio():
    """Fig 5: DEP/Random coordination ratio."""
    exp_path = Path("reports/mujoco_full_experiment.json")
    if not exp_path.exists():
        return

    with open(exp_path) as f:
        data = json.load(f)

    morphologies = ["MyoLeg (80)", "MyoLeg+Abs (86)", "MyoLeg+Back (290)"]
    labels = ["MyoLeg\n(80)", "MyoLeg+Abs\n(86)", "MyoLeg+Back\n(290)"]
    ratios = []
    for m in morphologies:
        r = data[f"{m}_Random"]["summary"]["action_coordination"]["mean"]
        d = data[f"{m}_DEP"]["summary"]["action_coordination"]["mean"]
        ratios.append(d / r)

    fig, ax = plt.subplots(figsize=(6, 3.5))
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    bars = ax.bar(labels, ratios, color=colors, alpha=0.85)
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
    print("  coordination_improvement.pdf")


def plot_self_organization():
    """Fig 6: DEP C-matrix evolution."""
    val_path = Path("reports/dep_validation.json")
    if not val_path.exists():
        return

    with open(val_path) as f:
        data = json.load(f)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.5))
    ax1.plot(data["self_org_c_norms"], color="#2ca02c", linewidth=1.5)
    ax1.set_xlabel("Timestep")
    ax1.set_ylabel("||C||")
    ax1.set_title("Controller Matrix Evolution")
    ax2.plot(data["self_org_action_std"], color="#ff7f0e", linewidth=1.5)
    ax2.set_xlabel("Timestep")
    ax2.set_ylabel("Action Std")
    ax2.set_title("Action Diversity Over Time")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "dep_self_organization.pdf")
    plt.close(fig)
    print("  dep_self_organization.pdf")


def main():
    os.chdir(str(Path(__file__).resolve().parents[1]))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    set_style()

    print("Generating report figures from training logs...")
    plot_learning_curves()
    plot_bar_charts()
    plot_sample_efficiency()
    plot_exploration_comparison()
    plot_coordination_ratio()
    plot_self_organization()

    figs = sorted(OUT_DIR.glob("*.pdf"))
    print(f"\n{len(figs)} figures in {OUT_DIR}/:")
    for f in figs:
        print(f"  {f.name}")


if __name__ == "__main__":
    main()
