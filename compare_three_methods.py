#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
compare_three_methods.py

Reads THREE per-episode CSVs (one for TQC, PPO, SAC) and produces THREE 2×3 figures
—one per scenario—where the x-axis lists the methods (TQC, PPO, SAC).

Top row  : Success Rate (%), SPL (%), Space Compliance (%)
Bottom   : Path length (box, successes), Time-to-Goal (box, successes),
           Angular Jerk (box, all episodes)

Expected per-episode CSV columns (as saved by your evaluator):
  episode, scenario_id, result, path_len, spl, space_compliance,
  avg_angular_jerk, time_to_goal

Usage example:
  python3 compare_three_methods.py \
    --tqc EVAL_METRICS/social_metrics_per_episode_TQCsuper_TQC.csv \
    --ppo EVAL_METRICS/social_metrics_per_episode_PPOnoumani_refine_PPO.csv \
    --sac EVAL_METRICS/social_metrics_per_episode_SACcurr3_SAC.csv \
    --outdir EVAL_METRICS --dpi 200
"""
import os
import argparse
from typing import List, Dict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from matplotlib.patches import Patch

# Pretty names for known scenarios (fallbacks to the numeric id)
SCEN_NAME = {
    14: "parallel traffic",
    15: "intersection",
    16: "perpendicular traffic",
}

# Colors for each trainer
METHOD_COLORS: Dict[str, str] = {
    "TQC": "tab:blue",
    "PPO": "tab:orange",
    "SAC": "tab:green",
}

REQUIRED = {
    "episode", "scenario_id", "result", "path_len", "spl",
    "space_compliance", "avg_angular_jerk", "time_to_goal"
}

def scen_label(sid):
    try:
        sid_i = int(sid)
    except Exception:
        return str(sid)
    return SCEN_NAME.get(sid_i, str(sid_i))

def load_csv(path: str, method: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"{method}: not found → {path}")
    df = pd.read_csv(path)
    miss = REQUIRED - set(df.columns)
    if miss:
        raise ValueError(f"{method}: CSV missing columns: {sorted(miss)}")
    df["trainer"] = method.upper()
    return df

def color_boxplot(bp, colors_by_index: List[str]):
    """Color a matplotlib boxplot (created with patch_artist=True) with a color per box."""
    for i, box in enumerate(bp["boxes"]):
        box.set(facecolor=colors_by_index[i], alpha=0.6, edgecolor="black", linewidth=1.0)
    for med in bp["medians"]:
        med.set(color="black", linewidth=1.2)
    for w in bp["whiskers"]:
        w.set(color="black", linewidth=1.0)
    for c in bp["caps"]:
        c.set(color="black", linewidth=1.0)

def make_fig_for_scenario(df_all: pd.DataFrame, scenario_id: int, methods: List[str], outdir: str, dpi: int):
    scen_name = scen_label(scenario_id).title()
    df_s = df_all[df_all["scenario_id"] == scenario_id]

    # Bars (%)
    sr_vals, spl_vals, sc_vals = [], [], []
    # Boxes
    pl_box, ttg_box, aj_box, sc_box = [], [], [], []

    for m in methods:
        d = df_s[df_s["trainer"] == m]
        # percentages
        sr_vals.append((d["result"].eq("success").mean() * 100.0) if len(d) else 0.0)
        spl_vals.append((d["spl"].mean() * 100.0) if len(d) else 0.0)
        sc_vals.append((d["space_compliance"].mean() * 100.0) if len(d) else 0.0)
        # box data
        pl_box.append(d.loc[d["result"].eq("success"), "path_len"].dropna().values)
        ttg_box.append(d.loc[d["result"].eq("success"), "time_to_goal"].dropna().values)
        aj_box.append(d["avg_angular_jerk"].dropna().values)
        sc_box.append((d["space_compliance"].dropna().values) * 100.0)

    colors = [METHOD_COLORS.get(m, "tab:gray") for m in methods]

    fig, axes = plt.subplots(2, 3, figsize=(22, 11), constrained_layout=True)
    # Big figure title = scenario name
    fig.suptitle(scen_name, fontsize=27, fontweight="bold")
    fig.subplots_adjust(top=0.90, left=0.16, right=0.90, bottom=0.08)

    # Legend (consistent across panels)
    legend_handles = [Patch(facecolor=METHOD_COLORS.get(m, "tab:gray"), edgecolor="black", label=m, alpha=0.6)
                      for m in methods]

    # (A) Success Rate — bars
    axes[0,0].bar(methods, sr_vals, color=colors, edgecolor="black", width=0.55)
    axes[0,0].set_ylabel("SR (%)")
    axes[0,0].set_ylim(0, 100)
    axes[0,0].yaxis.set_major_formatter(PercentFormatter(100))
    axes[0,0].grid(axis="y", linestyle="--", alpha=0.7)

    # (B) SPL — bars
    axes[0,1].bar(methods, spl_vals, color=colors, edgecolor="black", width=0.55)
    axes[0,1].set_ylabel("SPL (%)")
    axes[0,1].set_ylim(0, 100)
    axes[0,1].yaxis.set_major_formatter(PercentFormatter(100))
    axes[0,1].grid(axis="y", linestyle="--", alpha=0.7)

    # (C) Space Compliance — box (all episodes)
    if any(len(a) > 0 for a in sc_box):
        bp = axes[0,2].boxplot(sc_box, tick_labels=methods, showfliers=False, patch_artist=True)
        color_boxplot(bp, colors)
        axes[0,2].set_ylabel("SC (%)")
        axes[0,2].set_ylim(0, 100)
        axes[0,2].yaxis.set_major_formatter(PercentFormatter(100))
        axes[0,2].grid(axis="y", linestyle="--", alpha=0.7)
    else:
        axes[0,2].axis("off")

    # (D) Path length — box (successes)
    if any(len(a) > 0 for a in pl_box):
        bp = axes[1,0].boxplot(pl_box, tick_labels=methods, showfliers=False, patch_artist=True)
        color_boxplot(bp, colors)
        axes[1,0].set_ylabel("PL (m)")
        axes[1,0].grid(axis="y", linestyle="--", alpha=0.7)
    else:
        axes[1,0].axis("off")

    # (E) Time-to-Goal — box (successes)
    if any(len(a) > 0 for a in ttg_box):
        bp = axes[1,1].boxplot(ttg_box, tick_labels=methods, showfliers=False, patch_artist=True)
        color_boxplot(bp, colors)
        axes[1,1].set_ylabel("TTG (s)")
        axes[1,1].grid(axis="y", linestyle="--", alpha=0.7)
    else:
        axes[1,1].axis("off")

    # (F) Angular jerk — box (all episodes)
    if any(len(a) > 0 for a in aj_box):
        bp = axes[1,2].boxplot(aj_box, tick_labels=methods, showfliers=False, patch_artist=True)
        color_boxplot(bp, colors)
        axes[1,2].set_ylabel(r"AJ (rad/s$^3$)")
        axes[1,2].grid(axis="y", linestyle="--", alpha=0.7)
    else:
        axes[1,2].axis("off")

    # Cosmetics: only tweak spacing a bit
    for ax in axes.flatten():
        ax.yaxis.labelpad = 8

    # Optional legend (single place)
    # axes[0,2].legend(handles=legend_handles, loc="upper right", frameon=True)

    os.makedirs(outdir, exist_ok=True)
    out = os.path.join(outdir, f"comparison_6panels_by_method_{scen_label(scenario_id).replace(' ', '_')}.png")
    plt.savefig(out, dpi=dpi)
    plt.close(fig)
    print(f"Saved: {out}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tqc", type=str, required=True, help="Per-episode CSV for TQC.")
    ap.add_argument("--ppo", type=str, required=True, help="Per-episode CSV for PPO.")
    ap.add_argument("--sac", type=str, required=True, help="Per-episode CSV for SAC.")
    ap.add_argument("--outdir", type=str, default="EVAL_METRICS", help="Output directory.")
    ap.add_argument("--dpi", type=int, default=180, help="Figure DPI.")
    args = ap.parse_args()

    plt.rcParams.update({
        "font.size": 24,
        "axes.labelsize": 24,
        "xtick.labelsize": 24,
        "ytick.labelsize": 24,
        "legend.fontsize": 24,
    })

    df_tqc = load_csv(args.tqc, "TQC")
    df_ppo = load_csv(args.ppo, "PPO")
    df_sac = load_csv(args.sac, "SAC")

    df_all = pd.concat([df_tqc, df_ppo, df_sac], ignore_index=True)

    methods = ["PPO", "SAC", "TQC"]   # order controls x-axis and colors
    scen_ids = sorted(df_all["scenario_id"].dropna().unique().tolist())
    if not scen_ids:
        raise ValueError("No scenario_id values found in the provided CSVs.")

    os.makedirs(args.outdir, exist_ok=True)
    for sid in scen_ids:
        make_fig_for_scenario(df_all, int(sid), methods, args.outdir, args.dpi)

if __name__ == "__main__":
    main()
