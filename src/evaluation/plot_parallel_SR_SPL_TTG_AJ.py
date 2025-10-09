import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Percorsi CSV per parallel traffic (scenario_id == 14)
PPO_CSV = "EVAL_METRICS/social_metrics_by_scenario_PPOnoumani_refine_PPO.csv"
SAC_CSV = "EVAL_METRICS/social_metrics_by_scenario_SACcurr3_SAC.csv"
TQC_CSV = "EVAL_METRICS/social_metrics_by_scenario_TQCsuper_TQC.csv"

METHODS = ["PPO", "SAC", "TQC"]
COLORS = {"PPO": "tab:orange", "SAC": "tab:green", "TQC": "tab:blue"}


# Carica i dati aggregati per SR e SPL
def load_agg_metrics(csv_path):
    df = pd.read_csv(csv_path)
    row = df[df["scenario_id"] == 14].iloc[0]
    return {
        "SR": row["success_rate"] * 100,
        "SPL": row["mean_SPL"] * 100
    }

# Carica i dati per-episodio per TTG e AJ
def load_per_episode(csv_path):
    df = pd.read_csv(csv_path)
    df = df[df["scenario_id"] == 14]
    return df["time_to_goal"].dropna().values, df["avg_angular_jerk"].dropna().values

SR = []
SPL = []
TTG = []
AJ = []

agg_files = [
    "EVAL_METRICS/social_metrics_by_scenario_PPOnoumani_refine_PPO.csv",
    "EVAL_METRICS/social_metrics_by_scenario_SACcurr3_SAC.csv",
    "EVAL_METRICS/social_metrics_by_scenario_TQCsuper_TQC.csv"
]
per_episode_files = [
    "EVAL_METRICS/social_metrics_PPOnoumani_refine_PPO.csv",
    "EVAL_METRICS/social_metrics_SACcurr3_SAC.csv",
    "EVAL_METRICS/social_metrics_TQCsuper_TQC.csv"
]

for agg, per_ep in zip(agg_files, per_episode_files):
    m = load_agg_metrics(agg)
    SR.append(m["SR"])
    SPL.append(m["SPL"])
    ttg, aj = load_per_episode(per_ep)
    TTG.append(ttg)
    AJ.append(aj)

fig, axes = plt.subplots(2, 2, figsize=(10, 7))

# Success Rate
axes[0, 0].bar(METHODS, SR, color=[COLORS[m] for m in METHODS])
axes[0, 0].set_ylabel("%", fontsize=15)
axes[0, 0].set_title("Success Rate (SR)", fontsize=16)
axes[0, 0].set_ylim(0, 105)
axes[0, 0].grid(axis="y", linestyle="--", alpha=0.6)

# SPL
axes[0, 1].bar(METHODS, SPL, color=[COLORS[m] for m in METHODS])
axes[0, 1].set_ylabel("%", fontsize=15)
axes[0, 1].set_title("Success weighted Path Length (SPL)", fontsize=16)
axes[0, 1].set_ylim(0, 105)
axes[0, 1].grid(axis="y", linestyle="--", alpha=0.6)

# TTG boxplot
bp_ttg = axes[1, 0].boxplot(TTG, patch_artist=True, labels=METHODS, showfliers=False)
for patch, m in zip(bp_ttg['boxes'], METHODS):
    patch.set_facecolor(COLORS[m])
axes[1, 0].set_ylabel("s", fontsize=15)
axes[1, 0].set_title("Time to Goal (TTG)", fontsize=16)
axes[1, 0].grid(axis="y", linestyle="--", alpha=0.6)

# AJ boxplot
bp_aj = axes[1, 1].boxplot(AJ, patch_artist=True, labels=METHODS, showfliers=False)
for patch, m in zip(bp_aj['boxes'], METHODS):
    patch.set_facecolor(COLORS[m])
axes[1, 1].set_ylabel("m/s^3", fontsize=15)
axes[1, 1].set_title("Angular Jerk (AJ)", fontsize=16)
axes[1, 1].grid(axis="y", linestyle="--", alpha=0.6)

plt.tight_layout()
plt.savefig("EVAL_METRICS/parallel_traffic_SR_SPL_TTG_AJ.png", dpi=180)
print("Salvato: EVAL_METRICS/parallel_traffic_SR_SPL_TTG_AJ.png")
