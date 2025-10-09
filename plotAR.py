import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --------- INPUT (update paths if needed) ---------
PPO_PATH = '/home/LABAUT/alberto_vaglio/Desktop/NewFolder/PPO.csv'
SAC_PATH = '/home/LABAUT/alberto_vaglio/Desktop/NewFolder/SAC.csv'
TQC_PATH = '/home/LABAUT/alberto_vaglio/Desktop/NewFolder/TQC.csv'

LABELS = {
    "PPO": PPO_PATH,
    "SAC": SAC_PATH,
    "TQC": TQC_PATH,
}

COLORS = {"PPO": "orange", "SAC": "green", "TQC": "blue"}
OUTPUT = "success_rate_PPO_SAC_TQC.png"

# Cap the x-axis at 10M steps (set to None to keep full range)
X_MAX_CAP = 10_000_000  # or None

# --------- HELPERS ---------
def load_steps_and_success(csv_path: str) -> pd.DataFrame:
    """Load CSV and return a DataFrame with columns: Step (int), SR (float, in %)."""
    df = pd.read_csv(csv_path)
    cols = {c.lower(): c for c in df.columns}

    # Candidate column names
    step_candidates = ["step", "steps", "global_step", "timestep", "timesteps"]
    sr_candidates = [
        "success_rate", "sr", "success", "eval/success_rate", "charts/success_rate",
        "metric_success_rate", "successrate", "success%", "succ_rate"
    ]

    # Pick step and success columns if present
    step_col = next((cols[c] for c in step_candidates if c in cols), None)
    sr_col   = next((cols[c] for c in sr_candidates if c in cols), None)

    # Fallbacks: guess from numeric columns
    if step_col is None or sr_col is None:
        numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        # Try a column name containing 'succ' for success
        if sr_col is None:
            for c in df.columns:
                if "succ" in c.lower():
                    sr_col = c
                    break
        if step_col is None and numeric_cols:
            step_col = numeric_cols[0]
        if sr_col is None and len(numeric_cols) >= 2:
            sr_col = numeric_cols[-1]

    out = df[[step_col, sr_col]].dropna().rename(columns={step_col: "Step", sr_col: "SR"})
    out = out.sort_values("Step")

    # Convert SR to percentage if it looks like 0..1
    if out["SR"].between(0, 1).all():
        out["SR"] = out["SR"] * 100.0

    # Clean bounds
    out["SR"] = out["SR"].clip(0, 100)

    # Optional step cap
    if X_MAX_CAP is not None:
        out = out[out["Step"] <= X_MAX_CAP]

    return out

# --------- LOAD DATA ---------
series = {name: load_steps_and_success(path) for name, path in LABELS.items()}

# --------- PLOT ---------
plt.figure(figsize=(7, 9))
for name in ["PPO", "SAC", "TQC"]:
    d = series[name]
    if not d.empty:
        plt.plot(d["Step"], d["SR"], label=name, color=COLORS[name])

plt.xlabel("Steps", fontsize=24)
plt.ylabel("Success rate (%)", fontsize=24)
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
plt.legend(fontsize=24)
plt.title("Success rate during training", fontsize=26)
plt.grid(True)

# If we capped, enforce visible x-limit
if X_MAX_CAP is not None:
    left = min(s["Step"].min() for s in series.values() if not s.empty)
    plt.xlim(left=left, right=X_MAX_CAP)

plt.tight_layout()
plt.savefig(OUTPUT, dpi=150)
print(f"Saved: {OUTPUT}")
