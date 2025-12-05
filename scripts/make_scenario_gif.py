from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.animation import FuncAnimation

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from synpop_model import EconomyModel

# Compact, high-contrast scenario (200 steps total)
params = dict(
    seed=11,
    n_households=100,
    n_firms=8,
    wage=1.2,
    productivity=1.0,
    alpha_mean=0.9,
    alpha_std=0.05,
    household_config={
        "savings": {"mode": "lognormal", "mu": 0.0, "sigma": 0.5, "scale": 0.4},
        "reserve_wage": {"base": 0.9, "spread": 0.15},
        "skill": {"mu": 1.0, "sigma": 0.15},
    },
    firm_config={
        "productivity": {"mode": "lognormal", "mu": 0, "sigma": 0.08, "scale": 1.0},
        "cash": {"mode": "lognormal", "mu": 0.0, "sigma": 0.4, "scale": 1.5},
        "market_shares": [0.25, 0.2, 0.15, 0.15, 0.1, 0.07, 0.05, 0.03],
    },
    controls={
        "households": {"total_savings": 15.0},
        "firms": {"total_cash": 10.0},
    },
    skill_wage_weight=0.4,
    loan_rate=0.08,
    repayment_fraction=0.1,
    adaptation_rate=1.0,
    initial_employment_rate=0.9,
)

model = EconomyModel(**params)
# Baseline 50 steps
model.run_model(steps=50)
df_base = model.results_dataframe()

# Negative shock: collapse propensity to consume
for h in model.households:
    h.alpha = max(0.05, h.alpha - 0.85)
model.run_model(steps=50)
df_shock = model.results_dataframe().iloc[50:]

# Positive shock: overshoot above baseline
for h in model.households:
    h.alpha = min(0.99, h.alpha + 0.9)
model.run_model(steps=100)
df_recovery = model.results_dataframe().iloc[100:]

full_df = pd.concat([df_base, df_shock, df_recovery], ignore_index=True)

metrics = ["Employment", "Output", "Firm_Debt"]
fig, ax = plt.subplots(figsize=(8, 4))
colors = {"Employment": "tab:blue", "Output": "tab:green", "Firm_Debt": "tab:red"}
lines = {m: ax.plot([], [], label=m, color=colors[m])[0] for m in metrics}
y_max = max(full_df["Output"].max(), full_df["Employment"].max(), full_df["Firm_Debt"].max()) * 1.1
ax.set_xlim(0, len(full_df))
ax.set_ylim(0, y_max if y_max > 0 else 1)
ax.legend(loc="upper right")
ax.set_title("Demand shocks and recovery (compact)")
ax.set_xlabel("Step")
for x in [50, 100]:
    ax.axvline(x, color="gray", linestyle="--", alpha=0.4)

xdata = list(range(len(full_df)))


def init():
    for l in lines.values():
        l.set_data([], [])
    return lines.values()


def update(frame):
    for m in metrics:
        lines[m].set_data(xdata[: frame + 1], full_df[m].iloc[: frame + 1])
    return lines.values()


ani = FuncAnimation(fig, update, frames=len(full_df), init_func=init, blit=True)
output_dir = ROOT / "output"
output_dir.mkdir(exist_ok=True)
out_file = output_dir / "scenario.gif"
ani.save(out_file, writer="pillow", fps=10)
print("Saved", out_file, "with", len(full_df), "frames")
