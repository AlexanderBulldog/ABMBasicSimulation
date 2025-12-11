from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from synpop import EconomyModel  # noqa: E402
from synpop.scenarios import summarize_run  # noqa: E402

DEFAULT_STEPS = 150
DEFAULT_WINDOW = 30

PARAM_BOUNDS: Dict[str, Tuple[float, float]] = {
    "alpha_mean": (0.1, 0.95),
    "alpha_std": (0.01, 0.2),
    "wage": (0.5, 2.0),
    "productivity": (0.5, 2.5),
    "loan_rate": (0.005, 0.15),
    "deposit_rate": (0.0, 0.06),
    "bank_credit_multiplier": (2.0, 12.0),
    "hh_debt_cap_multiplier": (2.0, 8.0),
    "firm_debt_cap_multiplier": (1.0, 5.0),
    "adaptation_rate": (0.1, 1.0),
    "price_elasticity": (0.5, 4.0),
    "skill_wage_weight": (0.0, 0.8),
    "demand_smoothing": (0.0, 0.5),
}


def run_model(
    theta: Dict[str, float],
    seed: int = 0,
    steps: int = DEFAULT_STEPS,
    window: int = DEFAULT_WINDOW,
):
    """Run the ABM for a single parameter set and return summary + raw dataframe."""
    model = EconomyModel(seed=seed, **theta)
    model.run_model(steps=steps)
    df = model.results_dataframe()
    summary = summarize_run(df, window=window)
    summary["DefaultsHH_rate"] = float(df["DefaultsHH"].mean()) if "DefaultsHH" in df else np.nan
    summary["DefaultsFirm_rate"] = float(df["DefaultsFirm"].mean()) if "DefaultsFirm" in df else np.nan
    summary["BalanceOK_share"] = float(df["BalanceOK"].mean()) if "BalanceOK" in df else np.nan
    summary["seed"] = seed
    return summary, df


def _has_bad_values(df: pd.DataFrame) -> bool:
    """Detect NaN/inf in numeric model outputs."""
    numeric = df.select_dtypes(include=[np.number])
    if numeric.empty:
        return False
    return not np.isfinite(numeric.to_numpy()).all()


def run_batch(
    samples: Sequence[Dict[str, float]],
    seeds: Sequence[int] = (0, 1, 2),
    steps: int = DEFAULT_STEPS,
    window: int = DEFAULT_WINDOW,
    balance_ok_threshold: float = 0.5,
) -> pd.DataFrame:
    """Run multiple parameter sets and seeds; return aggregated rows."""
    rows: List[Dict[str, float]] = []
    for theta in samples:
        for seed in seeds:
            row: Dict[str, float] = {**theta, "seed": seed}
            try:
                summary, df = run_model(theta, seed=seed, steps=steps, window=window)
                row.update(summary)
                bad = _has_bad_values(df) or (summary.get("BalanceOK_share", 1.0) < balance_ok_threshold)
                row["bad_run"] = bool(bad)
                row["error"] = ""
            except Exception as exc:  # pragma: no cover - guardrail for batch runs
                row["error"] = str(exc)
                row["bad_run"] = True
            rows.append(row)
    return pd.DataFrame(rows)


def midpoint_params(bounds: Dict[str, Tuple[float, float]] = PARAM_BOUNDS) -> Dict[str, float]:
    """Convenience helper: midpoint of each bound."""
    return {k: 0.5 * (low + high) for k, (low, high) in bounds.items()}


def demo_run() -> pd.DataFrame:
    """Small smoke-test run on midpoint parameters and a single seed."""
    sample = midpoint_params()
    df = run_batch([sample], seeds=(0,), steps=50, window=20)
    return df


if __name__ == "__main__":
    out = demo_run()
    print(out.head())
