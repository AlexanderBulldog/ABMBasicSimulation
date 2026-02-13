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
    # Tuned to reduce collapse/insolvency regimes and improve representativeness of sampled runs.
    # These bounds are intentionally "safer" than the wide defaults, to make calibration/reporting
    # runs more stable (seminar/paper demo).
    "alpha_mean": (0.57, 0.92),
    "alpha_std": (0.03, 0.13),
    "wage": (0.9, 1.45),
    "productivity": (0.95, 1.95),
    "loan_rate": (0.012, 0.073),
    "deposit_rate": (0.001, 0.028),
    "bank_credit_multiplier": (4.2, 11.5),
    "hh_debt_cap_multiplier": (2.1, 4.8),
    "firm_debt_cap_multiplier": (1.6, 4.8),
    "adaptation_rate": (0.25, 0.95),
    "price_elasticity": (0.95, 2.9),
    "skill_wage_weight": (0.03, 0.49),
    "demand_smoothing": (0.06, 0.35),
    # Structural block: price dynamics.
    "price_adjust_speed": (0.08, 0.45),
    "price_stickiness": (0.45, 0.90),
    "max_price_step": (0.04, 0.22),
    "inventory_target_days": (0.8, 2.8),
    # Structural block: planning + labor frictions.
    "demand_expectation_memory": (0.15, 0.65),
    "max_hire_per_step": (3.0, 12.0),
    "max_fire_per_step": (3.0, 12.0),
    # Structural block: credit underwriting and firm resolution.
    "hh_max_dsr": (0.20, 0.65),
    "firm_max_dsr": (0.60, 2.20),
    "firm_reentry_lag": (0.0, 4.0),
    "firm_reentry_cash_fraction": (0.20, 0.85),
    # Structural block: household smoothing.
    "consumption_memory": (0.15, 0.75),
    "precautionary_saving": (0.03, 0.35),
    "consumption_floor_prop": (0.00, 0.35),
    "consumption_ceiling_prop": (1.00, 2.80),
    "unemployed_consumption_penalty": (0.05, 0.45),
    "unemployment_transfer": (0.00, 0.25),
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
    # Convert default counts per step into shares (0..1) for comparability across population sizes.
    summary["DefaultsHH_rate"] = (
        float(df["DefaultsHH"].mean()) / max(model.n_households, 1) if "DefaultsHH" in df else np.nan
    )
    summary["DefaultsFirm_rate"] = (
        float(df["DefaultsFirm"].mean()) / max(model.n_firms, 1) if "DefaultsFirm" in df else np.nan
    )
    summary["BalanceOK_share"] = float(df["BalanceOK"].mean()) if "BalanceOK" in df else np.nan
    summary["BankFailed_share"] = float(df["BankFailed"].mean()) if "BankFailed" in df else 0.0
    summary["BankResolved_share"] = float(df["BankResolved"].mean()) if "BankResolved" in df else 0.0
    summary["BankResolutionHaircut_mean"] = (
        float(df["BankResolutionHaircut"].mean()) if "BankResolutionHaircut" in df else 0.0
    )
    summary["BankBailedOut_share"] = float(df["BankBailedOut"].mean()) if "BankBailedOut" in df else 0.0
    summary["BankBailoutAmount_mean"] = float(df["BankBailoutAmount"].mean()) if "BankBailoutAmount" in df else 0.0
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
    employment_floor: float = 1e-9,
    output_floor: float = 1e-9,
    price_floor: float = 0.1,
) -> pd.DataFrame:
    """Run multiple parameter sets and seeds; return aggregated rows."""
    rows: List[Dict[str, float]] = []
    for theta in samples:
        for seed in seeds:
            row: Dict[str, float] = {**theta, "seed": seed}
            try:
                summary, df = run_model(theta, seed=seed, steps=steps, window=window)
                row.update(summary)
                employment = float(summary.get("Employment_mean", np.inf))
                output = float(summary.get("Output_mean", np.inf))
                price = float(summary.get("AvgPrice_mean", np.inf))
                bank_equity = float(summary.get("Bank_Equity_mean", 0.0))

                pathological = (
                    (not np.isfinite(employment))
                    or (not np.isfinite(output))
                    or (not np.isfinite(price))
                    or (employment <= employment_floor)
                    or (output <= output_floor)
                    or (price <= price_floor)
                    or (not np.isfinite(bank_equity))
                )

                bad = (
                    _has_bad_values(df)
                    or pathological
                    or (summary.get("BalanceOK_share", 1.0) < balance_ok_threshold)
                )
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
