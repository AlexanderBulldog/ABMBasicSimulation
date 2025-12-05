from __future__ import annotations

from pathlib import Path
from typing import Dict

try:
    import pandas as pd
except ImportError:  # pragma: no cover
    pd = None  # type: ignore

from .model import EconomyModel


def summarize_run(df, window: int = 20) -> Dict[str, float]:
    if pd is None:
        raise ImportError("pandas is required for summarize_run")
    tail = df.tail(window) if window and window > 0 else df
    metrics = [
        "Employment",
        "UnemploymentRate",
        "Output",
        "Consumption",
        "AvgPrice",
        "HH_Debt",
        "HH_Deposit",
        "Firm_Debt",
        "Bank_Equity",
        "Defaults",
    ]
    return {f"{m}_mean": float(tail[m].mean()) for m in metrics if m in tail}


def run_demo():
    model = EconomyModel(seed=1, n_households=80, n_firms=8, enable_credit=True, price_elasticity=2.0)
    model.run_model(steps=50)
    df = model.results_dataframe()
    print(df.head())
    print("\nFinal aggregates:")
    print(df.tail(1))


def run_scenarios(
    steps: int = 100,
    window: int = 40,
    plot: bool = False,
    save: bool = False,
    output_prefix: str = "scenario",
    output_dir: str = "output",
):
    out_dir = Path(output_dir)
    if save:
        out_dir.mkdir(parents=True, exist_ok=True)

    scenarios = {
        "baseline": dict(alpha_mean=0.85, alpha_std=0.05),
        "credit_crunch": dict(bank_credit_multiplier=3.0, loan_rate=0.06, deposit_rate=0.01, alpha_mean=0.8),
        "demand_shock": dict(alpha_mean=0.6, alpha_std=0.1),
        "cost_shock": dict(wage=1.2, productivity=0.9),
        "credit_expansion": dict(bank_credit_multiplier=10.0, loan_rate=0.015, deposit_rate=0.0, alpha_mean=0.9),
    }
    results = []
    dfs = {}
    for name, params in scenarios.items():
        m = EconomyModel(seed=1, n_households=80, n_firms=8, **params)
        m.run_model(steps=steps)
        df = m.results_dataframe()
        dfs[name] = df
        if save and pd is not None:
            df.to_csv(out_dir / f"{output_prefix}_{name}.csv", index=True)
        summary = summarize_run(df, window=window)
        summary["scenario"] = name
        results.append(summary)

    if pd is None:
        print("pandas not available; skipping summary table")
    else:
        table = pd.DataFrame(results).set_index("scenario")
        print(table)
        if save:
            table.to_csv(out_dir / f"{output_prefix}_summary.csv", index=True)
            concat = []
            for name, df in dfs.items():
                d = df.copy()
                d["scenario"] = name
                concat.append(d)
            pd.concat(concat, axis=0).to_csv(out_dir / f"{output_prefix}_timeseries.csv", index=True)

    if plot:
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib not available; skipping plots")
            return
        fig, axes = plt.subplots(2, 2, figsize=(10, 6))
        keys = list(dfs.keys())
        for name in keys:
            df = dfs[name]
            axes[0, 0].plot(df["Employment"], label=name)
            axes[0, 1].plot(df["Output"], label=name)
            axes[1, 0].plot(df["AvgPrice"], label=name)
            axes[1, 1].plot(df["HH_Debt"], label=name)
        axes[0, 0].set_title("Employment")
        axes[0, 1].set_title("Output")
        axes[1, 0].set_title("AvgPrice")
        axes[1, 1].set_title("HH Debt")
        for ax in axes.flat:
            ax.legend()
        fig.tight_layout()
        if save:
            fig.savefig(out_dir / f"{output_prefix}_plot.png", dpi=150)
        else:
            plt.show()
