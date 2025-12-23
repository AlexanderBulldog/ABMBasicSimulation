from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys
from typing import Dict, Iterable, List, Tuple

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPTS = ROOT / "scripts"
for p in (str(SRC), str(SCRIPTS)):
    if p not in sys.path:
        sys.path.insert(0, p)

from synpop import EconomyModel  # noqa: E402
from run_operator import PARAM_BOUNDS  # noqa: E402


FEATURES = [
    "inventory",
    "last_demand",
    "cash",
    "debt",
    "price",
    "n_workers",
    "bank_available_credit",
    "wage",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a firm-level policy dataset from representative runs."
    )
    parser.add_argument(
        "--runs-dir",
        type=str,
        default="output/results/report_set",
        help="Directory with rep_run_XX_seedY.csv files.",
    )
    parser.add_argument(
        "--summary",
        type=str,
        default="",
        help="Path to representative_summary.csv (defaults to runs-dir).",
    )
    parser.add_argument(
        "--tail",
        type=int,
        default=150,
        help="Use the last N steps from each trajectory.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="output/ml_datasets/firm_policy.csv",
        help="Output CSV path.",
    )
    return parser.parse_args()


def _parse_rep_info(path: Path) -> Tuple[int, int]:
    match = re.match(r"rep_run_(\d+)_seed(\d+)\.csv", path.name)
    if not match:
        raise ValueError(f"Unrecognized run filename: {path.name}")
    return int(match.group(1)), int(match.group(2))


def _load_summary(summary_path: Path) -> pd.DataFrame:
    df = pd.read_csv(summary_path)
    required = ["rep_id", "seed", *PARAM_BOUNDS.keys()]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise SystemExit(f"Missing columns in {summary_path}: {', '.join(missing)}")
    return df


def _theta_from_summary(df: pd.DataFrame, rep_id: int, seed: int) -> Dict[str, float]:
    row = df[(df["rep_id"] == rep_id) & (df["seed"] == seed)]
    if row.empty:
        raise KeyError(f"No summary row for rep_id={rep_id}, seed={seed}")
    series = row.iloc[0]
    return {k: float(series[k]) for k in PARAM_BOUNDS.keys()}


def _simulate_and_collect(
    model: EconomyModel,
    steps: int,
    capture_start: int,
    rep_id: int,
    seed: int,
) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []
    base_wage = float(model.wage)

    for step in range(steps):
        for h in model.households:
            h.begin_step()
        for f in model.firms:
            f.begin_step()

        model.bank.accrue_interest(model.households, model.firms)
        model._labor_market()

        total_production = 0.0
        total_wage_bill = 0.0
        total_overhead = 0.0
        for firm in model.firms:
            workers = [model._household_by_id(wid) for wid in firm.workers]
            workers = [w for w in workers if w is not None]
            wages = [model._wage_for_worker(w) for w in workers]

            if step >= capture_start:
                bank_credit = float(model.bank.available_credit())
                state = {
                    "inventory": float(firm.inventory),
                    "last_demand": float(firm.last_demand),
                    "cash": float(firm.cash),
                    "debt": float(firm.debt),
                    "price": float(firm.price),
                    "n_workers": float(len(firm.workers)),
                    "bank_available_credit": bank_credit,
                    "wage": base_wage,
                }

            firm.set_price(base_wage)
            paid, output, overhead = firm.produce_and_pay(wages, model.bank)

            if step >= capture_start:
                unit_cost = base_wage / max(firm.productivity, 1e-6)
                realized_markup = firm.price / max(unit_cost, 1e-6) - 1.0
                rows.append(
                    {
                        **state,
                        "realized_markup": float(realized_markup),
                        "step": int(step),
                        "rep_id": int(rep_id),
                        "seed": int(seed),
                        "firm_id": int(firm.unique_id),
                    }
                )

            total_wage_bill += paid
            total_production += output
            total_overhead += overhead

            wage_bill = float(sum(wages))
            pay_ratio = 1.0 if wage_bill <= 0 else max(0.0, min(1.0, paid / wage_bill))
            for worker, wage_amt in zip(workers, wages):
                worker.receive_wage(wage_amt * pay_ratio, firm.unique_id)

        if total_overhead > 0 and model.households:
            per_hh = total_overhead / max(len(model.households), 1)
            for h in model.households:
                h.deposit += per_hh
        model._last_transfers = total_overhead
        model._last_production = total_production
        model._last_wage_bill = total_wage_bill

        model._consumption_market()
        model._handle_defaults()
        model.bank.update_balance_sheet(model.households, model.firms)
        model._check_balance()

    return rows


def build_dataset(
    runs_dir: Path,
    summary_path: Path,
    tail: int,
    out_path: Path,
) -> None:
    run_files = sorted(runs_dir.glob("rep_run_*_seed*.csv"))
    if not run_files:
        raise SystemExit(f"No rep_run files found in {runs_dir}")

    summary_df = _load_summary(summary_path)
    rows: List[Dict[str, float]] = []

    for run_file in run_files:
        rep_id, seed = _parse_rep_info(run_file)
        theta = _theta_from_summary(summary_df, rep_id, seed)

        run_df = pd.read_csv(run_file)
        steps = int(len(run_df))
        if steps <= 0:
            continue
        tail_steps = min(int(tail), steps)
        capture_start = steps - tail_steps

        model = EconomyModel(seed=int(seed), **theta)
        if hasattr(model, "use_ml_policy"):
            model.use_ml_policy = False
        rows.extend(_simulate_and_collect(model, steps, capture_start, rep_id, seed))

    if not rows:
        raise SystemExit("No rows were collected; check run files and parameters.")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df = pd.DataFrame(rows)
    out_df.to_csv(out_path, index=False)
    print(f"Saved {len(out_df)} rows to {out_path}")


def main() -> None:
    args = parse_args()
    runs_dir = Path(args.runs_dir)
    summary_path = Path(args.summary) if args.summary else runs_dir / "representative_summary.csv"
    out_path = Path(args.out)

    build_dataset(runs_dir=runs_dir, summary_path=summary_path, tail=args.tail, out_path=out_path)


if __name__ == "__main__":
    main()
