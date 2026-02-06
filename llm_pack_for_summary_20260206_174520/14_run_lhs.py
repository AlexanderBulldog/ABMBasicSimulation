from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Sequence, Tuple
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
SRC = ROOT / "src"
for p in (str(SCRIPTS), str(SRC)):
    if p not in sys.path:
        sys.path.insert(0, p)

from run_operator import PARAM_BOUNDS, run_batch  # noqa: E402


def load_bounds_from_refined_csv(path: Path, kind: str = "p05p95") -> Dict[str, Tuple[float, float]]:
    df = pd.read_csv(path)
    out: Dict[str, Tuple[float, float]] = {}
    for _, row in df.iterrows():
        param = str(row["param"])
        if kind == "minmax":
            low = float(row["nroy_min"])
            high = float(row["nroy_max"])
        else:
            low = float(row["nroy_p05"])
            high = float(row["nroy_p95"])
        if param in PARAM_BOUNDS:
            base_low, base_high = PARAM_BOUNDS[param]
            low = max(base_low, min(low, base_high))
            high = max(base_low, min(high, base_high))
        if high <= low:
            continue
        out[param] = (low, high)
    # Fall back to defaults if something goes wrong.
    return out if len(out) >= 3 else dict(PARAM_BOUNDS)


def lhs_sample(
    bounds: Dict[str, Tuple[float, float]],
    n: int,
    seed: int = 0,
) -> List[Dict[str, float]]:
    """Simple Latin Hypercube sampler over given bounds."""
    rng = np.random.default_rng(seed)
    names = list(bounds.keys())
    draws = {}
    for name, (low, high) in bounds.items():
        cut = (rng.permutation(n) + rng.random(n)) / n
        draws[name] = low + cut * (high - low)
    samples: List[Dict[str, float]] = []
    for i in range(n):
        theta = {name: float(draws[name][i]) for name in names}
        samples.append(theta)
    return samples


def parse_args():
    parser = argparse.ArgumentParser(description="Run LHS design for the EconomyModel.")
    parser.add_argument("--n", type=int, default=200, help="Number of LHS samples")
    parser.add_argument("--seed", type=int, default=0, help="Seed for LHS sampler")
    parser.add_argument(
        "--seeds",
        type=str,
        default="0,1,2",
        help="Comma-separated seeds for model runs (e.g., 0,1,2)",
    )
    parser.add_argument("--steps", type=int, default=150, help="Steps per simulation")
    parser.add_argument("--window", type=int, default=30, help="Window for summarize_run")
    parser.add_argument(
        "--balance-ok-threshold",
        type=float,
        default=0.5,
        help="Minimum BalanceOK_share to consider run acceptable",
    )
    parser.add_argument(
        "--bounds-csv",
        type=str,
        default="",
        help="Optional refined bounds CSV (output/results/refined_intervals.csv) for wave-2 sampling",
    )
    parser.add_argument(
        "--bounds-kind",
        type=str,
        choices=["p05p95", "minmax"],
        default="p05p95",
        help="Which columns from refined CSV to use as bounds",
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["parquet", "csv"],
        default="parquet",
        help="Output format",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output/datasets/lhs_runs.parquet",
        help="Output file path",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    seeds: Sequence[int] = [int(x) for x in args.seeds.split(",") if x]
    bounds = dict(PARAM_BOUNDS)
    if args.bounds_csv:
        bounds = load_bounds_from_refined_csv(Path(args.bounds_csv), kind=args.bounds_kind)
    samples = lhs_sample(bounds, n=args.n, seed=args.seed)
    df = run_batch(
        samples,
        seeds=seeds,
        steps=args.steps,
        window=args.window,
        balance_ok_threshold=args.balance_ok_threshold,
    )
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if args.format == "parquet":
        df.to_parquet(out_path, index=False)
    else:
        df.to_csv(out_path, index=False)
    print(f"Saved {len(df)} rows to {out_path}")


if __name__ == "__main__":
    main()
