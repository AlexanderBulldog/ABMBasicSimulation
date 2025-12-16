from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
SRC = ROOT / "src"
for p in (str(SCRIPTS), str(SRC)):
    if p not in sys.path:
        sys.path.insert(0, p)

from run_operator import PARAM_BOUNDS, run_model  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description="Select a small representative set of runs for reporting.")
    p.add_argument("--data", type=str, default="output/datasets/lhs_runs.csv", help="CSV from run_lhs.py")
    p.add_argument("--outdir", type=str, default="output/results/report_set", help="Output directory")
    p.add_argument("--n", type=int, default=6, help="Number of representative points to select")
    p.add_argument("--steps", type=int, default=300, help="Steps for each representative run")
    p.add_argument("--window", type=int, default=50, help="Window used for summary metrics")
    p.add_argument("--seeds", type=str, default="0,1,2", help="Comma-separated seeds for representative runs")
    return p.parse_args()


def objective_score(row: pd.Series) -> float:
    # Heuristic "macro-plausible" targets (tunable for your report).
    targets = {
        "Employment_mean": 65.0,
        "Output_mean": 70.0,
        "Consumption_mean": 70.0,
        "Bank_Equity_mean": 50.0,
        "DefaultsHH_rate": 0.01,
        "DefaultsFirm_rate": 0.01,
        "BankFailed_share": 0.0,
        "BankResolved_share": 0.0,
    }
    scales = {
        "Employment_mean": 15.0,
        "Output_mean": 20.0,
        "Consumption_mean": 20.0,
        "Bank_Equity_mean": 50.0,
        "DefaultsHH_rate": 0.01,
        "DefaultsFirm_rate": 0.01,
        "BankFailed_share": 0.2,
        "BankResolved_share": 0.2,
    }
    score = 0.0
    for k, t in targets.items():
        if k not in row or pd.isna(row[k]):
            score += 10.0
            continue
        score += float(((row[k] - t) / scales[k]) ** 2)
    return score


def pick_representative(df: pd.DataFrame, n: int) -> pd.DataFrame:
    df = df.copy()
    df["score"] = df.apply(objective_score, axis=1)

    # Primary constraints for "report-ready" runs.
    constraints = (
        (df["bad_run"] == False)  # noqa: E712
        & (df["BankFailed_share"] <= 0.2)
        & (df["BankResolved_share"] <= 0.2)
        & (df["Employment_mean"] >= 40.0)
        & (df["Employment_mean"] <= 90.0)
        & (df["DefaultsHH_rate"] <= 0.03)
        & (df["DefaultsFirm_rate"] <= 0.05)
        & (df["AvgPrice_mean"] > 0.1)
    )
    ok = df[constraints].copy()
    if ok.empty:
        ok = df[df["bad_run"] == False].copy()  # noqa: E712

    # Ensure diversity along the most interpretable axis: employment.
    ok = ok.sort_values("Employment_mean")
    bins = np.linspace(float(ok["Employment_mean"].min()), float(ok["Employment_mean"].max()) + 1e-9, num=min(n, 6) + 1)
    chosen: List[pd.Series] = []
    used_idx = set()
    for lo, hi in zip(bins[:-1], bins[1:]):
        bucket = ok[(ok["Employment_mean"] >= lo) & (ok["Employment_mean"] < hi)].copy()
        if bucket.empty:
            continue
        best = bucket.sort_values("score").iloc[0]
        if best.name in used_idx:
            continue
        chosen.append(best)
        used_idx.add(best.name)
        if len(chosen) >= n:
            break

    if len(chosen) < n:
        remainder = ok.loc[~ok.index.isin(used_idx)].sort_values("score").head(n - len(chosen))
        chosen.extend([remainder.iloc[i] for i in range(len(remainder))])

    out = pd.DataFrame(chosen)
    out = out.sort_values(["score", "Employment_mean"]).head(n).reset_index(drop=True)
    return out


def theta_from_row(row: pd.Series) -> Dict[str, float]:
    theta: Dict[str, float] = {}
    for p in PARAM_BOUNDS:
        theta[p] = float(row[p])
    return theta


def main():
    args = parse_args()
    data_path = Path(args.data)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(data_path)
    # Keep unique parameter sets (avoid picking the same theta across different seeds).
    if "seed" in df.columns:
        df = df.sort_values("seed").drop_duplicates(subset=list(PARAM_BOUNDS.keys()), keep="first").reset_index(drop=True)

    picked = pick_representative(df, n=args.n)
    points_path = outdir / "representative_points.csv"
    picked.to_csv(points_path, index=False)

    seeds = [int(x) for x in args.seeds.split(",") if x]
    summaries: List[Dict[str, float]] = []
    for i in range(len(picked)):
        theta = theta_from_row(picked.iloc[i])
        for seed in seeds:
            summary, run_df = run_model(theta, seed=seed, steps=args.steps, window=args.window)
            row = {**theta, **summary, "rep_id": i, "seed": seed}
            summaries.append(row)
            run_df.to_csv(outdir / f"rep_run_{i:02d}_seed{seed}.csv", index=True)

    summary_path = outdir / "representative_summary.csv"
    pd.DataFrame(summaries).to_csv(summary_path, index=False)
    print(f"Saved points to {points_path}")
    print(f"Saved run summaries to {summary_path}")


if __name__ == "__main__":
    main()

