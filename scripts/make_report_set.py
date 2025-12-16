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
    p.add_argument(
        "--history-matching",
        type=str,
        default="",
        help="Optional history_matching.csv; if provided, selects points from NROY subset",
    )
    p.add_argument("--outdir", type=str, default="output/results/report_set", help="Output directory")
    p.add_argument("--n", type=int, default=6, help="Number of representative points to select")
    p.add_argument("--steps", type=int, default=300, help="Steps for each representative run")
    p.add_argument("--window", type=int, default=50, help="Window used for summary metrics")
    p.add_argument("--seeds", type=str, default="0,1,2", help="Comma-separated seeds for representative runs")
    p.add_argument("--min-employment", type=float, default=45.0, help="Min Employment_mean for report runs")
    p.add_argument("--max-unemployment", type=float, default=0.55, help="Max UnemploymentRate_mean for report runs")
    p.add_argument("--min-output", type=float, default=50.0, help="Min Output_mean for report runs")
    p.add_argument("--min-consumption", type=float, default=45.0, help="Min Consumption_mean for report runs")
    p.add_argument("--max-bank-resolved-share", type=float, default=0.08, help="Max BankResolved_share allowed")
    p.add_argument("--max-bank-bailedout-share", type=float, default=0.02, help="Max BankBailedOut_share allowed")
    p.add_argument("--max-haircut-mean", type=float, default=0.03, help="Max BankResolutionHaircut_mean allowed")
    p.add_argument("--max-defaults-hh-rate", type=float, default=0.01, help="Max DefaultsHH_rate allowed")
    p.add_argument("--max-defaults-firm-rate", type=float, default=0.01, help="Max DefaultsFirm_rate allowed")
    p.add_argument("--min-hh-deposit", type=float, default=0.05, help="Min HH_Deposit_mean for report runs")
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
        "BankResolved_share": 0.03,
        "BankBailedOut_share": 0.0,
        "BankResolutionHaircut_mean": 0.02,
        "HH_Deposit_mean": 1.0,
    }
    scales = {
        "Employment_mean": 15.0,
        "Output_mean": 20.0,
        "Consumption_mean": 20.0,
        "Bank_Equity_mean": 50.0,
        "DefaultsHH_rate": 0.01,
        "DefaultsFirm_rate": 0.01,
        "BankFailed_share": 0.2,
        "BankResolved_share": 0.05,
        "BankBailedOut_share": 0.05,
        "BankResolutionHaircut_mean": 0.03,
        "HH_Deposit_mean": 2.0,
    }
    score = 0.0
    for k, t in targets.items():
        if k not in row or pd.isna(row[k]):
            score += 10.0
            continue
        score += float(((row[k] - t) / scales[k]) ** 2)
    return score


def pick_representative(df: pd.DataFrame, n: int, args) -> pd.DataFrame:
    df = df.copy()
    df["score"] = df.apply(objective_score, axis=1)

    # Primary constraints for "report-ready" runs.
    constraints = (
        (df["bad_run"] == False)  # noqa: E712
        & (df.get("BankFailed_share", 0.0) <= 0.0)
        & (df.get("BankResolved_share", 0.0) <= args.max_bank_resolved_share)
        & (df.get("BankBailedOut_share", 0.0) <= args.max_bank_bailedout_share)
        & (df.get("BankResolutionHaircut_mean", 0.0) <= args.max_haircut_mean)
        & (df["Employment_mean"] >= args.min_employment)
        & (df.get("UnemploymentRate_mean", 1.0) <= args.max_unemployment)
        & (df.get("Output_mean", 0.0) >= args.min_output)
        & (df.get("Consumption_mean", 0.0) >= args.min_consumption)
        & (df.get("HH_Deposit_mean", 0.0) >= args.min_hh_deposit)
        & (df.get("DefaultsHH_rate", 0.0) <= args.max_defaults_hh_rate)
        & (df.get("DefaultsFirm_rate", 0.0) <= args.max_defaults_firm_rate)
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
    if args.history_matching:
        hm = pd.read_csv(Path(args.history_matching))
        key_cols = list(PARAM_BOUNDS.keys())
        keep_cols = list(dict.fromkeys(key_cols + ["nroy", "I_max"]))
        hm = hm[keep_cols]
        df = df.merge(hm, on=key_cols, how="left")
        df["nroy"] = df["nroy"].fillna(False)
        df = df[df["nroy"] == True].copy()  # noqa: E712
        if df.empty:
            raise SystemExit("No NROY rows after merge; check history_matching file and bounds.")

    # Keep unique parameter sets (avoid picking the same theta across different seeds).
    if "seed" in df.columns:
        df = df.sort_values("seed").drop_duplicates(subset=list(PARAM_BOUNDS.keys()), keep="first").reset_index(drop=True)

    picked = pick_representative(df, n=args.n, args=args)
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
