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
    p.add_argument("--min-consumption", type=float, default=50.0, help="Min Consumption_mean for report runs")
    p.add_argument("--max-bank-resolved-share", type=float, default=0.08, help="Max BankResolved_share allowed")
    p.add_argument("--max-bank-bailedout-share", type=float, default=0.02, help="Max BankBailedOut_share allowed")
    p.add_argument("--max-haircut-mean", type=float, default=0.03, help="Max BankResolutionHaircut_mean allowed")
    p.add_argument("--max-defaults-hh-rate", type=float, default=0.01, help="Max DefaultsHH_rate allowed")
    p.add_argument("--max-defaults-firm-rate", type=float, default=0.01, help="Max DefaultsFirm_rate allowed")
    p.add_argument("--min-hh-deposit", type=float, default=0.05, help="Min HH_Deposit_mean for report runs")
    p.add_argument(
        "--balance-ok-threshold",
        type=float,
        default=0.95,
        help="Minimum BalanceOK_share for full-length validation runs",
    )
    p.add_argument(
        "--no-relax",
        dest="relax_if_needed",
        action="store_false",
        help="Disable relaxing constraints if strict validation can't fill n points.",
    )
    p.set_defaults(relax_if_needed=True)
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


def _constraints_from_args(args):
    return {
        "min_employment": float(args.min_employment),
        "max_unemployment": float(args.max_unemployment),
        "min_output": float(args.min_output),
        "min_consumption": float(args.min_consumption),
        "max_bank_resolved_share": float(args.max_bank_resolved_share),
        "max_bank_bailedout_share": float(args.max_bank_bailedout_share),
        "max_haircut_mean": float(args.max_haircut_mean),
        "max_defaults_hh_rate": float(args.max_defaults_hh_rate),
        "max_defaults_firm_rate": float(args.max_defaults_firm_rate),
        "min_hh_deposit": float(args.min_hh_deposit),
        "balance_ok_threshold": float(args.balance_ok_threshold),
    }


def _passes_strict(summary: Dict[str, float], args) -> bool:
    return len(_strict_fail_reasons(summary, args)) == 0


def _strict_fail_reasons(summary: Dict[str, float], args) -> List[str]:
    c = _constraints_from_args(args)
    reasons: List[str] = []
    if summary.get("BalanceOK_share", 1.0) < c["balance_ok_threshold"]:
        reasons.append("balance_ok_below_threshold")
    if summary.get("Employment_mean", 0.0) < c["min_employment"]:
        reasons.append("employment_below_min")
    if summary.get("UnemploymentRate_mean", 1.0) > c["max_unemployment"]:
        reasons.append("unemployment_above_max")
    if summary.get("Output_mean", 0.0) < c["min_output"]:
        reasons.append("output_below_min")
    if summary.get("Consumption_mean", 0.0) < c["min_consumption"]:
        reasons.append("consumption_below_min")
    if summary.get("HH_Deposit_mean", 0.0) < c["min_hh_deposit"]:
        reasons.append("hh_deposit_below_min")
    if summary.get("DefaultsHH_rate", 0.0) > c["max_defaults_hh_rate"]:
        reasons.append("defaults_hh_above_max")
    if summary.get("DefaultsFirm_rate", 0.0) > c["max_defaults_firm_rate"]:
        reasons.append("defaults_firm_above_max")
    if summary.get("AvgPrice_mean", 0.0) <= 0.1:
        reasons.append("price_too_low")
    if summary.get("BankFailed_share", 0.0) > 0.0:
        reasons.append("bank_failed_positive")
    if summary.get("BankResolved_share", 0.0) > c["max_bank_resolved_share"]:
        reasons.append("bank_resolved_above_max")
    if summary.get("BankBailedOut_share", 0.0) > c["max_bank_bailedout_share"]:
        reasons.append("bank_bailedout_above_max")
    if summary.get("BankResolutionHaircut_mean", 0.0) > c["max_haircut_mean"]:
        reasons.append("haircut_above_max")
    return reasons


def _passes_relaxed(summary: Dict[str, float], args) -> bool:
    return len(_relaxed_fail_reasons(summary, args)) == 0


def _relaxed_fail_reasons(summary: Dict[str, float], args) -> List[str]:
    # Minimal viability constraints to avoid "collapse" trajectories.
    reasons: List[str] = []
    if summary.get("BalanceOK_share", 1.0) < min(0.9, float(args.balance_ok_threshold)):
        reasons.append("balance_ok_below_relaxed_threshold")
    if summary.get("Employment_mean", 0.0) <= 1e-9:
        reasons.append("employment_nonpositive")
    if summary.get("Output_mean", 0.0) <= 1e-9:
        reasons.append("output_nonpositive")
    if summary.get("Consumption_mean", 0.0) <= 1e-9:
        reasons.append("consumption_nonpositive")
    if summary.get("AvgPrice_mean", 0.0) <= 0.1:
        reasons.append("price_too_low")
    if summary.get("BankFailed_share", 0.0) > 0.0:
        reasons.append("bank_failed_positive")
    return reasons


def _theta_from_row(row: pd.Series) -> Dict[str, float]:
    theta: Dict[str, float] = {}
    for p in PARAM_BOUNDS:
        theta[p] = float(row[p])
    return theta


def pick_representative_candidates(df: pd.DataFrame, n: int, args) -> pd.DataFrame:
    df = df.copy()
    df["score"] = df.apply(objective_score, axis=1)

    # Candidate pool based on short-run dataset, for fast pruning.
    constraints = (
        (df.get("bad_run", False) == False)  # noqa: E712
        & (df.get("AvgPrice_mean", 0.0) > 0.1)
        & (df.get("Employment_mean", 0.0) > 0.0)
        & (df.get("Output_mean", 0.0) > 0.0)
        & (df.get("Consumption_mean", 0.0) > 0.0)
    )
    ok = df[constraints].copy()
    if ok.empty:
        ok = df[df.get("bad_run", False) == False].copy()  # noqa: E712

    # Diversity along employment + score (short-run).
    ok = ok.sort_values("Employment_mean")
    bins = np.linspace(
        float(ok["Employment_mean"].min()),
        float(ok["Employment_mean"].max()) + 1e-9,
        num=min(n, 6) + 1,
    )
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
        if len(chosen) >= max(n * 6, n + 12):
            break

    remainder = ok.loc[~ok.index.isin(used_idx)].sort_values("score").head(max(n * 6, n + 12) - len(chosen))
    chosen.extend([remainder.iloc[i] for i in range(len(remainder))])
    out = pd.DataFrame(chosen).reset_index(drop=True)
    return out


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
        df["nroy"] = df["nroy"].fillna(False).astype(bool)
        df = df[df["nroy"] == True].copy()  # noqa: E712
        if df.empty:
            raise SystemExit("No NROY rows after merge; check history_matching file and bounds.")

    # Keep unique parameter sets (avoid picking the same theta across different seeds).
    if "seed" in df.columns:
        df = df.sort_values("seed").drop_duplicates(subset=list(PARAM_BOUNDS.keys()), keep="first").reset_index(drop=True)

    candidates = pick_representative_candidates(df, n=args.n, args=args)
    seeds = [int(x) for x in args.seeds.split(",") if x]

    accepted_rows: List[pd.Series] = []
    accepted_runs: List[Tuple[int, int, Dict[str, float], pd.DataFrame]] = []  # (rep_id, seed, summary, df)
    rejection_rows: List[Dict[str, object]] = []

    strict_first = True
    while True:
        for _, cand in candidates.iterrows():
            if len(accepted_rows) >= args.n:
                break
            theta = _theta_from_row(cand)
            # Run full-length simulation per seed ONCE and keep the trajectories if accepted.
            run_cache: List[Tuple[int, Dict[str, float], pd.DataFrame]] = []
            ok = True
            for seed in seeds:
                summary, run_df = run_model(theta, seed=seed, steps=args.steps, window=args.window)
                if strict_first:
                    reasons = _strict_fail_reasons(summary, args)
                else:
                    reasons = _relaxed_fail_reasons(summary, args)
                ok = len(reasons) == 0
                if not ok:
                    rejection_rows.append(
                        {
                            "stage": "strict" if strict_first else "relaxed",
                            "seed": seed,
                            "reasons": ";".join(reasons),
                            **{p: float(theta[p]) for p in PARAM_BOUNDS},
                        }
                    )
                    break
                run_cache.append((seed, summary, run_df))

            if not ok:
                continue

            rep_id = len(accepted_rows)
            accepted_rows.append(cand)
            for seed, summary, run_df in run_cache:
                accepted_runs.append((rep_id, seed, summary, run_df))

        if len(accepted_rows) >= args.n:
            break
        if not args.relax_if_needed or not strict_first:
            break
        strict_first = False

    if len(accepted_rows) < args.n:
        if rejection_rows:
            pd.DataFrame(rejection_rows).to_csv(outdir / "representative_rejections.csv", index=False)
        raise SystemExit(
            f"Could not validate enough representative points ({len(accepted_rows)}/{args.n}). "
            "Try lowering thresholds (e.g., --min-employment/--min-output) or allow relaxed fallback (default, disable via --no-relax)."
        )

    picked = pd.DataFrame(accepted_rows).reset_index(drop=True)
    picked["validated_steps"] = int(args.steps)
    picked["validated_seeds"] = ",".join([str(s) for s in seeds])
    points_path = outdir / "representative_points.csv"
    picked.to_csv(points_path, index=False)

    summaries: List[Dict[str, float]] = []
    for rep_id, seed, summary, run_df in accepted_runs:
        theta = {k: float(picked.loc[rep_id, k]) for k in PARAM_BOUNDS.keys()}
        row = {**theta, **summary, "rep_id": rep_id, "seed": seed}
        summaries.append(row)
        run_df.to_csv(outdir / f"rep_run_{rep_id:02d}_seed{seed}.csv", index=True)

    summary_path = outdir / "representative_summary.csv"
    pd.DataFrame(summaries).to_csv(summary_path, index=False)
    if rejection_rows:
        pd.DataFrame(rejection_rows).to_csv(outdir / "representative_rejections.csv", index=False)
    print(f"Saved points to {points_path}")
    print(f"Saved run summaries to {summary_path}")


if __name__ == "__main__":
    main()
