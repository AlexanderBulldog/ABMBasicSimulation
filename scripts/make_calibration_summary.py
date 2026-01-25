from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser(description="Generate a concise calibration summary (wave1 vs wave2).")
    p.add_argument("--lhs1", type=str, default="output/datasets/lhs_runs.csv")
    p.add_argument("--hm1", type=str, default="output/results/history_matching.csv")
    p.add_argument("--ri1", type=str, default="output/results/refined_intervals.csv")
    p.add_argument("--lhs2", type=str, default="output/datasets/lhs_runs_wave2.csv")
    p.add_argument("--hm2", type=str, default="output/results/wave2/history_matching.csv")
    p.add_argument("--ri2", type=str, default="output/results/wave2/refined_intervals.csv")
    p.add_argument("--scores1", type=str, default="output/results/emulator_scores.csv")
    p.add_argument("--out", type=str, default="output/results/calibration_summary.md")
    return p.parse_args()


def qline(series: pd.Series, qs=(0.0, 0.05, 0.5, 0.95, 1.0)) -> str:
    q = series.quantile(list(qs))
    return ", ".join([f"{int(p*100)}%={q[p]:.4g}" for p in qs])


def shares(lhs: pd.DataFrame) -> Dict[str, float]:
    def share(cond) -> float:
        return float(cond.mean()) * 100.0

    return {
        "Employment_mean==0": share(lhs["Employment_mean"] <= 1e-12),
        "Output_mean==0": share(lhs["Output_mean"] <= 1e-12),
        "Consumption_mean==0": share(lhs["Consumption_mean"] <= 1e-12),
        "HH_Deposit_mean==0": share(lhs["HH_Deposit_mean"] <= 1e-12),
        "BankFailed_share>0": share(lhs.get("BankFailed_share", 0.0) > 0),
        "BankResolved_share>0": share(lhs.get("BankResolved_share", 0.0) > 0),
        "BankBailedOut_share>0": share(lhs.get("BankBailedOut_share", 0.0) > 0),
    }


def bad_run_share(lhs: pd.DataFrame) -> float:
    if "bad_run" not in lhs.columns:
        return 0.0
    # bad_run is serialized as bool in parquet, and as strings in CSV; handle both.
    col = lhs["bad_run"]
    if col.dtype == bool:
        return float(col.mean()) * 100.0
    return float((col.astype(str).str.lower() == "true").mean()) * 100.0


def good_runs(lhs: pd.DataFrame) -> pd.DataFrame:
    if "bad_run" not in lhs.columns:
        return lhs
    col = lhs["bad_run"]
    if col.dtype == bool:
        return lhs[lhs["bad_run"] == False].copy()  # noqa: E712
    return lhs[col.astype(str).str.lower() != "true"].copy()


def hm_stats(hm: pd.DataFrame) -> Tuple[float, float, float, float]:
    nroy = float(hm["nroy"].mean()) * 100.0
    med = float(hm["I_max"].median())
    p95 = float(hm["I_max"].quantile(0.95))
    mx = float(hm["I_max"].max())
    return nroy, med, p95, mx


def interval_stats(ri: pd.DataFrame) -> Tuple[float, float]:
    return float(ri["shrink_pct"].mean()), float(ri["shrink_pct"].max())


def emulator_stats(scores: pd.DataFrame) -> Tuple[int, int, float, float]:
    trained = scores[~scores["skipped"]].copy()
    return int(len(trained)), int(scores["skipped"].sum()), float(trained["gpr_r2_cv_mean"].mean()), float(trained["gpr_r2_cv_mean"].median())


def main():
    args = parse_args()
    lhs1 = pd.read_csv(Path(args.lhs1))
    hm1 = pd.read_csv(Path(args.hm1))
    ri1 = pd.read_csv(Path(args.ri1))
    lhs2 = pd.read_csv(Path(args.lhs2))
    hm2 = pd.read_csv(Path(args.hm2))
    ri2 = pd.read_csv(Path(args.ri2))
    scores1 = pd.read_csv(Path(args.scores1))

    s1 = shares(lhs1)
    s2 = shares(lhs2)
    g1 = good_runs(lhs1)
    g2 = good_runs(lhs2)
    s1g = shares(g1)
    s2g = shares(g2)
    bad1 = bad_run_share(lhs1)
    bad2 = bad_run_share(lhs2)
    nroy1, med1, p951, mx1 = hm_stats(hm1)
    nroy2, med2, p952, mx2 = hm_stats(hm2)
    mean_sh1, max_sh1 = interval_stats(ri1)
    mean_sh2, max_sh2 = interval_stats(ri2)
    n_tr, n_sk, r2_mean, r2_med = emulator_stats(scores1)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    lines = []
    lines.append("# Calibration summary (wave1 vs wave2)\n")
    lines.append("## Emulator quality (wave1)\n")
    lines.append(f"- Trained metrics: {n_tr}, skipped: {n_sk}\n")
    lines.append(f"- GPR CV R2 mean: {r2_mean:.3f}, median: {r2_med:.3f}\n\n")

    lines.append("## History matching (report targets)\n")
    lines.append(f"- Wave1 NROY: {nroy1:.1f}% | I_max median={med1:.3f}, p95={p951:.3f}, max={mx1:.3f}\n")
    lines.append(f"- Wave2 NROY: {nroy2:.1f}% | I_max median={med2:.3f}, p95={p952:.3f}, max={mx2:.3f}\n\n")

    lines.append("## Interval shrink (refined_intervals.csv)\n")
    lines.append(f"- Wave1 shrink mean={mean_sh1:.2f}%, max={max_sh1:.2f}%\n")
    lines.append(f"- Wave2 shrink mean={mean_sh2:.2f}%, max={max_sh2:.2f}%\n\n")

    lines.append("## Model-pathology shares (raw LHS datasets)\n")
    lines.append("| Metric | Wave1 | Wave2 |\n")
    lines.append("|---|---:|---:|\n")
    for k in s1:
        lines.append(f"| {k} | {s1[k]:.1f}% | {s2[k]:.1f}% |\n")
    lines.append("\n")

    lines.append("## Data quality (bad_run-filtered)\n")
    lines.append(f"- Wave1 bad_run share: {bad1:.1f}% ({len(g1)}/{len(lhs1)} kept)\n")
    lines.append(f"- Wave2 bad_run share: {bad2:.1f}% ({len(g2)}/{len(lhs2)} kept)\n\n")
    lines.append("### Pathologies among good runs only\n")
    lines.append("| Metric | Wave1(good) | Wave2(good) |\n")
    lines.append("|---|---:|---:|\n")
    for k in s1g:
        lines.append(f"| {k} | {s1g[k]:.1f}% | {s2g[k]:.1f}% |\n")
    lines.append("\n")

    lines.append("## Core distribution snapshots (wave2)\n")
    for c in ["Employment_mean", "Output_mean", "Consumption_mean", "AvgPrice_mean", "DefaultsHH_rate", "BankResolutionHaircut_mean", "BankBailedOut_share"]:
        if c in lhs2.columns:
            lines.append(f"- `{c}`: {qline(lhs2[c])}\n")
    lines.append("\n")

    out.write_text("".join(lines), encoding="utf-8")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
