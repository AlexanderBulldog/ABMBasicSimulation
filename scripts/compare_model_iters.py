from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


def _to_bool(v) -> bool:
    if isinstance(v, bool):
        return v
    if v is None:
        return False
    return str(v).strip().lower() in {"1", "true", "yes", "y"}


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _load_iter_metrics(root: Path) -> Dict[str, float | str]:
    w1 = _read_csv(root / "01_wave1" / "lhs_runs.csv")
    w2 = _read_csv(root / "02_wave2" / "lhs_runs_wave2.csv")
    hm2 = _read_csv(root / "02_wave2" / "history_matching.csv")
    sa2 = _read_csv(root / "02_wave2" / "sensitivity_uncertainty.csv")
    gate = _read_csv(root / "04_master" / "research_core_tables" / "quality_gates.csv")

    confirm_path = root / "04_master" / "confirmatory_summary.json"
    confirm = {}
    if confirm_path.exists():
        confirm = json.loads(confirm_path.read_text(encoding="utf-8"))

    nroy_pct = float(hm2["nroy"].map(_to_bool).mean() * 100.0) if "nroy" in hm2.columns and len(hm2) else np.nan
    imax_median = float(hm2["I_max"].median()) if "I_max" in hm2.columns and len(hm2) else np.nan
    bad_w1 = float(w1["bad_run"].map(_to_bool).mean() * 100.0) if "bad_run" in w1.columns and len(w1) else np.nan
    bad_w2 = float(w2["bad_run"].map(_to_bool).mean() * 100.0) if "bad_run" in w2.columns and len(w2) else np.nan
    zero_emp_w2 = float((w2["Employment_mean"] <= 1e-9).mean() * 100.0) if "Employment_mean" in w2.columns and len(w2) else np.nan
    zero_out_w2 = float((w2["Output_mean"] <= 1e-9).mean() * 100.0) if "Output_mean" in w2.columns and len(w2) else np.nan

    md_share = np.nan
    md_rank = np.nan
    ou_share = np.nan
    if len(sa2):
        md = sa2[sa2["component"] == "MD"]
        ou = sa2[sa2["component"] == "OU"]
        if len(md):
            md_share = float(md["share_at_max_reduction"].iloc[0])
            md_rank = float(md["rank_at_max_reduction"].iloc[0])
        if len(ou):
            ou_share = float(ou["share_at_max_reduction"].iloc[0])

    price_fail_share = np.nan
    if len(hm2) and "nroy" in hm2.columns:
        I_cols = [c for c in hm2.columns if c.endswith("_I")]
        non = hm2[~hm2["nroy"].map(_to_bool)].copy()
        if len(non) and I_cols:
            arr = non[I_cols].to_numpy(dtype=float)
            argmax = np.nanargmax(arr, axis=1)
            drivers = [I_cols[i][:-2] for i in argmax]
            price_fail_share = float(sum(1 for d in drivers if d == "AvgPrice_mean") / len(drivers))

    confirm_delta = float(confirm.get("nroy_delta_pp", np.nan))
    confirm_top2 = bool(confirm.get("sa_top2_stable", False)) if confirm else False

    legacy_pass = np.nan
    if len(gate) and "pass" in gate.columns:
        legacy_pass = float(gate["pass"].map(_to_bool).all())

    return {
        "iter_name": root.name,
        "nroy_pct": nroy_pct,
        "imax_median": imax_median,
        "bad_run_pct_w1": bad_w1,
        "bad_run_pct_w2": bad_w2,
        "zero_employment_pct_w2": zero_emp_w2,
        "zero_output_pct_w2": zero_out_w2,
        "md_share_at_max_reduction": md_share,
        "md_rank_at_max_reduction": md_rank,
        "ou_share_at_max_reduction": ou_share,
        "price_fail_driver_share": price_fail_share,
        "confirmatory_nroy_delta_pp": confirm_delta,
        "confirmatory_top2_stable": float(confirm_top2),
        "legacy_all_gates_pass": legacy_pass,
    }


def _composite_score(row: pd.Series, baseline: pd.Series) -> float:
    score = 0.0
    if np.isfinite(row["md_share_at_max_reduction"]) and np.isfinite(baseline["md_share_at_max_reduction"]):
        score += 2.0 * (baseline["md_share_at_max_reduction"] - row["md_share_at_max_reduction"])
    if np.isfinite(row["imax_median"]) and np.isfinite(baseline["imax_median"]):
        score += 0.5 * (baseline["imax_median"] - row["imax_median"])
    if np.isfinite(row["bad_run_pct_w2"]) and np.isfinite(baseline["bad_run_pct_w2"]):
        score += 0.1 * (baseline["bad_run_pct_w2"] - row["bad_run_pct_w2"])
    if np.isfinite(row["price_fail_driver_share"]) and np.isfinite(baseline["price_fail_driver_share"]):
        score += 1.0 * (baseline["price_fail_driver_share"] - row["price_fail_driver_share"])
    if np.isfinite(row["confirmatory_nroy_delta_pp"]):
        score -= 0.05 * abs(row["confirmatory_nroy_delta_pp"])
    if not bool(row["confirmatory_top2_stable"]):
        score -= 0.5
    if not bool(row["legacy_all_gates_pass"]):
        score -= 1.0
    return float(score)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare structural ABM tuning iterations.")
    p.add_argument("--iters", type=str, required=True, help="Comma-separated output dirs to compare.")
    p.add_argument("--baseline", type=str, required=True, help="Baseline output dir, must be included in --iters.")
    p.add_argument("--out-csv", type=str, default="reports/model_tuning/summary_leaderboard.csv")
    p.add_argument("--out-md", type=str, default="reports/model_tuning/summary_leaderboard.md")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    iter_dirs = [Path(x.strip()) for x in args.iters.split(",") if x.strip()]
    rows: List[Dict[str, float | str]] = [_load_iter_metrics(p) for p in iter_dirs]
    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit("No iterations found to compare")

    baseline_name = Path(args.baseline).name
    if baseline_name not in set(df["iter_name"].astype(str)):
        raise SystemExit(f"Baseline {baseline_name} not present in --iters")
    base = df[df["iter_name"] == baseline_name].iloc[0]

    df["composite_score"] = df.apply(lambda r: _composite_score(r, base), axis=1)
    df = df.sort_values("composite_score", ascending=False).reset_index(drop=True)
    df["retain_recommendation"] = np.where(df["composite_score"] > 0, "retain", "revert")

    out_csv = Path(args.out_csv)
    out_md = Path(args.out_md)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)

    cols = [
        "iter_name",
        "composite_score",
        "retain_recommendation",
        "nroy_pct",
        "imax_median",
        "bad_run_pct_w2",
        "md_share_at_max_reduction",
        "ou_share_at_max_reduction",
        "price_fail_driver_share",
        "confirmatory_nroy_delta_pp",
        "confirmatory_top2_stable",
        "legacy_all_gates_pass",
    ]
    view = df[cols].copy()
    lines = ["# Model Tuning Leaderboard\n\n", view.to_markdown(index=False), "\n"]
    out_md.write_text("".join(lines), encoding="utf-8")
    print(f"Saved {out_csv}")
    print(f"Saved {out_md}")


if __name__ == "__main__":
    main()
