from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


def _to_bool(s) -> bool:
    if isinstance(s, bool):
        return s
    if s is None:
        return False
    return str(s).strip().lower() in {"1", "true", "yes", "y"}


def _hm_stats(df: pd.DataFrame) -> Dict[str, float]:
    if df.empty:
        return {"rows": 0, "nroy_pct": np.nan, "Imax_median": np.nan, "Imax_p95": np.nan, "Imax_max": np.nan, "metrics_used_mean": np.nan}
    nroy = df["nroy"].map(_to_bool) if "nroy" in df.columns else pd.Series([False] * len(df))
    return {
        "rows": float(len(df)),
        "nroy_pct": float(nroy.mean() * 100.0),
        "Imax_median": float(df["I_max"].median()) if "I_max" in df.columns else np.nan,
        "Imax_p95": float(df["I_max"].quantile(0.95)) if "I_max" in df.columns else np.nan,
        "Imax_max": float(df["I_max"].max()) if "I_max" in df.columns else np.nan,
        "metrics_used_mean": float(df["metrics_used"].mean()) if "metrics_used" in df.columns else np.nan,
    }


def _emu_stats(df: pd.DataFrame) -> Dict[str, float]:
    if df.empty:
        return {"trained": 0, "skipped": 0, "gpr_cv_mean": np.nan, "gpr_cv_median": np.nan}
    if "skipped" not in df.columns:
        return {"trained": float(len(df)), "skipped": 0.0, "gpr_cv_mean": np.nan, "gpr_cv_median": np.nan}
    trained = df[~df["skipped"].map(_to_bool)].copy()
    return {
        "trained": float(len(trained)),
        "skipped": float(df["skipped"].map(_to_bool).sum()),
        "gpr_cv_mean": float(trained["gpr_r2_cv_mean"].mean()) if not trained.empty else np.nan,
        "gpr_cv_median": float(trained["gpr_r2_cv_mean"].median()) if not trained.empty else np.nan,
    }


def _bad_run_pct(df: pd.DataFrame) -> float:
    if "bad_run" not in df.columns or df.empty:
        return np.nan
    return float(df["bad_run"].map(_to_bool).mean() * 100.0)


def _rep_stats(df: pd.DataFrame) -> Dict[str, float]:
    if df.empty:
        return {
            "rows": 0.0,
            "Employment_min": np.nan,
            "Output_min": np.nan,
            "Consumption_min": np.nan,
            "BankFailed_max": np.nan,
            "BalanceOK_min": np.nan,
        }
    return {
        "rows": float(len(df)),
        "Employment_min": float(df["Employment_mean"].min()) if "Employment_mean" in df else np.nan,
        "Output_min": float(df["Output_mean"].min()) if "Output_mean" in df else np.nan,
        "Consumption_min": float(df["Consumption_mean"].min()) if "Consumption_mean" in df else np.nan,
        "BankFailed_max": float(df["BankFailed_share"].max()) if "BankFailed_share" in df else np.nan,
        "BalanceOK_min": float(df["BalanceOK_share"].min()) if "BalanceOK_share" in df else np.nan,
    }


def _gate_eval(
    wave2_hm: pd.DataFrame,
    rep_summary: pd.DataFrame,
    sa_df: pd.DataFrame,
    gate_cfg: Dict[str, float],
    confirm_stats: Dict[str, float] | None = None,
) -> pd.DataFrame:
    hm = _hm_stats(wave2_hm)
    nroy_pct = hm["nroy_pct"]
    i_med = hm["Imax_median"]
    rep = _rep_stats(rep_summary)

    components = set(sa_df["component"].astype(str).unique()) if not sa_df.empty and "component" in sa_df.columns else set()
    has_rank = ("rank_at_max_reduction" in sa_df.columns) and ("rank_by_mean" in sa_df.columns)
    has_16 = len(sa_df) >= 16

    checks: List[Tuple[str, bool, str]] = [
        (
            f"NROY wave2 in [{gate_cfg['nroy_wave2_min_pct']:.0f},{gate_cfg['nroy_wave2_max_pct']:.0f}]%",
            bool(np.isfinite(nroy_pct) and gate_cfg["nroy_wave2_min_pct"] <= nroy_pct <= gate_cfg["nroy_wave2_max_pct"]),
            f"{nroy_pct:.2f}%",
        ),
        ("I_max median wave2 < 3.2", bool(np.isfinite(i_med) and i_med < gate_cfg["imax_wave2_median_max"]), f"{i_med:.4f}"),
        (
            f"Representative Employment_mean >= {gate_cfg['rep_employment_min']:.0f}",
            bool(np.isfinite(rep["Employment_min"]) and rep["Employment_min"] >= gate_cfg["rep_employment_min"]),
            f"min={rep['Employment_min']:.4f}",
        ),
        (
            f"Representative Output_mean >= {gate_cfg['rep_output_min']:.0f}",
            bool(np.isfinite(rep["Output_min"]) and rep["Output_min"] >= gate_cfg["rep_output_min"]),
            f"min={rep['Output_min']:.4f}",
        ),
        (
            f"Representative Consumption_mean >= {gate_cfg['rep_consumption_min']:.0f}",
            bool(np.isfinite(rep["Consumption_min"]) and rep["Consumption_min"] >= gate_cfg["rep_consumption_min"]),
            f"min={rep['Consumption_min']:.4f}",
        ),
        ("Representative BankFailed_share == 0", bool(np.isfinite(rep["BankFailed_max"]) and rep["BankFailed_max"] == 0.0), f"max={rep['BankFailed_max']:.6f}"),
        ("Representative BalanceOK_share == 1", bool(np.isfinite(rep["BalanceOK_min"]) and rep["BalanceOK_min"] == 1.0), f"min={rep['BalanceOK_min']:.6f}"),
        ("SA components EV/OU/MD/CU present", components == {"EV", "OU", "MD", "CU"}, f"components={sorted(list(components))}"),
        ("SA has >=16 rows", has_16, f"rows={len(sa_df)}"),
        ("SA ranks present", has_rank, f"rank_cols={has_rank}"),
    ]
    if confirm_stats is not None:
        tol = float(gate_cfg.get("confirmatory_nroy_tol_pp", 5.0))
        nroy_delta = float(confirm_stats.get("nroy_delta_pp", np.nan))
        top2_stable = bool(confirm_stats.get("sa_top2_stable", False))
        checks.append(
            (
                f"Confirmatory stability (|NROY delta| <= {tol:.1f}pp and SA top2 stable)",
                bool(np.isfinite(nroy_delta) and abs(nroy_delta) <= tol and top2_stable),
                f"nroy_delta_pp={nroy_delta:.2f}; sa_top2_stable={top2_stable}",
            )
        )
    out = pd.DataFrame(checks, columns=["check", "pass", "value"])
    out["pass"] = out["pass"].astype(bool)
    return out


def _fmt(v: float, nd: int = 4) -> str:
    if v is None or not np.isfinite(v):
        return "n/a"
    return f"{v:.{nd}f}"


def _structural_gate_eval(wave2_lhs: pd.DataFrame, rep_summary: pd.DataFrame) -> pd.DataFrame:
    checks: List[Tuple[str, bool, str]] = []
    if "PriceDispersion_mean" in wave2_lhs.columns and not wave2_lhs.empty:
        p50 = float(wave2_lhs["PriceDispersion_mean"].median())
        checks.append(
            (
                "Structural: PriceDispersion median in [0.03,0.30]",
                bool(0.03 <= p50 <= 0.30),
                f"median={p50:.4f}",
            )
        )
    if "InventoryGap_mean" in wave2_lhs.columns and not wave2_lhs.empty:
        p95 = float(wave2_lhs["InventoryGap_mean"].abs().quantile(0.95))
        checks.append(
            (
                "Structural: |InventoryGap| p95 <= 1.5",
                bool(p95 <= 1.5),
                f"p95_abs={p95:.4f}",
            )
        )
    if "CreditRejections_mean" in wave2_lhs.columns and not wave2_lhs.empty:
        p90 = float(wave2_lhs["CreditRejections_mean"].quantile(0.90))
        checks.append(
            (
                "Structural: CreditRejections p90 <= 10",
                bool(p90 <= 10.0),
                f"p90={p90:.4f}",
            )
        )
    if "FirmDowntimeShare_mean" in rep_summary.columns and not rep_summary.empty:
        mx = float(rep_summary["FirmDowntimeShare_mean"].max())
        checks.append(
            (
                "Structural: FirmDowntimeShare max <= 0.20",
                bool(mx <= 0.20),
                f"max={mx:.4f}",
            )
        )
    return pd.DataFrame(checks, columns=["check", "pass", "value"]) if checks else pd.DataFrame(columns=["check", "pass", "value"])


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build master research-core markdown report.")
    p.add_argument("--root", type=str, default="output_research_core")
    p.add_argument("--run-config", type=str, default="")
    p.add_argument("--out", type=str, default="output_research_core/04_master/research_core_report.md")
    p.add_argument("--tables-dir", type=str, default="output_research_core/04_master/research_core_tables")
    p.add_argument("--confirmatory-summary", type=str, default="")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    tables_dir = Path(args.tables_dir)
    out_path = Path(args.out)
    tables_dir.mkdir(parents=True, exist_ok=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    run_cfg = {}
    if args.run_config:
        run_cfg = json.loads(Path(args.run_config).read_text(encoding="utf-8"))

    wave1_dir = root / "01_wave1"
    wave2_dir = root / "02_wave2"
    rep_dir = root / "03_report_set"

    wave1_lhs = pd.read_csv(wave1_dir / "lhs_runs.csv")
    wave2_lhs = pd.read_csv(wave2_dir / "lhs_runs_wave2.csv")
    wave1_emu = pd.read_csv(wave1_dir / "emulator_scores.csv")
    wave2_emu = pd.read_csv(wave2_dir / "emulator_scores.csv")
    wave1_hm = pd.read_csv(wave1_dir / "history_matching.csv")
    wave2_hm = pd.read_csv(wave2_dir / "history_matching.csv")
    wave1_ri = pd.read_csv(wave1_dir / "refined_intervals.csv")
    wave2_ri = pd.read_csv(wave2_dir / "refined_intervals.csv")
    wave1_sa = pd.read_csv(wave1_dir / "sensitivity_uncertainty.csv")
    wave2_sa = pd.read_csv(wave2_dir / "sensitivity_uncertainty.csv")
    rep_summary = pd.read_csv(rep_dir / "representative_summary.csv")
    rep_points = pd.read_csv(rep_dir / "representative_points.csv")
    confirm_stats = None
    if args.confirmatory_summary and Path(args.confirmatory_summary).exists():
        confirm_stats = json.loads(Path(args.confirmatory_summary).read_text(encoding="utf-8"))

    emu_tbl = pd.DataFrame(
        [
            {"wave": "wave1", **_emu_stats(wave1_emu)},
            {"wave": "wave2", **_emu_stats(wave2_emu)},
        ]
    )
    hm_tbl = pd.DataFrame(
        [
            {"wave": "wave1", **_hm_stats(wave1_hm)},
            {"wave": "wave2", **_hm_stats(wave2_hm)},
        ]
    )
    compare_tbl = pd.DataFrame(
        [
            {
                "metric": "NROY_pct",
                "wave1": _hm_stats(wave1_hm)["nroy_pct"],
                "wave2": _hm_stats(wave2_hm)["nroy_pct"],
                "delta_wave2_minus_wave1": _hm_stats(wave2_hm)["nroy_pct"] - _hm_stats(wave1_hm)["nroy_pct"],
            },
            {
                "metric": "Imax_median",
                "wave1": _hm_stats(wave1_hm)["Imax_median"],
                "wave2": _hm_stats(wave2_hm)["Imax_median"],
                "delta_wave2_minus_wave1": _hm_stats(wave2_hm)["Imax_median"] - _hm_stats(wave1_hm)["Imax_median"],
            },
            {
                "metric": "RI_shrink_mean",
                "wave1": float(wave1_ri["shrink_pct"].mean()) if not wave1_ri.empty else np.nan,
                "wave2": float(wave2_ri["shrink_pct"].mean()) if not wave2_ri.empty else np.nan,
                "delta_wave2_minus_wave1": (float(wave2_ri["shrink_pct"].mean()) if not wave2_ri.empty else np.nan)
                - (float(wave1_ri["shrink_pct"].mean()) if not wave1_ri.empty else np.nan),
            },
            {
                "metric": "bad_run_pct",
                "wave1": _bad_run_pct(wave1_lhs),
                "wave2": _bad_run_pct(wave2_lhs),
                "delta_wave2_minus_wave1": _bad_run_pct(wave2_lhs) - _bad_run_pct(wave1_lhs),
            },
        ]
    )

    sa_rank_tbl = wave2_sa[["component", "share_at_max_reduction", "share_mean", "rank_at_max_reduction", "rank_by_mean"]].drop_duplicates()
    rep_tbl = pd.DataFrame([_rep_stats(rep_summary)])

    gate_cfg = run_cfg.get(
        "quality_gate",
        {
            "nroy_wave2_min_pct": 25.0,
            "nroy_wave2_max_pct": 60.0,
            "imax_wave2_median_max": 3.2,
            "rep_employment_min": 45.0,
            "rep_output_min": 50.0,
            "rep_consumption_min": 50.0,
            "confirmatory_nroy_tol_pp": 5.0,
        },
    )
    gate_tbl = _gate_eval(wave2_hm, rep_summary, wave2_sa, gate_cfg, confirm_stats=confirm_stats)
    all_pass = bool(gate_tbl["pass"].all())
    structural_gate_tbl = _structural_gate_eval(wave2_lhs, rep_summary)
    structural_all_pass = bool(structural_gate_tbl["pass"].all()) if not structural_gate_tbl.empty else True

    emu_tbl.to_csv(tables_dir / "emulator_quality.csv", index=False)
    hm_tbl.to_csv(tables_dir / "history_matching_summary.csv", index=False)
    compare_tbl.to_csv(tables_dir / "wave1_wave2_comparison.csv", index=False)
    sa_rank_tbl.to_csv(tables_dir / "sa_ranking_wave2.csv", index=False)
    rep_tbl.to_csv(tables_dir / "representative_summary_stats.csv", index=False)
    gate_tbl.to_csv(tables_dir / "quality_gates.csv", index=False)
    structural_gate_tbl.to_csv(tables_dir / "quality_gates_structural.csv", index=False)

    lines: List[str] = []
    lines.append("# Research Core Report\n\n")
    lines.append("## ABM Input A\n")
    lines.append("- Baseline branch excluded from core analysis.\n")
    lines.append("- Parameter bounds: `scripts/run_operator.py:PARAM_BOUNDS`.\n")
    if run_cfg:
        lines.append(f"- Wave1 settings: `{run_cfg.get('wave1_lhs', {})}`.\n")
        lines.append(f"- Wave2 settings: `{run_cfg.get('wave2_lhs', {})}`.\n")
        lines.append(f"- HM/SA settings: `{run_cfg.get('hm_sa', {})}`.\n")
        lines.append(f"- Representative settings: `{run_cfg.get('report_set', {})}`.\n")
    lines.append("\n")

    lines.append("## Emulator/HM B\n")
    lines.append("| Wave | Trained metrics | Skipped | GPR CV mean | GPR CV median | NROY % | I_max median | I_max p95 | I_max max | metrics_used mean |\n")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
    for _, row in hm_tbl.merge(emu_tbl, on="wave").iterrows():
        lines.append(
            f"| {row['wave']} | {int(row['trained'])} | {int(row['skipped'])} | {_fmt(row['gpr_cv_mean'],3)} | {_fmt(row['gpr_cv_median'],3)} | "
            f"{_fmt(row['nroy_pct'],2)} | {_fmt(row['Imax_median'],3)} | {_fmt(row['Imax_p95'],3)} | {_fmt(row['Imax_max'],3)} | {_fmt(row['metrics_used_mean'],2)} |\n"
        )
    lines.append("\n")

    lines.append("## Improvement C (Wave2 vs Wave1)\n")
    lines.append("| Metric | Wave1 | Wave2 | Delta |\n")
    lines.append("|---|---:|---:|---:|\n")
    for _, row in compare_tbl.iterrows():
        lines.append(f"| {row['metric']} | {_fmt(row['wave1'],4)} | {_fmt(row['wave2'],4)} | {_fmt(row['delta_wave2_minus_wave1'],4)} |\n")
    lines.append("\n")

    lines.append("## Interpretation G (SA, wave2)\n")
    lines.append("| Component | Share@max reduction | Share mean | Rank@max | Rank mean |\n")
    lines.append("|---|---:|---:|---:|---:|\n")
    for _, row in sa_rank_tbl.sort_values("rank_at_max_reduction").iterrows():
        lines.append(
            f"| {row['component']} | {_fmt(float(row['share_at_max_reduction']),4)} | {_fmt(float(row['share_mean']),4)} | "
            f"{int(row['rank_at_max_reduction'])} | {int(row['rank_by_mean'])} |\n"
        )
    lines.append("\n")

    lines.append("## Representative Runs\n")
    lines.append(f"- Selected points: `{len(rep_points)}`\n")
    lines.append(f"- Long-run realizations: `{len(rep_summary)}` (expected n_points x n_seeds)\n")
    rs = _rep_stats(rep_summary)
    lines.append(f"- Employment_mean min: `{_fmt(rs['Employment_min'],4)}`\n")
    lines.append(f"- Output_mean min: `{_fmt(rs['Output_min'],4)}`\n")
    lines.append(f"- Consumption_mean min: `{_fmt(rs['Consumption_min'],4)}`\n")
    lines.append(f"- BankFailed_share max: `{_fmt(rs['BankFailed_max'],6)}`\n")
    lines.append(f"- BalanceOK_share min: `{_fmt(rs['BalanceOK_min'],6)}`\n\n")

    lines.append("## Final Scientific Verdict\n")
    lines.append(f"- Overall quality gate: `{'PASS' if all_pass else 'FAIL'}`\n")
    lines.append("| Check | Pass | Value |\n")
    lines.append("|---|---|---|\n")
    for _, row in gate_tbl.iterrows():
        lines.append(f"| {row['check']} | {'PASS' if bool(row['pass']) else 'FAIL'} | {row['value']} |\n")
    lines.append("\n")
    lines.append("## Model-Structure Diagnostics\n")
    if structural_gate_tbl.empty:
        lines.append("- Structural gates: `n/a` (metrics absent in this run).\n\n")
    else:
        lines.append(f"- Structural gate (contour B): `{'PASS' if structural_all_pass else 'FAIL'}`\n")
        lines.append("| Check | Pass | Value |\n")
        lines.append("|---|---|---|\n")
        for _, row in structural_gate_tbl.iterrows():
            lines.append(f"| {row['check']} | {'PASS' if bool(row['pass']) else 'FAIL'} | {row['value']} |\n")
        lines.append("\n")
    if confirm_stats is not None:
        lines.append("## Confirmatory Stability\n")
        lines.append(f"- confirmatory_nroy_pct: `{_fmt(float(confirm_stats.get('confirm_nroy_pct', np.nan)),2)}`\n")
        lines.append(f"- main_nroy_pct: `{_fmt(float(confirm_stats.get('main_nroy_pct', np.nan)),2)}`\n")
        lines.append(f"- nroy_delta_pp: `{_fmt(float(confirm_stats.get('nroy_delta_pp', np.nan)),2)}`\n")
        lines.append(f"- SA top2 stable: `{bool(confirm_stats.get('sa_top2_stable', False))}`\n\n")
    if not all_pass:
        lines.append("### Failure Reasons & Corrective Actions\n")
        failed = gate_tbl[~gate_tbl["pass"]]
        for _, row in failed.iterrows():
            lines.append(f"- `{row['check']}` failed (`{row['value']}`).\n")
        lines.append("- Suggested actions: retune targets sigma_model/sigma_obs, adjust EV quantile, or refine wave2 bounds.\n")

    out_path.write_text("".join(lines), encoding="utf-8")
    print(f"Saved report to {out_path}")
    print(f"Saved tables to {tables_dir}")


if __name__ == "__main__":
    main()
