from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

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


def _emu_stats(df: pd.DataFrame, share_threshold: float = 0.30, min_r2_gate: float = 0.25) -> Dict[str, float]:
    if df.empty:
        return {"trained": 0, "gate_metrics": 0, "skipped": 0, "gpr_cv_mean": np.nan, "gpr_cv_median": np.nan, "gpr_cv_share_ge": np.nan}
    trained = df[~df["skipped"].map(_to_bool)].copy() if "skipped" in df.columns else df.copy()
    if trained.empty or "gpr_r2_cv_mean" not in trained.columns:
        return {"trained": float(len(trained)), "gate_metrics": 0.0, "skipped": 0.0, "gpr_cv_mean": np.nan, "gpr_cv_median": np.nan, "gpr_cv_share_ge": np.nan}
    gate_df = trained[trained["gpr_r2_cv_mean"] >= float(min_r2_gate)].copy()
    if gate_df.empty:
        return {
            "trained": float(len(trained)),
            "gate_metrics": 0.0,
            "skipped": float(df["skipped"].map(_to_bool).sum()) if "skipped" in df.columns else 0.0,
            "gpr_cv_mean": np.nan,
            "gpr_cv_median": np.nan,
            "gpr_cv_share_ge": np.nan,
        }
    return {
        "trained": float(len(trained)),
        "gate_metrics": float(len(gate_df)),
        "skipped": float(df["skipped"].map(_to_bool).sum()) if "skipped" in df.columns else 0.0,
        "gpr_cv_mean": float(gate_df["gpr_r2_cv_mean"].mean()),
        "gpr_cv_median": float(gate_df["gpr_r2_cv_mean"].median()),
        "gpr_cv_share_ge": float((gate_df["gpr_r2_cv_mean"] >= share_threshold).mean()),
    }


def _bad_run_pct(df: pd.DataFrame) -> float:
    if "bad_run" not in df.columns or df.empty:
        return np.nan
    return float(df["bad_run"].map(_to_bool).mean() * 100.0)


def _rep_stats(df: pd.DataFrame) -> Dict[str, float]:
    if df.empty:
        return {"rows": 0.0, "Employment_min": np.nan, "Output_min": np.nan, "Consumption_min": np.nan, "BankFailed_max": np.nan, "BalanceOK_min": np.nan, "CreditRejections_p90": np.nan}
    return {
        "rows": float(len(df)),
        "Employment_min": float(df["Employment_mean"].min()) if "Employment_mean" in df else np.nan,
        "Output_min": float(df["Output_mean"].min()) if "Output_mean" in df else np.nan,
        "Consumption_min": float(df["Consumption_mean"].min()) if "Consumption_mean" in df else np.nan,
        "BankFailed_max": float(df["BankFailed_share"].max()) if "BankFailed_share" in df else np.nan,
        "BalanceOK_min": float(df["BalanceOK_share"].min()) if "BalanceOK_share" in df else np.nan,
        "CreditRejections_p90": float(df["CreditRejections_mean"].quantile(0.90)) if "CreditRejections_mean" in df else np.nan,
    }


def _mk_gate(check: str, ok: bool, value: str, contour: str, blocking: bool, gate_group: str, reason_code: str) -> Dict[str, object]:
    return {
        "check": check,
        "pass": bool(ok),
        "value": value,
        "contour": contour,
        "blocking": bool(blocking),
        "gate_group": gate_group,
        "reason_code": "ok" if ok else reason_code,
    }


def _legacy_gate_eval(wave2_hm: pd.DataFrame, gate_cfg: Dict[str, float]) -> pd.DataFrame:
    hm = _hm_stats(wave2_hm)
    lo = float(gate_cfg.get("legacy_nroy_wave2_min_pct", 25.0))
    hi = float(gate_cfg.get("legacy_nroy_wave2_max_pct", 60.0))
    ok = bool(np.isfinite(hm["nroy_pct"]) and lo <= hm["nroy_pct"] <= hi)
    rows = [_mk_gate(f"NROY wave2 in [{lo:.0f},{hi:.0f}]%", ok, f"{hm['nroy_pct']:.2f}%", "legacy", False, "emulator_hm", "legacy_nroy_out_of_range")]
    return pd.DataFrame(rows)


def _blocking_gates(
    wave2_hm: pd.DataFrame,
    wave2_lhs: pd.DataFrame,
    wave2_emu: pd.DataFrame,
    rep_summary: pd.DataFrame,
    sa_df: pd.DataFrame,
    gate_cfg: Dict[str, float],
    confirm_stats: Dict[str, float] | None,
) -> pd.DataFrame:
    hm = _hm_stats(wave2_hm)
    hm_min_r2 = float(gate_cfg.get("hm_min_r2_for_gate", 0.25))
    emu = _emu_stats(wave2_emu, share_threshold=float(gate_cfg.get("emu_r2_share_threshold", 0.30)), min_r2_gate=hm_min_r2)
    rep = _rep_stats(rep_summary)
    bad_run = _bad_run_pct(wave2_lhs)

    components = set(sa_df["component"].astype(str).unique()) if not sa_df.empty and "component" in sa_df.columns else set()
    has_rank = ("rank_at_max_reduction" in sa_df.columns) and ("rank_by_mean" in sa_df.columns)
    has_16 = len(sa_df) >= 16

    rows: List[Dict[str, object]] = []
    lo = float(gate_cfg.get("nroy_wave2_min_pct", 25.0))
    hi = float(gate_cfg.get("nroy_wave2_max_pct", 65.0))
    rows.append(_mk_gate(f"NROY wave2 in [{lo:.0f},{hi:.0f}]%", bool(np.isfinite(hm["nroy_pct"]) and lo <= hm["nroy_pct"] <= hi), f"{hm['nroy_pct']:.2f}%", "stability", True, "emulator_hm", "nroy_out_of_range"))
    rows.append(_mk_gate(f"I_max median wave2 < {float(gate_cfg.get('imax_wave2_median_max',3.0)):.1f}", bool(np.isfinite(hm["Imax_median"]) and hm["Imax_median"] < float(gate_cfg.get("imax_wave2_median_max", 3.0))), f"{hm['Imax_median']:.4f}", "stability", True, "emulator_hm", "imax_median_too_high"))
    rows.append(_mk_gate(f"I_max p95 wave2 < {float(gate_cfg.get('imax_wave2_p95_max',4.5)):.1f}", bool(np.isfinite(hm["Imax_p95"]) and hm["Imax_p95"] < float(gate_cfg.get("imax_wave2_p95_max", 4.5))), f"{hm['Imax_p95']:.4f}", "stability", True, "emulator_hm", "imax_p95_too_high"))
    rows.append(_mk_gate(f"Emulator CV-R2 median >= {float(gate_cfg.get('emu_r2_cv_median_min',0.60)):.2f}", bool(np.isfinite(emu["gpr_cv_median"]) and emu["gpr_cv_median"] >= float(gate_cfg.get("emu_r2_cv_median_min", 0.60))), f"{emu['gpr_cv_median']:.4f}", "stability", True, "emulator_hm", "emu_median_below_min"))
    rows.append(_mk_gate(f"Emulator CV-R2 share>={float(gate_cfg.get('emu_r2_share_threshold',0.30)):.2f} >= {float(gate_cfg.get('emu_r2_share_min',0.65)):.2f}", bool(np.isfinite(emu["gpr_cv_share_ge"]) and emu["gpr_cv_share_ge"] >= float(gate_cfg.get("emu_r2_share_min", 0.65))), f"{emu['gpr_cv_share_ge']:.4f}", "stability", True, "emulator_hm", "emu_share_below_min"))
    rows.append(_mk_gate("bad_run_pct_w2 <= 5%", bool(np.isfinite(bad_run) and bad_run <= float(gate_cfg.get("bad_run_pct_w2_max", 5.0))), f"{bad_run:.2f}%", "stability", True, "abm", "bad_run_too_high"))
    rows.append(_mk_gate("Representative Employment_mean >= 45", bool(np.isfinite(rep["Employment_min"]) and rep["Employment_min"] >= float(gate_cfg.get("rep_employment_min", 45.0))), f"min={rep['Employment_min']:.4f}", "stability", True, "interpretability", "employment_below_floor"))
    rows.append(_mk_gate("Representative Output_mean >= 50", bool(np.isfinite(rep["Output_min"]) and rep["Output_min"] >= float(gate_cfg.get("rep_output_min", 50.0))), f"min={rep['Output_min']:.4f}", "stability", True, "interpretability", "output_below_floor"))
    rows.append(_mk_gate("Representative Consumption_mean >= 50", bool(np.isfinite(rep["Consumption_min"]) and rep["Consumption_min"] >= float(gate_cfg.get("rep_consumption_min", 50.0))), f"min={rep['Consumption_min']:.4f}", "stability", True, "interpretability", "consumption_below_floor"))
    rows.append(_mk_gate("Representative BankFailed_share == 0", bool(np.isfinite(rep["BankFailed_max"]) and rep["BankFailed_max"] == 0.0), f"max={rep['BankFailed_max']:.6f}", "stability", True, "abm", "bank_failed_positive"))
    rows.append(_mk_gate("Representative BalanceOK_share == 1", bool(np.isfinite(rep["BalanceOK_min"]) and rep["BalanceOK_min"] == 1.0), f"min={rep['BalanceOK_min']:.6f}", "stability", True, "abm", "balance_not_one"))
    rows.append(_mk_gate("SA components EV/OU/MD/CU present", components == {"EV", "OU", "MD", "CU"}, f"components={sorted(list(components))}", "stability", True, "confirmatory", "sa_components_missing"))
    rows.append(_mk_gate("SA has >=16 rows", has_16, f"rows={len(sa_df)}", "stability", True, "confirmatory", "sa_rows_too_few"))
    rows.append(_mk_gate("SA ranks present", has_rank, f"rank_cols={has_rank}", "stability", True, "confirmatory", "sa_ranks_missing"))

    if "PriceDispersion_mean" in wave2_lhs.columns and not wave2_lhs.empty:
        p50 = float(wave2_lhs["PriceDispersion_mean"].median())
        rows.append(_mk_gate("Structural: PriceDispersion median in [0.03,0.30]", bool(0.03 <= p50 <= 0.30), f"median={p50:.4f}", "structural", True, "interpretability", "price_dispersion_out_of_range"))
    if "InventoryGap_mean" in wave2_lhs.columns and not wave2_lhs.empty:
        p95 = float(wave2_lhs["InventoryGap_mean"].abs().quantile(0.95))
        rows.append(_mk_gate("Structural: |InventoryGap| p95 <= 1.5", bool(p95 <= 1.5), f"p95_abs={p95:.4f}", "structural", True, "interpretability", "inventory_gap_too_high"))
    if "CreditRejectionRate_mean" in wave2_lhs.columns and not wave2_lhs.empty:
        mean_rate = float(wave2_lhs["CreditRejectionRate_mean"].mean())
        p90_rate = float(wave2_lhs["CreditRejectionRate_mean"].quantile(0.90))
        rate_thr = float(gate_cfg.get("credit_rejection_rate_mean_max", 0.80))
        rows.append(
            _mk_gate(
                f"Structural: CreditRejectionRate mean <= {rate_thr:.2f}",
                bool(mean_rate <= rate_thr),
                f"mean={mean_rate:.4f}; p90={p90_rate:.4f}",
                "structural",
                True,
                "interpretability",
                "credit_rejection_rate_too_high",
            )
        )
    if "CreditRejections_mean" in wave2_lhs.columns and not wave2_lhs.empty:
        p90 = float(wave2_lhs["CreditRejections_mean"].quantile(0.90))
        ref_thr = float(gate_cfg.get("credit_rejections_p90_max", 30.0))
        abs_thr = float(gate_cfg.get("credit_rejections_p90_abs_max", 90.0))
        baseline = float(gate_cfg.get("credit_rejections_baseline_p90", np.nan))
        improve_min = float(gate_cfg.get("credit_rejections_improvement_min_pct", 15.0))
        improve_pct = ((baseline - p90) / baseline * 100.0) if np.isfinite(baseline) and baseline > 0 else np.nan
        rows.append(_mk_gate(f"Structural(reference): CreditRejections p90 <= {ref_thr:.0f}", bool(p90 <= ref_thr), f"p90={p90:.4f}", "structural", False, "interpretability", "credit_rejections_reference_failed"))
        rows.append(_mk_gate(f"Structural(reference): CreditRejections p90 <= {abs_thr:.0f}", bool(p90 <= abs_thr), f"p90={p90:.4f}", "structural", False, "interpretability", "credit_rejections_abs_reference_failed"))
        rows.append(_mk_gate(f"Structural(reference): CreditRejections p90 improvement vs baseline >= {improve_min:.0f}%", bool(np.isfinite(improve_pct) and improve_pct >= improve_min), f"improve={improve_pct:.2f}% (baseline={baseline:.3f}, p90={p90:.4f})", "structural", False, "interpretability", "credit_rejections_improvement_reference_failed"))

    if confirm_stats is not None:
        tol = float(gate_cfg.get("confirmatory_nroy_tol_pp", 5.0))
        d = float(confirm_stats.get("nroy_delta_pp", np.nan))
        stable = bool(confirm_stats.get("sa_top2_stable", False))
        rows.append(_mk_gate(f"Confirmatory stability (|NROY delta| <= {tol:.1f}pp and SA top2 stable)", bool(np.isfinite(d) and abs(d) <= tol and stable), f"nroy_delta_pp={d:.2f}; sa_top2_stable={stable}", "stability", True, "confirmatory", "confirmatory_unstable"))

    return pd.DataFrame(rows)


def _fmt(v: float, nd: int = 4) -> str:
    if v is None or not np.isfinite(v):
        return "n/a"
    return f"{v:.{nd}f}"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build master research-core markdown report.")
    p.add_argument("--root", type=str, default="output_research_core_v3")
    p.add_argument("--run-config", type=str, default="")
    p.add_argument("--out", type=str, default="output_research_core_v3/04_master/research_core_report.md")
    p.add_argument("--tables-dir", type=str, default="output_research_core_v3/04_master/research_core_tables")
    p.add_argument("--confirmatory-summary", type=str, default="")
    p.add_argument("--waves-summary", type=str, default="")
    p.add_argument("--stopping-diagnostics", type=str, default="")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    tables_dir = Path(args.tables_dir)
    out_path = Path(args.out)
    tables_dir.mkdir(parents=True, exist_ok=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    run_cfg = json.loads(Path(args.run_config).read_text(encoding="utf-8")) if args.run_config else {}
    paths = run_cfg.get("paths", {})
    wave1_dir = root / str(paths.get("wave1_dir", "01_wave1"))
    wave2_dir = root / str(paths.get("wave2_dir", "02_wave2"))
    rep_dir = root / str(paths.get("report_set_dir", "03_report_set"))

    wave1_lhs = pd.read_csv(wave1_dir / "lhs_runs.csv")
    wave2_lhs_candidates = sorted(wave2_dir.glob("lhs_runs_wave*.csv")) + sorted(wave2_dir.glob("lhs_runs*.csv"))
    wave2_lhs = pd.read_csv(wave2_lhs_candidates[0]) if wave2_lhs_candidates else pd.DataFrame()
    wave1_emu = pd.read_csv(wave1_dir / "emulator_scores.csv")
    wave2_emu = pd.read_csv(wave2_dir / "emulator_scores.csv")
    wave1_hm = pd.read_csv(wave1_dir / "history_matching.csv")
    wave2_hm = pd.read_csv(wave2_dir / "history_matching.csv")
    wave1_ri = pd.read_csv(wave1_dir / "refined_intervals.csv")
    wave2_ri = pd.read_csv(wave2_dir / "refined_intervals.csv")
    wave2_sa = pd.read_csv(wave2_dir / "sensitivity_uncertainty.csv")
    rep_summary = pd.read_csv(rep_dir / "representative_summary.csv")
    rep_points = pd.read_csv(rep_dir / "representative_points.csv")

    confirm_stats = json.loads(Path(args.confirmatory_summary).read_text(encoding="utf-8")) if args.confirmatory_summary and Path(args.confirmatory_summary).exists() else None
    waves_summary = pd.read_csv(Path(args.waves_summary)) if args.waves_summary and Path(args.waves_summary).exists() else pd.DataFrame()
    stopping_diag = json.loads(Path(args.stopping_diagnostics).read_text(encoding="utf-8")) if args.stopping_diagnostics and Path(args.stopping_diagnostics).exists() else {}

    gate_cfg = dict(run_cfg.get("quality_gate", {}))
    hm_min_r2 = float(run_cfg.get("hm_sa", {}).get("min_r2_for_history_matching", 0.25))
    gate_cfg["hm_min_r2_for_gate"] = hm_min_r2
    legacy = _legacy_gate_eval(wave2_hm, gate_cfg)
    blocking = _blocking_gates(wave2_hm, wave2_lhs, wave2_emu, rep_summary, wave2_sa, gate_cfg, confirm_stats)
    gates = pd.concat([legacy, blocking], ignore_index=True)

    overall_pass_v3 = bool(gates[gates["blocking"].map(_to_bool)]["pass"].map(_to_bool).all())
    legacy_pass = bool(gates[gates["contour"] == "legacy"]["pass"].map(_to_bool).all())

    emu_tbl = pd.DataFrame([
        {"wave": "wave1", **_emu_stats(wave1_emu, float(gate_cfg.get("emu_r2_share_threshold", 0.30)), hm_min_r2)},
        {"wave": "wave2", **_emu_stats(wave2_emu, float(gate_cfg.get("emu_r2_share_threshold", 0.30)), hm_min_r2)},
    ])
    hm_tbl = pd.DataFrame([
        {"wave": "wave1", **_hm_stats(wave1_hm)},
        {"wave": "wave2", **_hm_stats(wave2_hm)},
    ])
    compare_tbl = pd.DataFrame([
        {"metric": "NROY_pct", "wave1": _hm_stats(wave1_hm)["nroy_pct"], "wave2": _hm_stats(wave2_hm)["nroy_pct"]},
        {"metric": "Imax_median", "wave1": _hm_stats(wave1_hm)["Imax_median"], "wave2": _hm_stats(wave2_hm)["Imax_median"]},
        {"metric": "RI_shrink_mean", "wave1": float(wave1_ri["shrink_pct"].mean()) if not wave1_ri.empty else np.nan, "wave2": float(wave2_ri["shrink_pct"].mean()) if not wave2_ri.empty else np.nan},
        {"metric": "bad_run_pct", "wave1": _bad_run_pct(wave1_lhs), "wave2": _bad_run_pct(wave2_lhs)},
    ])
    compare_tbl["delta_wave2_minus_wave1"] = compare_tbl["wave2"] - compare_tbl["wave1"]

    emu_tbl.to_csv(tables_dir / "emulator_quality.csv", index=False)
    hm_tbl.to_csv(tables_dir / "history_matching_summary.csv", index=False)
    compare_tbl.to_csv(tables_dir / "wave1_wave2_comparison.csv", index=False)
    gates.to_csv(tables_dir / "quality_gates.csv", index=False)
    blocking[blocking["contour"] == "structural"].to_csv(tables_dir / "quality_gates_structural.csv", index=False)

    lines: List[str] = []
    lines.append("# Research Core Report\n\n")
    lines.append("## Scientific Verdict v3 (blocking)\n")
    lines.append(f"- overall_gate_pass_v3: `{'PASS' if overall_pass_v3 else 'FAIL'}`\n")
    lines.append(f"- legacy_gate_pass(reference): `{'PASS' if legacy_pass else 'FAIL'}`\n\n")
    lines.append("| Check | Pass | Value | Group | Reason |\n")
    lines.append("|---|---|---|---|---|\n")
    for _, r in gates[gates["blocking"].map(_to_bool)].iterrows():
        lines.append(f"| {r['check']} | {'PASS' if bool(r['pass']) else 'FAIL'} | {r['value']} | {r['gate_group']} | {r['reason_code']} |\n")
    lines.append("\n")

    lines.append("## Wave-by-wave Convergence\n")
    if waves_summary.empty:
        lines.append("- waves_summary.csv not found.\n\n")
    else:
        lines.append("```\n")
        lines.append(waves_summary.to_string(index=False))
        lines.append("\n```\n\n")
    if stopping_diag:
        lines.append(f"- stopping_diagnostics: `{stopping_diag}`\n\n")

    lines.append("## Economic Interpretability\n")
    lines.append(f"- Representative points: `{len(rep_points)}`\n")
    rs = _rep_stats(rep_summary)
    lines.append(f"- Employment_mean min: `{_fmt(rs['Employment_min'])}`\n")
    lines.append(f"- Output_mean min: `{_fmt(rs['Output_min'])}`\n")
    lines.append(f"- Consumption_mean min: `{_fmt(rs['Consumption_min'])}`\n")
    lines.append(f"- CreditRejections p90: `{_fmt(rs['CreditRejections_p90'])}`\n\n")

    lines.append("## Legacy Contour (reference)\n")
    lines.append("| Check | Pass | Value |\n")
    lines.append("|---|---|---|\n")
    for _, r in gates[gates["contour"] == "legacy"].iterrows():
        lines.append(f"| {r['check']} | {'PASS' if bool(r['pass']) else 'FAIL'} | {r['value']} |\n")
    lines.append("\n")

    out_path.write_text("".join(lines), encoding="utf-8")
    print(f"Saved report to {out_path}")
    print(f"Saved tables to {tables_dir}")


if __name__ == "__main__":
    main()
