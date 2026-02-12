from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd


def _python_cmd() -> str:
    return sys.executable


def _run(cmd: List[str], cwd: Path, env: Dict[str, str]) -> None:
    print(f"[RUN] {' '.join(cmd)}")
    proc = subprocess.run(cmd, cwd=str(cwd), env=env)
    if proc.returncode != 0:
        raise SystemExit(f"Command failed ({proc.returncode}): {' '.join(cmd)}")


def _cfg(mode: str) -> Dict[str, object]:
    base: Dict[str, object] = {
        "protocol": {
            "version_tag": "research-core-v2",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "goal": "strict_calibration_for_scientific_presentation",
            "allowed_adjustments": ["sigma_model", "sigma_obs"],
        },
        "mode": mode,
        "paths": {
            "wave1_dir": "01_wave1",
            "wave2_dir": "02_wave2",
            "report_set_dir": "03_report_set",
            "master_dir": "04_master",
            "confirmatory_dir": "05_confirmatory",
        },
        "wave1_lhs": {
            "n": 400,
            "seed": 0,
            "seeds": "0,1,2",
            "steps": 180,
            "window": 30,
            "balance_ok_threshold": 0.60,
        },
        "hm_sa": {
            "targets": "scripts/targets_report.json",
            "min_r2_for_history_matching": 0.25,
            "improb_threshold": 3.0,
            "ev_mode": "seed_replicates",
            "ev_default_quantile": 0.90,
            "sigma_calibration_mode_wave1": "wave1_empirical",
            "sigma_calibration_mode_wave2": "fixed",
            "sa_enable": True,
            "sa_domain": "nroy",
            "sa_reductions": "0.1,0.2,0.3,0.4",
        },
        "sigma_tuning": {
            "enabled": True,
            "max_iters": 3,
            "step": 0.10,
            "nroy_min_pct": 25.0,
            "nroy_max_pct": 60.0,
        },
        "wave2_lhs": {
            "n": 400,
            "seed": 1,
            "seeds": "0,1,2",
            "steps": 180,
            "window": 30,
            "balance_ok_threshold": 0.60,
            "bounds_kind": "p05p95",
        },
        "report_set": {
            "n": 8,
            "steps": 400,
            "window": 60,
            "seeds": "0,1,2",
            "min_employment": 45,
            "min_output": 50,
            "min_consumption": 50,
            "min_hh_deposit": 0.2,
            "max_bank_resolved_share": 0.06,
            "max_bank_bailedout_share": 0.02,
            "max_haircut_mean": 0.03,
            "max_defaults_hh_rate": 0.01,
            "max_defaults_firm_rate": 0.01,
            "balance_ok_threshold": 0.98,
            "relax_if_needed": True,
        },
        "confirmatory": {
            "enabled": True,
            "seed_offset": 101,
        },
        "quality_gate": {
            "nroy_wave2_min_pct": 25.0,
            "nroy_wave2_max_pct": 60.0,
            "imax_wave2_median_max": 3.2,
            "rep_employment_min": 45.0,
            "rep_output_min": 50.0,
            "rep_consumption_min": 50.0,
            "confirmatory_nroy_tol_pp": 5.0,
        },
    }
    if mode == "dry":
        base["wave1_lhs"] = {**base["wave1_lhs"], "n": 20, "steps": 60, "window": 20}  # type: ignore[index]
        base["wave2_lhs"] = {**base["wave2_lhs"], "n": 20, "steps": 60, "window": 20}  # type: ignore[index]
        base["report_set"] = {**base["report_set"], "n": 2, "steps": 120, "window": 30}  # type: ignore[index]
        base["sigma_tuning"] = {**base["sigma_tuning"], "max_iters": 1}  # type: ignore[index]
    return base


def _to_bool(v) -> bool:
    if isinstance(v, bool):
        return v
    if v is None:
        return False
    return str(v).strip().lower() in {"1", "true", "yes", "y"}


def _hm_stats(hm_path: Path) -> Dict[str, float]:
    df = pd.read_csv(hm_path)
    nroy = df["nroy"].map(_to_bool) if "nroy" in df.columns else pd.Series([False] * len(df))
    return {
        "rows": float(len(df)),
        "nroy_pct": float(nroy.mean() * 100.0),
        "Imax_median": float(df["I_max"].median()) if "I_max" in df.columns else float("nan"),
        "Imax_p95": float(df["I_max"].quantile(0.95)) if "I_max" in df.columns else float("nan"),
    }


def _sa_top2(sa_path: Path) -> List[str]:
    if not sa_path.exists():
        return []
    df = pd.read_csv(sa_path)
    if "rank_at_max_reduction" not in df.columns or "component" not in df.columns:
        return []
    top = sorted(df[df["rank_at_max_reduction"] <= 2]["component"].astype(str).dropna().unique().tolist())
    return top


def _scale_sigma_model_in_targets(src: Path, dst: Path, factor: float, reason: str) -> None:
    payload = json.loads(src.read_text(encoding="utf-8"))
    if "_meta" not in payload or not isinstance(payload["_meta"], dict):
        payload["_meta"] = {}
    payload["_meta"]["sigma_tuning_factor"] = factor
    payload["_meta"]["sigma_tuning_reason"] = reason
    for metric, spec in list(payload.items()):
        if not isinstance(metric, str) or metric.startswith("_") or not isinstance(spec, dict):
            continue
        if "sigma_model" not in spec:
            continue
        sigma0 = float(spec.get("sigma_model", 0.0))
        spec["sigma_model"] = max(1e-9, sigma0 * factor)
    dst.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_protocol(cfg: Dict[str, object], out_path: Path) -> None:
    p = cfg.get("protocol", {})
    q = cfg.get("quality_gate", {})
    txt = [
        "# Research Core Protocol\n\n",
        f"- version_tag: `{p.get('version_tag', 'n/a')}`\n",
        f"- created_utc: `{p.get('created_utc', 'n/a')}`\n",
        f"- goal: `{p.get('goal', 'n/a')}`\n",
        f"- allowed_adjustments: `{p.get('allowed_adjustments', [])}`\n\n",
        "## HM Formula\n",
        "- `I = |mu - target| / sqrt(var_obs + var_ev + var_md + var_cu)`\n",
        "- `I_max = max_j I_j`, `NROY = (I_max < threshold)`\n\n",
        "## Quality Gates\n",
        f"- NROY wave2 in [{q.get('nroy_wave2_min_pct', 25)}, {q.get('nroy_wave2_max_pct', 60)}]%\n",
        f"- I_max median wave2 < {q.get('imax_wave2_median_max', 3.2)}\n",
        f"- Representative floors: E>={q.get('rep_employment_min', 45)}, O>={q.get('rep_output_min', 50)}, C>={q.get('rep_consumption_min', 50)}\n",
    ]
    out_path.write_text("".join(txt), encoding="utf-8")


def _train_emulator(
    py: str,
    root: Path,
    env: Dict[str, str],
    data_path: Path,
    targets_path: Path,
    outdir: Path,
    hm_sa: Dict[str, object],
    sigma_mode: str,
    sigma_summary_name: str,
    calibrated_targets_name: str,
) -> None:
    _run(
        [
            py,
            "scripts/train_emulator.py",
            "--data",
            str(data_path),
            "--targets",
            str(targets_path),
            "--min-r2-for-history-matching",
            str(hm_sa["min_r2_for_history_matching"]),
            "--improb-threshold",
            str(hm_sa["improb_threshold"]),
            "--ev-mode",
            str(hm_sa["ev_mode"]),
            "--ev-default-quantile",
            str(hm_sa["ev_default_quantile"]),
            "--sigma-calibration-mode",
            sigma_mode,
            "--sigma-calibration-summary",
            str(outdir / sigma_summary_name),
            "--calibrated-targets-out",
            str(outdir / calibrated_targets_name),
            "--sa-enable",
            "--sa-domain",
            str(hm_sa["sa_domain"]),
            "--sa-reductions",
            str(hm_sa["sa_reductions"]),
            "--outdir",
            str(outdir),
        ],
        cwd=root,
        env=env,
    )


def _run_wave2_hm_with_tuning(
    py: str,
    root: Path,
    env: Dict[str, str],
    wave2_lhs_path: Path,
    wave2_dir: Path,
    initial_targets: Path,
    hm_sa: Dict[str, object],
    sigma_tuning: Dict[str, object],
) -> Tuple[Path, pd.DataFrame]:
    tuning_rows: List[Dict[str, object]] = []
    targets_curr = initial_targets
    max_iters = int(sigma_tuning["max_iters"])
    step = float(sigma_tuning["step"])
    nroy_min = float(sigma_tuning["nroy_min_pct"])
    nroy_max = float(sigma_tuning["nroy_max_pct"])

    for i in range(max_iters + 1):
        _train_emulator(
            py=py,
            root=root,
            env=env,
            data_path=wave2_lhs_path,
            targets_path=targets_curr,
            outdir=wave2_dir,
            hm_sa=hm_sa,
            sigma_mode=str(hm_sa.get("sigma_calibration_mode_wave2", "fixed")),
            sigma_summary_name=f"sigma_calibration_summary_iter{i}.csv",
            calibrated_targets_name=f"targets_calibrated_iter{i}.json",
        )
        stats = _hm_stats(wave2_dir / "history_matching.csv")
        row: Dict[str, object] = {
            "iter": i,
            "targets_path": str(targets_curr),
            **stats,
        }
        tuning_rows.append(row)
        nroy = float(stats["nroy_pct"])
        if (nroy_min <= nroy <= nroy_max) or i >= max_iters:
            break

        if nroy > nroy_max:
            factor = 1.0 - step
            reason = "nroy_above_max_tighten_sigma_model"
        else:
            factor = 1.0 + step
            reason = "nroy_below_min_relax_sigma_model"
        next_targets = wave2_dir / f"targets_tuned_iter{i+1}.json"
        _scale_sigma_model_in_targets(targets_curr, next_targets, factor=factor, reason=reason)
        targets_curr = next_targets

    tuning_df = pd.DataFrame(tuning_rows)
    tuning_df.to_csv(wave2_dir / "sigma_tuning_iterations.csv", index=False)
    final_targets = targets_curr
    return final_targets, tuning_df


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run full research-core pipeline (ABM -> HM/SA -> report set -> master report).")
    p.add_argument("--mode", choices=["dry", "full"], default="full")
    p.add_argument("--outdir", type=str, default="output_research_core")
    p.add_argument("--root", type=str, default=".")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.root).resolve()
    outdir = (root / args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    cfg = _cfg(args.mode)
    run_cfg_path = outdir / "run_config.json"
    run_cfg_path.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[INFO] Saved run config: {run_cfg_path}")

    master_dir = outdir / str(cfg["paths"]["master_dir"])
    master_dir.mkdir(parents=True, exist_ok=True)
    _write_protocol(cfg, master_dir / "research_protocol.md")

    py = _python_cmd()
    env = dict(os.environ)
    env["PYTHONPATH"] = str((root / "src").resolve())

    wave1_dir = outdir / str(cfg["paths"]["wave1_dir"])
    wave2_dir = outdir / str(cfg["paths"]["wave2_dir"])
    rep_dir = outdir / str(cfg["paths"]["report_set_dir"])
    confirm_root = outdir / str(cfg["paths"]["confirmatory_dir"])
    for p in (wave1_dir, wave2_dir, rep_dir, confirm_root):
        p.mkdir(parents=True, exist_ok=True)

    w1_lhs = cfg["wave1_lhs"]
    w2_lhs = cfg["wave2_lhs"]
    hm_sa = cfg["hm_sa"]
    rep = cfg["report_set"]
    sigma_tuning = cfg["sigma_tuning"]

    wave1_lhs_path = wave1_dir / "lhs_runs.csv"
    _run(
        [
            py,
            "scripts/run_lhs.py",
            "--n",
            str(w1_lhs["n"]),
            "--seed",
            str(w1_lhs["seed"]),
            "--seeds",
            str(w1_lhs["seeds"]),
            "--steps",
            str(w1_lhs["steps"]),
            "--window",
            str(w1_lhs["window"]),
            "--balance-ok-threshold",
            str(w1_lhs["balance_ok_threshold"]),
            "--format",
            "csv",
            "--output",
            str(wave1_lhs_path),
        ],
        cwd=root,
        env=env,
    )

    base_targets = root / str(hm_sa["targets"])
    _train_emulator(
        py=py,
        root=root,
        env=env,
        data_path=wave1_lhs_path,
        targets_path=base_targets,
        outdir=wave1_dir,
        hm_sa=hm_sa,
        sigma_mode=str(hm_sa.get("sigma_calibration_mode_wave1", "wave1_empirical")),
        sigma_summary_name="sigma_calibration_summary.csv",
        calibrated_targets_name="targets_calibrated.json",
    )

    wave2_lhs_path = wave2_dir / "lhs_runs_wave2.csv"
    _run(
        [
            py,
            "scripts/run_lhs.py",
            "--n",
            str(w2_lhs["n"]),
            "--seed",
            str(w2_lhs["seed"]),
            "--seeds",
            str(w2_lhs["seeds"]),
            "--steps",
            str(w2_lhs["steps"]),
            "--window",
            str(w2_lhs["window"]),
            "--balance-ok-threshold",
            str(w2_lhs["balance_ok_threshold"]),
            "--bounds-csv",
            str(wave1_dir / "refined_intervals.csv"),
            "--bounds-kind",
            str(w2_lhs["bounds_kind"]),
            "--format",
            "csv",
            "--output",
            str(wave2_lhs_path),
        ],
        cwd=root,
        env=env,
    )

    tuned_targets, tuning_df = _run_wave2_hm_with_tuning(
        py=py,
        root=root,
        env=env,
        wave2_lhs_path=wave2_lhs_path,
        wave2_dir=wave2_dir,
        initial_targets=wave1_dir / "targets_calibrated.json",
        hm_sa=hm_sa,
        sigma_tuning=sigma_tuning,
    )
    (wave2_dir / "targets_for_wave2_final.json").write_text(tuned_targets.read_text(encoding="utf-8"), encoding="utf-8")
    print(f"[INFO] Sigma tuning iterations: {len(tuning_df)}")

    rep_cmd = [
        py,
        "scripts/make_report_set.py",
        "--data",
        str(wave2_lhs_path),
        "--history-matching",
        str(wave2_dir / "history_matching.csv"),
        "--outdir",
        str(rep_dir),
        "--n",
        str(rep["n"]),
        "--steps",
        str(rep["steps"]),
        "--window",
        str(rep["window"]),
        "--seeds",
        str(rep["seeds"]),
        "--min-employment",
        str(rep["min_employment"]),
        "--min-output",
        str(rep["min_output"]),
        "--min-consumption",
        str(rep["min_consumption"]),
        "--min-hh-deposit",
        str(rep["min_hh_deposit"]),
        "--max-bank-resolved-share",
        str(rep["max_bank_resolved_share"]),
        "--max-bank-bailedout-share",
        str(rep["max_bank_bailedout_share"]),
        "--max-haircut-mean",
        str(rep["max_haircut_mean"]),
        "--max-defaults-hh-rate",
        str(rep["max_defaults_hh_rate"]),
        "--max-defaults-firm-rate",
        str(rep["max_defaults_firm_rate"]),
        "--balance-ok-threshold",
        str(rep["balance_ok_threshold"]),
    ]
    if not bool(rep["relax_if_needed"]):
        rep_cmd.append("--no-relax")
    _run(rep_cmd, cwd=root, env=env)

    _run(
        [
            py,
            "scripts/plot_report_runs.py",
            "--runs-dir",
            str(rep_dir),
            "--out",
            str(rep_dir / "rep_plots"),
        ],
        cwd=root,
        env=env,
    )

    confirm_summary_path = master_dir / "confirmatory_summary.json"
    confirm_cfg = cfg["confirmatory"]
    if bool(confirm_cfg.get("enabled", True)):
        c_wave1 = confirm_root / "01_wave1"
        c_wave2 = confirm_root / "02_wave2"
        c_wave1.mkdir(parents=True, exist_ok=True)
        c_wave2.mkdir(parents=True, exist_ok=True)
        seed_offset = int(confirm_cfg.get("seed_offset", 101))

        c_wave1_lhs = c_wave1 / "lhs_runs.csv"
        _run(
            [
                py,
                "scripts/run_lhs.py",
                "--n",
                str(w1_lhs["n"]),
                "--seed",
                str(int(w1_lhs["seed"]) + seed_offset),
                "--seeds",
                str(w1_lhs["seeds"]),
                "--steps",
                str(w1_lhs["steps"]),
                "--window",
                str(w1_lhs["window"]),
                "--balance-ok-threshold",
                str(w1_lhs["balance_ok_threshold"]),
                "--format",
                "csv",
                "--output",
                str(c_wave1_lhs),
            ],
            cwd=root,
            env=env,
        )
        _train_emulator(
            py=py,
            root=root,
            env=env,
            data_path=c_wave1_lhs,
            targets_path=base_targets,
            outdir=c_wave1,
            hm_sa=hm_sa,
            sigma_mode=str(hm_sa.get("sigma_calibration_mode_wave1", "wave1_empirical")),
            sigma_summary_name="sigma_calibration_summary.csv",
            calibrated_targets_name="targets_calibrated.json",
        )

        c_wave2_lhs = c_wave2 / "lhs_runs_wave2.csv"
        _run(
            [
                py,
                "scripts/run_lhs.py",
                "--n",
                str(w2_lhs["n"]),
                "--seed",
                str(int(w2_lhs["seed"]) + seed_offset),
                "--seeds",
                str(w2_lhs["seeds"]),
                "--steps",
                str(w2_lhs["steps"]),
                "--window",
                str(w2_lhs["window"]),
                "--balance-ok-threshold",
                str(w2_lhs["balance_ok_threshold"]),
                "--bounds-csv",
                str(c_wave1 / "refined_intervals.csv"),
                "--bounds-kind",
                str(w2_lhs["bounds_kind"]),
                "--format",
                "csv",
                "--output",
                str(c_wave2_lhs),
            ],
            cwd=root,
            env=env,
        )
        _, c_tuning_df = _run_wave2_hm_with_tuning(
            py=py,
            root=root,
            env=env,
            wave2_lhs_path=c_wave2_lhs,
            wave2_dir=c_wave2,
            initial_targets=c_wave1 / "targets_calibrated.json",
            hm_sa=hm_sa,
            sigma_tuning=sigma_tuning,
        )
        print(f"[INFO] Confirmatory sigma tuning iterations: {len(c_tuning_df)}")

        main_hm = _hm_stats(wave2_dir / "history_matching.csv")
        conf_hm = _hm_stats(c_wave2 / "history_matching.csv")
        main_top2 = _sa_top2(wave2_dir / "sensitivity_uncertainty.csv")
        conf_top2 = _sa_top2(c_wave2 / "sensitivity_uncertainty.csv")
        confirm_payload = {
            "main_nroy_pct": main_hm["nroy_pct"],
            "confirm_nroy_pct": conf_hm["nroy_pct"],
            "nroy_delta_pp": float(conf_hm["nroy_pct"] - main_hm["nroy_pct"]),
            "main_sa_top2": main_top2,
            "confirm_sa_top2": conf_top2,
            "sa_top2_stable": set(main_top2) == set(conf_top2),
        }
        confirm_summary_path.write_text(json.dumps(confirm_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    _run(
        [
            py,
            "scripts/make_research_master_report.py",
            "--root",
            str(outdir),
            "--run-config",
            str(run_cfg_path),
            "--out",
            str(master_dir / "research_core_report.md"),
            "--tables-dir",
            str(master_dir / "research_core_tables"),
            "--confirmatory-summary",
            str(confirm_summary_path),
        ],
        cwd=root,
        env=env,
    )

    print("[DONE] Research core pipeline completed.")


if __name__ == "__main__":
    main()
