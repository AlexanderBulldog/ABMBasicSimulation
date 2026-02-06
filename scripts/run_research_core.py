from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List


def _python_cmd() -> str:
    return sys.executable


def _run(cmd: List[str], cwd: Path, env: Dict[str, str]) -> None:
    print(f"[RUN] {' '.join(cmd)}")
    proc = subprocess.run(cmd, cwd=str(cwd), env=env)
    if proc.returncode != 0:
        raise SystemExit(f"Command failed ({proc.returncode}): {' '.join(cmd)}")


def _cfg(mode: str) -> Dict[str, object]:
    base = {
        "mode": mode,
        "paths": {
            "wave1_dir": "01_wave1",
            "wave2_dir": "02_wave2",
            "report_set_dir": "03_report_set",
            "master_dir": "04_master",
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
            "sa_enable": True,
            "sa_domain": "nroy",
            "sa_reductions": "0.1,0.2,0.3,0.4",
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
            "min_employment": 55,
            "min_output": 60,
            "min_consumption": 60,
            "min_hh_deposit": 0.2,
            "max_bank_resolved_share": 0.06,
            "max_bank_bailedout_share": 0.02,
            "max_haircut_mean": 0.03,
            "max_defaults_hh_rate": 0.01,
            "max_defaults_firm_rate": 0.01,
            "balance_ok_threshold": 0.98,
            "relax_if_needed": True,
        },
        "quality_gate": {
            "nroy_wave2_min_pct": 25.0,
            "nroy_wave2_max_pct": 60.0,
            "imax_wave2_median_max": 3.2,
        },
    }
    if mode == "dry":
        base["wave1_lhs"].update({"n": 20, "steps": 60, "window": 20})  # type: ignore[union-attr]
        base["wave2_lhs"].update({"n": 20, "steps": 60, "window": 20})  # type: ignore[union-attr]
        base["report_set"].update({"n": 2, "steps": 120, "window": 30})  # type: ignore[union-attr]
    return base


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

    py = _python_cmd()
    env = dict(os.environ)
    env["PYTHONPATH"] = str((root / "src").resolve())

    wave1_dir = outdir / str(cfg["paths"]["wave1_dir"])
    wave2_dir = outdir / str(cfg["paths"]["wave2_dir"])
    rep_dir = outdir / str(cfg["paths"]["report_set_dir"])
    master_dir = outdir / str(cfg["paths"]["master_dir"])
    for p in (wave1_dir, wave2_dir, rep_dir, master_dir):
        p.mkdir(parents=True, exist_ok=True)

    w1_lhs = cfg["wave1_lhs"]
    w2_lhs = cfg["wave2_lhs"]
    hm_sa = cfg["hm_sa"]
    rep = cfg["report_set"]

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

    _run(
        [
            py,
            "scripts/train_emulator.py",
            "--data",
            str(wave1_lhs_path),
            "--targets",
            str(hm_sa["targets"]),
            "--min-r2-for-history-matching",
            str(hm_sa["min_r2_for_history_matching"]),
            "--improb-threshold",
            str(hm_sa["improb_threshold"]),
            "--ev-mode",
            str(hm_sa["ev_mode"]),
            "--ev-default-quantile",
            str(hm_sa["ev_default_quantile"]),
            "--sa-enable",
            "--sa-domain",
            str(hm_sa["sa_domain"]),
            "--sa-reductions",
            str(hm_sa["sa_reductions"]),
            "--outdir",
            str(wave1_dir),
        ],
        cwd=root,
        env=env,
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

    _run(
        [
            py,
            "scripts/train_emulator.py",
            "--data",
            str(wave2_lhs_path),
            "--targets",
            str(hm_sa["targets"]),
            "--min-r2-for-history-matching",
            str(hm_sa["min_r2_for_history_matching"]),
            "--improb-threshold",
            str(hm_sa["improb_threshold"]),
            "--ev-mode",
            str(hm_sa["ev_mode"]),
            "--ev-default-quantile",
            str(hm_sa["ev_default_quantile"]),
            "--sa-enable",
            "--sa-domain",
            str(hm_sa["sa_domain"]),
            "--sa-reductions",
            str(hm_sa["sa_reductions"]),
            "--outdir",
            str(wave2_dir),
        ],
        cwd=root,
        env=env,
    )

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
        ],
        cwd=root,
        env=env,
    )

    print("[DONE] Research core pipeline completed.")


if __name__ == "__main__":
    main()
