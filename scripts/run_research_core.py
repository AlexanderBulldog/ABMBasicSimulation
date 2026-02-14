from __future__ import annotations

import argparse
import json
import os
import shutil
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


def _to_bool(v) -> bool:
    if isinstance(v, bool):
        return v
    if v is None:
        return False
    return str(v).strip().lower() in {"1", "true", "yes", "y"}


def _parse_nroy_band(raw: str) -> tuple[float, float]:
    toks = [t.strip() for t in raw.split(",") if t.strip()]
    if len(toks) != 2:
        raise SystemExit("--nroy-band must be in form 'min,max'")
    lo = float(toks[0])
    hi = float(toks[1])
    if lo >= hi:
        raise SystemExit("--nroy-band requires min < max")
    return lo, hi


def _cfg(args: argparse.Namespace) -> Dict[str, object]:
    nroy_min, nroy_max = _parse_nroy_band(args.nroy_band)
    base: Dict[str, object] = {
        "protocol": {
            "version_tag": "research-core-v3",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "goal": "economically_explainable_strict_calibration_for_scientific_presentation",
            "allowed_adjustments": ["sigma_model", "sigma_obs"],
            "andrianakis_style": True,
        },
        "mode": args.mode,
        "profile": args.profile,
        "paths": {
            "wave1_dir": "01_wave1",
            "wave2_dir": "02_wave2",
            "extra_waves_dir": "waves",
            "report_set_dir": "03_report_set",
            "master_dir": "04_master",
            "confirmatory_dir": "05_confirmatory",
        },
        "wave_control": {
            "adaptive_waves": bool(args.adaptive_waves),
            "wave_min": int(args.wave_min),
            "wave_max": int(args.wave_max),
            "stop_rule": str(args.stop_rule),
            "stability_shrink_delta_pp_max": 5.0,
        },
        "wave_lhs": {
            "n": 350,
            "seed": 0,
            "seeds": "0,1,2",
            "steps": 220,
            "window": 30,
            "balance_ok_threshold": 0.60,
            "bounds_kind": "p05p95",
            "wave1_bounds_csv": str(args.wave1_bounds_csv or ""),
            "wave1_bounds_kind": str(args.wave1_bounds_kind or "p05p95"),
        },
        "hm_sa": {
            "targets": str(args.targets_path or "scripts/targets_report.json"),
            "min_r2_for_history_matching": float(args.min_r2_for_history_matching),
            "improb_threshold": float(args.improb_threshold_base),
            "ev_mode": "seed_replicates",
            "ev_default_quantile": float(args.ev_default_quantile),
            "sigma_calibration_mode_wave1": "fixed",
            "sigma_calibration_mode_wavek": "fixed",
            "sa_enable": True,
            "sa_domain": "nroy",
            "sa_reductions": "0.1,0.2,0.3,0.4",
        },
        "sigma_tuning": {
            "enabled": True,
            "max_iters": int(args.sigma_tuning_max_iters),
            "step": float(args.sigma_tuning_step),
            "nroy_min_pct": nroy_min,
            "nroy_max_pct": nroy_max,
        },
        "report_set": {
            "n": 10,
            "steps": 500,
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
            "nroy_wave2_min_pct": nroy_min,
            "nroy_wave2_max_pct": nroy_max,
            "legacy_nroy_wave2_min_pct": 25.0,
            "legacy_nroy_wave2_max_pct": 60.0,
            "active_nroy_band_label": "active-v3",
            "legacy_nroy_band_label": "legacy-reference",
            "use_dual_contour": True,
            "imax_wave2_median_max": float(args.imax_median_max),
            "imax_wave2_p95_max": float(args.imax_p95_max),
            "rep_employment_min": 45.0,
            "rep_output_min": 50.0,
            "rep_consumption_min": 50.0,
            "confirmatory_nroy_tol_pp": 5.0,
            "bad_run_pct_w2_max": 5.0,
            "credit_rejections_p90_max": 30.0,
            "credit_rejections_p90_abs_max": float(args.credit_p90_abs_max),
            "credit_rejections_baseline_p90": float(args.credit_p90_baseline),
            "credit_rejections_improvement_min_pct": float(args.credit_p90_improve_min_pct),
            "credit_rejection_rate_mean_max": float(args.credit_rejection_rate_mean_max),
            "emu_r2_cv_median_min": float(args.emu_r2_median_min),
            "emu_r2_share_threshold": float(args.emu_r2_share_threshold),
            "emu_r2_share_min": float(args.emu_r2_share_min),
        },
    }

    if args.profile == "nightly_v3":
        base["wave_lhs"] = {**base["wave_lhs"], "n": 350, "steps": 220, "window": 30}  # type: ignore[index]
        base["report_set"] = {**base["report_set"], "n": 10, "steps": 500, "window": 60}  # type: ignore[index]
        base["confirmatory"] = {**base["confirmatory"], "enabled": True}  # type: ignore[index]

    if args.mode == "dry":
        base["wave_lhs"] = {**base["wave_lhs"], "n": 20, "steps": 60, "window": 20}  # type: ignore[index]
        base["report_set"] = {**base["report_set"], "n": 2, "steps": 120, "window": 30}  # type: ignore[index]
        base["sigma_tuning"] = {**base["sigma_tuning"], "max_iters": 1}  # type: ignore[index]
        base["confirmatory"] = {**base["confirmatory"], "enabled": False}  # type: ignore[index]
    elif args.mode == "preflight":
        base["wave_lhs"] = {**base["wave_lhs"], "n": 120, "steps": 120, "window": 20}  # type: ignore[index]
        base["report_set"] = {**base["report_set"], "n": 4, "steps": 220, "window": 40}  # type: ignore[index]
        base["confirmatory"] = {**base["confirmatory"], "enabled": False}  # type: ignore[index]

    # Optional explicit overrides for experiment sweeps.
    if int(args.wave_n) > 0:
        base["wave_lhs"] = {**base["wave_lhs"], "n": int(args.wave_n)}  # type: ignore[index]
    if int(args.wave_steps) > 0:
        base["wave_lhs"] = {**base["wave_lhs"], "steps": int(args.wave_steps)}  # type: ignore[index]
    if int(args.report_n) > 0:
        base["report_set"] = {**base["report_set"], "n": int(args.report_n)}  # type: ignore[index]
    if int(args.report_steps) > 0:
        base["report_set"] = {**base["report_set"], "steps": int(args.report_steps)}  # type: ignore[index]

    return base


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
    return sorted(df[df["rank_at_max_reduction"] <= 2]["component"].astype(str).dropna().unique().tolist())


def _emulator_gate_stats(scores_path: Path, share_threshold: float, min_r2_gate: float) -> Dict[str, float]:
    df = pd.read_csv(scores_path)
    trained = df[~df["skipped"].map(_to_bool)].copy() if "skipped" in df.columns else df.copy()
    if trained.empty or "gpr_r2_cv_mean" not in trained.columns:
        return {
            "trained": float(len(trained)),
            "gate_metrics": 0.0,
            "gpr_r2_cv_median": float("nan"),
            "gpr_r2_cv_share_ge": float("nan"),
        }
    gate_df = trained[trained["gpr_r2_cv_mean"] >= min_r2_gate].copy()
    if gate_df.empty:
        return {
            "trained": float(len(trained)),
            "gate_metrics": 0.0,
            "gpr_r2_cv_median": float("nan"),
            "gpr_r2_cv_share_ge": float("nan"),
        }
    med = float(gate_df["gpr_r2_cv_mean"].median())
    share = float((gate_df["gpr_r2_cv_mean"] >= share_threshold).mean())
    return {"trained": float(len(trained)), "gate_metrics": float(len(gate_df)), "gpr_r2_cv_median": med, "gpr_r2_cv_share_ge": share}


def _scale_sigma_model_in_targets(src: Path, dst: Path, factor: float, reason: str) -> None:
    payload = json.loads(src.read_text(encoding="utf-8"))
    if "_meta" not in payload or not isinstance(payload["_meta"], dict):
        payload["_meta"] = {}
    payload["_meta"]["sigma_tuning_factor"] = factor
    payload["_meta"]["sigma_tuning_reason"] = reason
    for metric, spec in list(payload.items()):
        if not isinstance(metric, str) or metric.startswith("_") or not isinstance(spec, dict):
            continue
        if "sigma_model" in spec:
            spec["sigma_model"] = max(1e-9, float(spec.get("sigma_model", 0.0)) * factor)
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
        "## Quality Gates (v3)\n",
        f"- Active NROY band: [{q.get('nroy_wave2_min_pct', 25)}, {q.get('nroy_wave2_max_pct', 65)}]%\n",
        f"- I_max median < {q.get('imax_wave2_median_max', 3.0)}\n",
        f"- I_max p95 < {q.get('imax_wave2_p95_max', 4.5)}\n",
        f"- Emulator median CV-R2 >= {q.get('emu_r2_cv_median_min', 0.60)}\n",
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


def _sigma_tuning_decision(nroy: float, nroy_min: float, nroy_max: float, step: float) -> tuple[float, str] | None:
    if nroy_min <= nroy <= nroy_max:
        return None
    if nroy > nroy_max:
        return (1.0 - step, "nroy_above_max_tighten_sigma_model")
    return (1.0 + step, "nroy_below_min_relax_sigma_model")


def _run_hm_with_tuning(
    py: str,
    root: Path,
    env: Dict[str, str],
    data_path: Path,
    wave_dir: Path,
    initial_targets: Path,
    hm_sa: Dict[str, object],
    sigma_tuning: Dict[str, object],
    wave_id: int,
) -> Tuple[Path, pd.DataFrame]:
    rows: List[Dict[str, object]] = []
    targets_curr = initial_targets
    max_iters = int(sigma_tuning["max_iters"])
    step = float(sigma_tuning["step"])
    nmin = float(sigma_tuning["nroy_min_pct"])
    nmax = float(sigma_tuning["nroy_max_pct"])

    for i in range(max_iters + 1):
        _train_emulator(
            py=py,
            root=root,
            env=env,
            data_path=data_path,
            targets_path=targets_curr,
            outdir=wave_dir,
            hm_sa=hm_sa,
            sigma_mode=str(hm_sa.get("sigma_calibration_mode_wavek", "fixed")),
            sigma_summary_name=f"sigma_calibration_summary_iter{i}.csv",
            calibrated_targets_name=f"targets_calibrated_iter{i}.json",
        )
        stats = _hm_stats(wave_dir / "history_matching.csv")
        rows.append({"wave": wave_id, "iter": i, "targets_path": str(targets_curr), **stats})
        decision = _sigma_tuning_decision(stats["nroy_pct"], nmin, nmax, step)
        if decision is None or i >= max_iters:
            break
        factor, reason = decision
        next_targets = wave_dir / f"targets_tuned_iter{i + 1}.json"
        _scale_sigma_model_in_targets(targets_curr, next_targets, factor=factor, reason=reason)
        targets_curr = next_targets

    tuning_df = pd.DataFrame(rows)
    tuning_df.to_csv(wave_dir / "sigma_tuning_iterations.csv", index=False)
    tuning_df.to_csv(wave_dir / f"sigma_tuning_iterations_wave{wave_id}.csv", index=False)
    return targets_curr, tuning_df


def _wave_n_for_index(base_n: int, idx: int, profile: str) -> int:
    if profile == "nightly_v3" and idx >= 4:
        return 250
    return base_n


def _wave_dir(outdir: Path, paths_cfg: Dict[str, object], idx: int) -> Path:
    if idx == 1:
        return outdir / str(paths_cfg["wave1_dir"])
    if idx == 2:
        return outdir / str(paths_cfg["wave2_dir"])
    return outdir / str(paths_cfg["extra_waves_dir"]) / f"wave_{idx:02d}"


def _evaluate_wave_blocking(wave_stats: Dict[str, float], emu_stats: Dict[str, float], q: Dict[str, object]) -> bool:
    return bool(
        float(q["nroy_wave2_min_pct"]) <= wave_stats["nroy_pct"] <= float(q["nroy_wave2_max_pct"])
        and wave_stats["Imax_median"] < float(q["imax_wave2_median_max"])
        and wave_stats["Imax_p95"] < float(q["imax_wave2_p95_max"])
        and pd.notna(emu_stats["gpr_r2_cv_median"])
        and emu_stats["gpr_r2_cv_median"] >= float(q["emu_r2_cv_median_min"])
        and pd.notna(emu_stats["gpr_r2_cv_share_ge"])
        and emu_stats["gpr_r2_cv_share_ge"] >= float(q["emu_r2_share_min"])
    )


def _sync_final_wave_to_wave2(final_wave_dir: Path, canonical_wave2_dir: Path) -> None:
    if final_wave_dir.resolve() == canonical_wave2_dir.resolve():
        return
    if canonical_wave2_dir.exists():
        shutil.rmtree(canonical_wave2_dir)
    shutil.copytree(final_wave_dir, canonical_wave2_dir)


def _run_adaptive_waves(root: Path, outdir: Path, cfg: Dict[str, object], py: str, env: Dict[str, str]) -> Dict[str, object]:
    paths_cfg = dict(cfg["paths"])
    wave_cfg = dict(cfg["wave_lhs"])
    hm_sa = dict(cfg["hm_sa"])
    sigma_tuning = dict(cfg["sigma_tuning"])
    q = dict(cfg["quality_gate"])
    control = dict(cfg["wave_control"])

    wave_min = int(control["wave_min"])
    wave_max = int(control["wave_max"])
    adaptive = bool(control["adaptive_waves"])
    shrink_delta_max = float(control["stability_shrink_delta_pp_max"])

    base_targets = root / str(hm_sa["targets"])
    prev_bounds: Path | None = None
    prev_targets: Path = base_targets
    prev_wave_pass = False
    prev_shrink = float("nan")
    rows: List[Dict[str, object]] = []
    final_wave_dir: Path | None = None
    final_wave_idx = 0
    stop_reason = "max_waves_reached"

    for idx in range(1, wave_max + 1):
        wdir = _wave_dir(outdir, paths_cfg, idx)
        wdir.mkdir(parents=True, exist_ok=True)
        wave_n = _wave_n_for_index(int(wave_cfg["n"]), idx, str(cfg["profile"]))

        data_name = "lhs_runs.csv" if idx == 1 else f"lhs_runs_wave{idx}.csv"
        data_path = wdir / data_name
        lhs_cmd = [
            py, "scripts/run_lhs.py", "--n", str(wave_n), "--seed", str(int(wave_cfg["seed"]) + idx - 1),
            "--seeds", str(wave_cfg["seeds"]), "--steps", str(wave_cfg["steps"]), "--window", str(wave_cfg["window"]),
            "--balance-ok-threshold", str(wave_cfg["balance_ok_threshold"]), "--format", "csv", "--output", str(data_path),
        ]
        if idx == 1 and str(wave_cfg.get("wave1_bounds_csv", "")).strip():
            lhs_cmd.extend(
                [
                    "--bounds-csv",
                    str(wave_cfg["wave1_bounds_csv"]),
                    "--bounds-kind",
                    str(wave_cfg.get("wave1_bounds_kind", wave_cfg["bounds_kind"])),
                ]
            )
        if idx > 1 and prev_bounds is not None:
            lhs_cmd.extend(["--bounds-csv", str(prev_bounds), "--bounds-kind", str(wave_cfg["bounds_kind"])])
        _run(lhs_cmd, cwd=root, env=env)

        if idx == 1:
            _train_emulator(
                py=py, root=root, env=env, data_path=data_path, targets_path=base_targets, outdir=wdir, hm_sa=hm_sa,
                sigma_mode=str(hm_sa.get("sigma_calibration_mode_wave1", "wave1_empirical")),
                sigma_summary_name="sigma_calibration_summary.csv", calibrated_targets_name="targets_calibrated.json",
            )
            tuned_targets = wdir / "targets_calibrated.json"
            pd.DataFrame([{"wave": idx, "iter": 0, **_hm_stats(wdir / "history_matching.csv")}]).to_csv(
                wdir / "sigma_tuning_iterations.csv", index=False
            )
        else:
            tuned_targets, _ = _run_hm_with_tuning(
                py=py, root=root, env=env, data_path=data_path, wave_dir=wdir, initial_targets=prev_targets,
                hm_sa=hm_sa, sigma_tuning=sigma_tuning, wave_id=idx,
            )

        final_targets_path = wdir / "targets_for_wave_final.json"
        final_targets_path.write_text(tuned_targets.read_text(encoding="utf-8"), encoding="utf-8")
        hm_stats = _hm_stats(wdir / "history_matching.csv")
        emu_stats = _emulator_gate_stats(
            wdir / "emulator_scores.csv",
            float(q["emu_r2_share_threshold"]),
            float(hm_sa.get("min_r2_for_history_matching", 0.25)),
        )
        ri = pd.read_csv(wdir / "refined_intervals.csv")
        shrink_mean = float(ri["shrink_pct"].mean()) if not ri.empty and "shrink_pct" in ri.columns else float("nan")
        shrink_delta = float("nan") if pd.isna(prev_shrink) else abs(shrink_mean - prev_shrink)
        wave_pass = _evaluate_wave_blocking(hm_stats, emu_stats, q)
        stable_candidate = bool(idx >= 2 and wave_pass and prev_wave_pass and pd.notna(shrink_delta) and shrink_delta < shrink_delta_max)

        rows.append(
            {
                "wave": idx,
                "wave_dir": str(wdir),
                "n": wave_n,
                "steps": int(wave_cfg["steps"]),
                "nroy_pct": hm_stats["nroy_pct"],
                "Imax_median": hm_stats["Imax_median"],
                "Imax_p95": hm_stats["Imax_p95"],
                "gpr_r2_cv_median": emu_stats["gpr_r2_cv_median"],
                "gpr_r2_cv_share_ge_threshold": emu_stats["gpr_r2_cv_share_ge"],
                "shrink_mean": shrink_mean,
                "shrink_delta": shrink_delta,
                "wave_blocking_pass": wave_pass,
                "stable_candidate": stable_candidate,
            }
        )

        prev_bounds = wdir / "refined_intervals.csv"
        prev_targets = final_targets_path
        prev_shrink = shrink_mean
        prev_wave_pass = wave_pass
        final_wave_dir = wdir
        final_wave_idx = idx

        if adaptive and idx >= wave_min and stable_candidate:
            stop_reason = "stability_reached"
            break

    if final_wave_dir is None:
        raise SystemExit("No wave executed")

    canonical_wave2_dir = outdir / str(paths_cfg["wave2_dir"])
    _sync_final_wave_to_wave2(final_wave_dir, canonical_wave2_dir)

    pd.DataFrame(rows).to_csv(outdir / "waves_summary.csv", index=False)
    (outdir / "stopping_diagnostics.json").write_text(
        json.dumps(
            {
                "stop_reason": stop_reason,
                "final_wave": final_wave_idx,
                "waves_run": len(rows),
                "wave_min": wave_min,
                "wave_max": wave_max,
                "adaptive_waves": adaptive,
                "stability_shrink_delta_pp_max": shrink_delta_max,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    return {
        "final_wave": final_wave_idx,
        "final_wave_dir": final_wave_dir,
        "wave1_dir": outdir / str(paths_cfg["wave1_dir"]),
        "wave2_dir": canonical_wave2_dir,
        "waves_summary": outdir / "waves_summary.csv",
        "stopping_diagnostics": outdir / "stopping_diagnostics.json",
    }


def _run_confirmatory_for_final_wave(
    root: Path,
    confirm_root: Path,
    py: str,
    env: Dict[str, str],
    cfg: Dict[str, object],
    final_bounds_csv: Path,
    final_targets: Path,
    final_wave_id: int,
) -> Dict[str, object]:
    wave_cfg = dict(cfg["wave_lhs"])
    hm_sa = dict(cfg["hm_sa"])
    seed_offset = int(dict(cfg["confirmatory"]).get("seed_offset", 101))

    c_wave_dir = confirm_root / f"wave_{final_wave_id:02d}"
    c_wave_dir.mkdir(parents=True, exist_ok=True)

    data_path = c_wave_dir / f"lhs_runs_confirm_wave{final_wave_id}.csv"
    _run(
        [
            py,
            "scripts/run_lhs.py",
            "--n",
            str(int(wave_cfg["n"])),
            "--seed",
            str(int(wave_cfg["seed"]) + seed_offset + final_wave_id - 1),
            "--seeds",
            str(wave_cfg["seeds"]),
            "--steps",
            str(wave_cfg["steps"]),
            "--window",
            str(wave_cfg["window"]),
            "--balance-ok-threshold",
            str(wave_cfg["balance_ok_threshold"]),
            "--bounds-csv",
            str(final_bounds_csv),
            "--bounds-kind",
            str(wave_cfg["bounds_kind"]),
            "--format",
            "csv",
            "--output",
            str(data_path),
        ],
        cwd=root,
        env=env,
    )

    _train_emulator(
        py=py,
        root=root,
        env=env,
        data_path=data_path,
        targets_path=final_targets,
        outdir=c_wave_dir,
        hm_sa=hm_sa,
        sigma_mode=str(hm_sa.get("sigma_calibration_mode_wavek", "fixed")),
        sigma_summary_name="sigma_calibration_summary.csv",
        calibrated_targets_name="targets_calibrated.json",
    )

    return {
        "dir": c_wave_dir,
        "hm": c_wave_dir / "history_matching.csv",
        "sa": c_wave_dir / "sensitivity_uncertainty.csv",
    }


def _evaluate_go_no_go(quality_gates_path: Path) -> Dict[str, object]:
    qg = pd.read_csv(quality_gates_path)
    blocking = qg[qg["blocking"].map(_to_bool)] if "blocking" in qg.columns else qg
    checks = [
        {
            "check": str(row.get("check", "")),
            "pass": bool(_to_bool(row.get("pass", False))),
            "value": str(row.get("value", "")),
            "reason_code": str(row.get("reason_code", "")),
        }
        for _, row in blocking.iterrows()
    ]
    return {"pass": all(c["pass"] for c in checks), "checks": checks, "n_blocking": len(checks)}


def _preflight_improb_thresholds(base: float, bump: float = 0.15, max_value: float = 2.9) -> list[float]:
    first = float(base)
    second = float(min(base + bump, max_value))
    return [first] if abs(second - first) < 1e-12 else [first, second]

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run research-core v3 pipeline.")
    p.add_argument("--mode", choices=["dry", "full", "preflight"], default="full")
    p.add_argument("--profile", choices=["standard", "nightly_v3"], default="standard")
    p.add_argument("--outdir", type=str, default="output_research_core_v3")
    p.add_argument("--root", type=str, default=".")
    p.add_argument("--wave-min", type=int, default=2)
    p.add_argument("--wave-max", type=int, default=5)
    p.add_argument("--adaptive-waves", type=_to_bool, default=True)
    p.add_argument("--stop-rule", choices=["stability_v1"], default="stability_v1")
    p.add_argument("--nroy-band", type=str, default="25,65")
    p.add_argument("--imax-median-max", type=float, default=3.0)
    p.add_argument("--imax-p95-max", type=float, default=4.5)
    p.add_argument("--emu-r2-median-min", type=float, default=0.60)
    p.add_argument("--emu-r2-share-threshold", type=float, default=0.30)
    p.add_argument("--emu-r2-share-min", type=float, default=0.65)
    p.add_argument("--targets-path", type=str, default="scripts/targets_report.json")
    p.add_argument("--wave1-bounds-csv", type=str, default="")
    p.add_argument("--wave1-bounds-kind", choices=["p05p95", "minmax"], default="p05p95")
    p.add_argument("--wave-n", type=int, default=0, help="Override wave LHS n if >0")
    p.add_argument("--wave-steps", type=int, default=0, help="Override wave steps if >0")
    p.add_argument("--report-n", type=int, default=0, help="Override report_set n if >0")
    p.add_argument("--report-steps", type=int, default=0, help="Override report_set steps if >0")
    p.add_argument("--improb-threshold-base", type=float, default=2.6)
    p.add_argument("--min-r2-for-history-matching", type=float, default=0.30)
    p.add_argument("--ev-default-quantile", type=float, default=0.75)
    p.add_argument("--sigma-tuning-step", type=float, default=0.12)
    p.add_argument("--sigma-tuning-max-iters", type=int, default=6)
    p.add_argument("--credit-p90-abs-max", type=float, default=90.0)
    p.add_argument("--credit-p90-baseline", type=float, default=104.805)
    p.add_argument("--credit-p90-improve-min-pct", type=float, default=15.0)
    p.add_argument("--credit-rejection-rate-mean-max", type=float, default=0.80)
    p.add_argument("--preflight-launch-full", type=_to_bool, default=True)
    return p.parse_args()


def _execute_pipeline(root: Path, outdir: Path, cfg: Dict[str, object]) -> Dict[str, Path]:
    outdir.mkdir(parents=True, exist_ok=True)
    run_cfg_path = outdir / "run_config.json"
    run_cfg_path.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[INFO] Saved run config: {run_cfg_path}")

    paths_cfg = dict(cfg["paths"])
    master_dir = outdir / str(paths_cfg["master_dir"])
    master_dir.mkdir(parents=True, exist_ok=True)
    _write_protocol(cfg, master_dir / "research_protocol.md")

    py = _python_cmd()
    env = dict(os.environ)
    env["PYTHONPATH"] = str((root / "src").resolve())

    for p in (
        outdir / str(paths_cfg["wave1_dir"]),
        outdir / str(paths_cfg["wave2_dir"]),
        outdir / str(paths_cfg["extra_waves_dir"]),
        outdir / str(paths_cfg["report_set_dir"]),
        outdir / str(paths_cfg["confirmatory_dir"]),
    ):
        p.mkdir(parents=True, exist_ok=True)

    wave_exec = _run_adaptive_waves(root=root, outdir=outdir, cfg=cfg, py=py, env=env)
    wave2_dir = Path(wave_exec["wave2_dir"])
    wave2_data = wave2_dir / "lhs_runs_wave2.csv"
    if not wave2_data.exists():
        candidates = sorted(wave2_dir.glob("lhs_runs_wave*.csv")) + sorted(wave2_dir.glob("lhs_runs*.csv"))
        if not candidates:
            raise SystemExit("Could not find wave2 LHS dataset in final wave directory")
        wave2_data = candidates[0]

    rep_cfg = dict(cfg["report_set"])
    rep_dir = outdir / str(paths_cfg["report_set_dir"])
    rep_cmd = [
        py,
        "scripts/make_report_set.py",
        "--data",
        str(wave2_data),
        "--history-matching",
        str(wave2_dir / "history_matching.csv"),
        "--outdir",
        str(rep_dir),
        "--n",
        str(rep_cfg["n"]),
        "--steps",
        str(rep_cfg["steps"]),
        "--window",
        str(rep_cfg["window"]),
        "--seeds",
        str(rep_cfg["seeds"]),
        "--min-employment",
        str(rep_cfg["min_employment"]),
        "--min-output",
        str(rep_cfg["min_output"]),
        "--min-consumption",
        str(rep_cfg["min_consumption"]),
        "--min-hh-deposit",
        str(rep_cfg["min_hh_deposit"]),
        "--max-bank-resolved-share",
        str(rep_cfg["max_bank_resolved_share"]),
        "--max-bank-bailedout-share",
        str(rep_cfg["max_bank_bailedout_share"]),
        "--max-haircut-mean",
        str(rep_cfg["max_haircut_mean"]),
        "--max-defaults-hh-rate",
        str(rep_cfg["max_defaults_hh_rate"]),
        "--max-defaults-firm-rate",
        str(rep_cfg["max_defaults_firm_rate"]),
        "--balance-ok-threshold",
        str(rep_cfg["balance_ok_threshold"]),
    ]
    if not bool(rep_cfg["relax_if_needed"]):
        rep_cmd.append("--no-relax")
    _run(rep_cmd, cwd=root, env=env)

    _run(
        [py, "scripts/plot_report_runs.py", "--runs-dir", str(rep_dir), "--out", str(rep_dir / "rep_plots")],
        cwd=root,
        env=env,
    )

    confirm_summary_path = master_dir / "confirmatory_summary.json"
    confirm_cfg = dict(cfg["confirmatory"])
    if bool(confirm_cfg.get("enabled", True)):
        confirm_root = outdir / str(paths_cfg["confirmatory_dir"])
        final_wave_dir = Path(wave_exec["final_wave_dir"])
        c_out = _run_confirmatory_for_final_wave(
            root=root,
            confirm_root=confirm_root,
            py=py,
            env=env,
            cfg=cfg,
            final_bounds_csv=final_wave_dir / "refined_intervals.csv",
            final_targets=final_wave_dir / "targets_for_wave_final.json",
            final_wave_id=int(wave_exec["final_wave"]),
        )
        main_hm = _hm_stats(wave2_dir / "history_matching.csv")
        conf_hm = _hm_stats(Path(c_out["hm"]))
        main_top2 = _sa_top2(wave2_dir / "sensitivity_uncertainty.csv")
        conf_top2 = _sa_top2(Path(c_out["sa"]))
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
            "--waves-summary",
            str(outdir / "waves_summary.csv"),
            "--stopping-diagnostics",
            str(outdir / "stopping_diagnostics.json"),
        ],
        cwd=root,
        env=env,
    )

    return {
        "run_config": run_cfg_path,
        "quality_gates": master_dir / "research_core_tables" / "quality_gates.csv",
        "waves_summary": outdir / "waves_summary.csv",
        "stopping_diagnostics": outdir / "stopping_diagnostics.json",
        "report": master_dir / "research_core_report.md",
    }


def main() -> None:
    args = parse_args()
    if args.wave_min < 2:
        raise SystemExit("--wave-min must be >= 2")
    if args.wave_max < args.wave_min:
        raise SystemExit("--wave-max must be >= --wave-min")

    root = Path(args.root).resolve()
    outdir = (root / args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    if args.mode in {"dry", "full"}:
        cfg = _cfg(args)
        _execute_pipeline(root=root, outdir=outdir, cfg=cfg)
        print("[DONE] Research core v3 pipeline completed.")
        return

    cfg_preflight = _cfg(args)
    hm_sa_cfg = dict(cfg_preflight["hm_sa"])
    thresholds = _preflight_improb_thresholds(float(hm_sa_cfg["improb_threshold"]), bump=0.15, max_value=2.9)
    attempts: list[dict[str, object]] = []
    preflight_pass = False
    for idx, thr in enumerate(thresholds):
        hm_sa_cfg["improb_threshold"] = float(thr)
        cfg_preflight["hm_sa"] = hm_sa_cfg
        print(f"[INFO] Preflight attempt {idx + 1}/{len(thresholds)} with improb_threshold={thr:.2f}")
        artifacts = _execute_pipeline(root=root, outdir=outdir, cfg=cfg_preflight)
        go_no_go = _evaluate_go_no_go(artifacts["quality_gates"])
        attempts.append({"attempt": idx + 1, "improb_threshold": thr, "pass": bool(go_no_go["pass"]), "checks": go_no_go["checks"]})
        if bool(go_no_go["pass"]):
            preflight_pass = True
            break

    status_payload: Dict[str, object] = {"mode": "preflight", "profile": args.profile, "preflight_pass": preflight_pass, "attempts": attempts}

    if preflight_pass and bool(args.preflight_launch_full):
        full_args = argparse.Namespace(**vars(args))
        full_args.mode = "full"
        full_args.profile = "nightly_v3"
        cfg_full = _cfg(full_args)
        full_outdir = outdir / "overnight_full"
        print("[INFO] Preflight PASS -> launching overnight full run (nightly_v3).")
        full_artifacts = _execute_pipeline(root=root, outdir=full_outdir, cfg=cfg_full)
        status_payload["overnight_launched"] = True
        status_payload["overnight_outdir"] = str(full_outdir)
        status_payload["overnight_report"] = str(full_artifacts["report"])
    elif preflight_pass and not bool(args.preflight_launch_full):
        print("[INFO] Preflight PASS; full run launch skipped by flag.")
        status_payload["overnight_launched"] = False
        status_payload["reason"] = "preflight_pass_full_skipped"
    else:
        print("[WARN] Preflight failed after retry; overnight full run is blocked.")
        status_payload["overnight_launched"] = False
        status_payload["reason"] = "preflight_go_no_go_failed"

    status_path = outdir / "go_no_go_status.json"
    status_path.write_text(json.dumps(status_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[INFO] Saved go/no-go status: {status_path}")
    print("[DONE] Preflight flow completed.")


if __name__ == "__main__":
    main()
