from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd


@dataclass
class Experiment:
    name: str
    args: Dict[str, Any]


def _python() -> str:
    return sys.executable


def _bool(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    return str(v).strip().lower() in {"1", "true", "yes", "y"}


def _load_blocking_summary(qg_path: Path) -> Dict[str, Any]:
    if not qg_path.exists():
        return {"blocking_pass": False, "n_blocking": 0, "n_failed": 0, "failed_checks": ["missing_quality_gates"]}
    qg = pd.read_csv(qg_path)
    if "blocking" in qg.columns:
        blk = qg[qg["blocking"].map(_bool)].copy()
    else:
        blk = qg.copy()
    if blk.empty:
        return {"blocking_pass": False, "n_blocking": 0, "n_failed": 0, "failed_checks": ["no_blocking_rows"]}
    failed = blk[~blk["pass"].map(_bool)].copy()
    return {
        "blocking_pass": bool(failed.empty),
        "n_blocking": int(len(blk)),
        "n_failed": int(len(failed)),
        "failed_checks": failed["check"].astype(str).tolist() if "check" in failed.columns else ["unknown"],
    }


def _extract_key_metrics(qg_path: Path) -> Dict[str, str]:
    keys = [
        "NROY wave2 in [25,65]%",
        "I_max median wave2 < 3.0",
        "I_max p95 wave2 < 4.5",
        "Emulator CV-R2 median >= 0.60",
        "Emulator CV-R2 share>=0.30 >= 0.65",
        "Structural: CreditRejections p90 <= 30",
    ]
    out: Dict[str, str] = {}
    if not qg_path.exists():
        return out
    qg = pd.read_csv(qg_path)
    for k in keys:
        row = qg[qg["check"] == k]
        if row.empty:
            continue
        r = row.iloc[0]
        out[k] = f"{bool(_bool(r['pass']))}:{r['value']}"
    return out


def _blend_bounds_csv(a: Path, b: Path, out_csv: Path) -> Path:
    da = pd.read_csv(a)
    db = pd.read_csv(b)
    ma = {str(r["param"]): r for _, r in da.iterrows()}
    mb = {str(r["param"]): r for _, r in db.iterrows()}
    rows: List[Dict[str, float | str]] = []
    for p in sorted(set(ma.keys()) & set(mb.keys())):
        ra = ma[p]
        rb = mb[p]
        low = max(float(ra["nroy_p05"]), float(rb["nroy_p05"]))
        high = min(float(ra["nroy_p95"]), float(rb["nroy_p95"]))
        if high <= low:
            low = max(float(ra["nroy_min"]), float(rb["nroy_min"]))
            high = min(float(ra["nroy_max"]), float(rb["nroy_max"]))
        if high <= low:
            continue
        rows.append(
            {
                "param": p,
                "nroy_min": low,
                "nroy_max": high,
                "nroy_p05": low,
                "nroy_p95": high,
            }
        )
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    return out_csv


def _campaign_experiments(root: Path, campaign_dir: Path) -> List[Experiment]:
    baseline_bounds = root / "output_research_core_v3" / "waves" / "wave_03" / "refined_intervals.csv"
    credit_bounds = root / "output_research_core_v3_tuned2" / "waves" / "wave_05" / "refined_intervals.csv"
    blended = campaign_dir / "blended_wave1_bounds.csv"
    if baseline_bounds.exists() and credit_bounds.exists():
        _blend_bounds_csv(baseline_bounds, credit_bounds, blended)

    exps: List[Experiment] = [
        Experiment(
            name="exp01_baseline_dense",
            args={
                "wave_n": 180,
                "wave_steps": 140,
                "report_n": 4,
                "report_steps": 220,
                "sigma_tuning_step": 0.10,
                "sigma_tuning_max_iters": 4,
                "ev_default_quantile": 0.80,
            },
        ),
        Experiment(
            name="exp02_emu_focus_high_n",
            args={
                "wave_n": 220,
                "wave_steps": 150,
                "report_n": 4,
                "report_steps": 220,
                "sigma_tuning_step": 0.08,
                "sigma_tuning_max_iters": 3,
                "ev_default_quantile": 0.80,
            },
        ),
    ]
    if baseline_bounds.exists():
        exps.append(
            Experiment(
                name="exp03_wave1_from_baseline_bounds",
                args={
                    "wave1_bounds_csv": str(baseline_bounds),
                    "wave1_bounds_kind": "p05p95",
                    "wave_n": 200,
                    "wave_steps": 140,
                    "sigma_tuning_step": 0.10,
                    "sigma_tuning_max_iters": 4,
                    "ev_default_quantile": 0.80,
                },
            )
        )
    if credit_bounds.exists():
        exps.append(
            Experiment(
                name="exp04_wave1_from_credit_bounds",
                args={
                    "wave1_bounds_csv": str(credit_bounds),
                    "wave1_bounds_kind": "p05p95",
                    "wave_n": 200,
                    "wave_steps": 140,
                    "sigma_tuning_step": 0.10,
                    "sigma_tuning_max_iters": 4,
                    "ev_default_quantile": 0.80,
                },
            )
        )
    if blended.exists():
        exps.append(
            Experiment(
                name="exp05_wave1_from_blended_bounds",
                args={
                    "wave1_bounds_csv": str(blended),
                    "wave1_bounds_kind": "p05p95",
                    "wave_n": 220,
                    "wave_steps": 150,
                    "sigma_tuning_step": 0.08,
                    "sigma_tuning_max_iters": 3,
                    "ev_default_quantile": 0.80,
                },
            )
        )
    return exps


def _adaptive_followup(prev: Experiment, result: Dict[str, Any], next_idx: int, root: Path) -> Experiment | None:
    failed: List[str] = [str(x) for x in result.get("failed_checks", [])]
    base = dict(prev.args)

    # If emulator gates fail, increase design density and reduce sigma step to stabilize CV quality.
    if any("Emulator CV-R2" in c for c in failed):
        base["wave_n"] = min(320, int(base.get("wave_n", 180)) + 40)
        base["wave_steps"] = min(180, int(base.get("wave_steps", 140)) + 10)
        base["sigma_tuning_step"] = max(0.06, float(base.get("sigma_tuning_step", 0.10)) - 0.02)
        base["sigma_tuning_max_iters"] = min(6, int(base.get("sigma_tuning_max_iters", 4)) + 1)
        return Experiment(name=f"exp{next_idx:02d}_adapt_emu", args=base)

    # If structural credit gate fails, bias first wave to credit-friendly region if available.
    if any("CreditRejections" in c for c in failed):
        credit_bounds = root / "output_research_core_v3_tuned2" / "waves" / "wave_05" / "refined_intervals.csv"
        if credit_bounds.exists():
            base["wave1_bounds_csv"] = str(credit_bounds)
            base["wave1_bounds_kind"] = "p05p95"
        base["improb_threshold_base"] = min(2.9, float(base.get("improb_threshold_base", 2.6)) + 0.15)
        base["wave_n"] = min(280, int(base.get("wave_n", 180)) + 20)
        return Experiment(name=f"exp{next_idx:02d}_adapt_credit", args=base)

    return None


def main() -> None:
    p = argparse.ArgumentParser(description="Run overnight quality campaign across multiple preflight experiments.")
    p.add_argument("--root", type=str, default=".")
    p.add_argument("--outdir", type=str, default="")
    p.add_argument("--targets-path", type=str, default="scripts/targets_report.json")
    p.add_argument("--max-runs", type=int, default=8)
    args = p.parse_args()

    root = Path(args.root).resolve()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    campaign_dir = (root / (args.outdir or f"output_quality_campaign_{stamp}")).resolve()
    campaign_dir.mkdir(parents=True, exist_ok=True)

    experiments = _campaign_experiments(root, campaign_dir)
    results: List[Dict[str, Any]] = []
    seen_signatures: set[str] = set()
    queue: List[Experiment] = list(experiments)

    idx = 0
    while queue and idx < int(args.max_runs):
        exp = queue.pop(0)
        idx += 1
        exp_out = campaign_dir / exp.name
        cmd = [
            _python(),
            "scripts/run_research_core.py",
            "--mode",
            "preflight",
            "--profile",
            "nightly_v3",
            "--preflight-launch-full",
            "false",
            "--outdir",
            str(exp_out),
            "--targets-path",
            str(args.targets_path),
            "--nroy-band",
            "25,65",
            "--imax-median-max",
            "3.0",
            "--imax-p95-max",
            "4.5",
            "--emu-r2-median-min",
            "0.60",
            "--emu-r2-share-threshold",
            "0.30",
            "--emu-r2-share-min",
            "0.65",
        ]
        for k, v in exp.args.items():
            cmd.extend([f"--{k.replace('_', '-')}", str(v)])

        print(f"\n[{idx}/{int(args.max_runs)}] {exp.name}")
        print("[RUN]", " ".join(cmd))
        t0 = time.time()
        proc = subprocess.run(cmd, cwd=str(root))
        dt = round(time.time() - t0, 1)

        qg = exp_out / "04_master" / "research_core_tables" / "quality_gates.csv"
        summary = _load_blocking_summary(qg)
        key = _extract_key_metrics(qg)
        rec: Dict[str, Any] = {
            "experiment": exp.name,
            "returncode": int(proc.returncode),
            "runtime_sec": dt,
            "outdir": str(exp_out),
            **summary,
            **key,
        }
        results.append(rec)
        pd.DataFrame(results).to_csv(campaign_dir / "campaign_results.csv", index=False)
        (campaign_dir / "campaign_results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

        sig = json.dumps(exp.args, sort_keys=True, ensure_ascii=False)
        seen_signatures.add(sig)
        follow = _adaptive_followup(exp, rec, next_idx=idx + 1, root=root)
        if follow is not None and len(results) < int(args.max_runs):
            fsig = json.dumps(follow.args, sort_keys=True, ensure_ascii=False)
            if fsig not in seen_signatures:
                queue.append(follow)

    df = pd.DataFrame(results)
    if not df.empty:
        df["score"] = (df["n_blocking"] - df["n_failed"]).fillna(0)
        df = df.sort_values(["blocking_pass", "score"], ascending=[False, False])
        df.to_csv(campaign_dir / "campaign_ranked.csv", index=False)

    print(f"\n[DONE] Campaign results saved to {campaign_dir}")


if __name__ == "__main__":
    main()
