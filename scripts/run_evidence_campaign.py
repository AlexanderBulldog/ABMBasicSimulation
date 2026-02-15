from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import pandas as pd


def _python_cmd() -> str:
    return sys.executable


def _run(cmd: List[str], cwd: Path) -> None:
    print(f"[RUN] {' '.join(cmd)}")
    proc = subprocess.run(cmd, cwd=str(cwd))
    if proc.returncode != 0:
        raise SystemExit(f"Command failed ({proc.returncode}): {' '.join(cmd)}")


def _to_bool(v) -> bool:
    if isinstance(v, bool):
        return v
    if v is None:
        return False
    return str(v).strip().lower() in {"1", "true", "yes", "y"}


def _read_quality_gates(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(f"Missing quality gates: {path}")
    return pd.read_csv(path)


def _core_pass_eval(gates: pd.DataFrame) -> bool:
    if gates.empty:
        return False
    if "blocking" not in gates.columns or "pass" not in gates.columns:
        return False
    b = gates[gates["blocking"].map(_to_bool)]
    if b.empty:
        return False
    return bool(b["pass"].map(_to_bool).all())


def _metric_from_gates(gates: pd.DataFrame, check_prefix: str) -> float:
    if "check" not in gates.columns or "value" not in gates.columns:
        return float("nan")
    row = gates[gates["check"].astype(str).str.startswith(check_prefix)]
    if row.empty:
        return float("nan")
    raw = str(row.iloc[0]["value"])
    try:
        if raw.endswith("%"):
            return float(raw.rstrip("%")) / 100.0
        if "min=" in raw:
            return float(raw.split("min=")[-1].split(";")[0].strip())
        if "max=" in raw:
            return float(raw.split("max=")[-1].split(";")[0].strip())
        if "=" in raw:
            return float(raw.split("=")[-1].split(";")[0].strip())
        return float(raw)
    except Exception:
        return float("nan")


def _candidate_score(nroy_frac: float, r2_median: float, imax_median: float) -> float:
    # score = +2*(R2_median-0.60) + 1*(0.65-|NROY-0.55|) + 0.5*(3.0-Imax_median)
    return float(2.0 * (r2_median - 0.60) + (0.65 - abs(nroy_frac - 0.55)) + 0.5 * (3.0 - imax_median))


def _consistency_eval(pass_runs: List[Dict[str, float]], nroy_tol_pp: float = 8.0, r2_tol: float = 0.08) -> Dict[str, object]:
    if len(pass_runs) < 2:
        return {"ok": True, "nroy_max_delta_pp": 0.0, "r2_max_delta": 0.0}
    nroy = [float(x["nroy_pct"]) for x in pass_runs]
    r2 = [float(x["r2_median"]) for x in pass_runs]
    nroy_delta_pp = (max(nroy) - min(nroy)) * 100.0
    r2_delta = max(r2) - min(r2)
    return {
        "ok": bool(nroy_delta_pp <= nroy_tol_pp and r2_delta <= r2_tol),
        "nroy_max_delta_pp": float(nroy_delta_pp),
        "r2_max_delta": float(r2_delta),
        "nroy_tol_pp": float(nroy_tol_pp),
        "r2_tol": float(r2_tol),
    }


@dataclass
class Candidate:
    name: str
    improb_base: float
    sigma_step: float
    ev_q: float


def _run_core(
    root: Path,
    outdir: Path,
    campaign_id: str,
    mode: str,
    improb_base: float,
    sigma_step: float,
    ev_q: float,
    seed_offset: int,
    wave_n: int,
    confirm_seed_offset_base: int,
) -> Dict[str, object]:
    cmd = [
        _python_cmd(),
        "scripts/run_research_core.py",
        "--mode",
        mode,
        "--profile",
        "nightly_v3",
        "--outdir",
        str(outdir),
        "--campaign-id",
        campaign_id,
        "--seed-offset",
        str(seed_offset),
        "--wave-min",
        "2",
        "--wave-max",
        "3",
        "--adaptive-waves",
        "true",
        "--wave-n",
        str(int(wave_n)),
        "--improb-threshold-base",
        str(float(improb_base)),
        "--sigma-tuning-step",
        str(sigma_step),
        "--sigma-tuning-max-iters",
        "3",
        "--ev-default-quantile",
        str(ev_q),
        "--min-r2-for-history-matching",
        "0.30",
        "--credit-rejection-rate-mean-max",
        "0.80",
        "--confirm-seed-offset-base",
        str(int(confirm_seed_offset_base)),
        "--final-wave-alias",
        "final",
    ]
    if mode == "preflight":
        cmd.extend(
            [
                "--preflight-launch-full",
                "false",
                "--preflight-retry-policy",
                "nroy_safe",
                "--preflight-threshold-up-step",
                "0.15",
                "--preflight-threshold-down-step",
                "0.10",
                "--preflight-threshold-min",
                "2.45",
                "--preflight-threshold-max",
                "2.90",
            ]
        )
    _run(cmd, cwd=root)

    gates = _read_quality_gates(outdir / "04_master" / "research_core_tables" / "quality_gates.csv")
    core_pass = _core_pass_eval(gates)
    nroy_frac = _metric_from_gates(gates, "NROY wave2 in [25,65]%")
    r2_median = _metric_from_gates(gates, "Emulator CV-R2 median")
    imax_median = _metric_from_gates(gates, "I_max median wave2")
    selected_attempt = 1
    attempts_digest: list[dict[str, object]] = []
    status_path = outdir / "go_no_go_status.json"
    if mode == "preflight" and status_path.exists():
        status = json.loads(status_path.read_text(encoding="utf-8"))
        attempts = list(status.get("attempts", []))
        best_tuple = None
        for i, a in enumerate(attempts, start=1):
            checks = pd.DataFrame(list(a.get("checks", [])))
            if checks.empty:
                continue
            pass_a = _core_pass_eval(
                checks.assign(
                    blocking=True,
                    **{"pass": checks["pass"].map(_to_bool)},
                )
            )
            nroy_a = _metric_from_gates(checks.assign(check=checks["check"], value=checks["value"]), "NROY wave2 in [25,65]%")
            r2_a = _metric_from_gates(checks.assign(check=checks["check"], value=checks["value"]), "Emulator CV-R2 median")
            imax_a = _metric_from_gates(checks.assign(check=checks["check"], value=checks["value"]), "I_max median wave2")
            score_a = float(_candidate_score(nroy_a, r2_a, imax_a))
            attempts_digest.append(
                {
                    "attempt": i,
                    "improb_threshold": float(a.get("improb_threshold", float("nan"))),
                    "pass": bool(pass_a),
                    "nroy_pct": float(nroy_a),
                    "r2_median": float(r2_a),
                    "imax_median": float(imax_a),
                    "score": score_a,
                    "threshold_decision_reason": str(a.get("threshold_decision_reason", "")),
                }
            )
            rank_key = (1 if pass_a else 0, score_a)
            if best_tuple is None or rank_key > best_tuple[0]:
                best_tuple = (rank_key, i, pass_a, nroy_a, r2_a, imax_a, score_a)
        if best_tuple is not None:
            selected_attempt = int(best_tuple[1])
            core_pass = bool(best_tuple[2])
            nroy_frac = float(best_tuple[3])
            r2_median = float(best_tuple[4])
            imax_median = float(best_tuple[5])

    result: Dict[str, object] = {
        "outdir": str(outdir),
        "core_pass": bool(core_pass),
        "nroy_pct": float(nroy_frac),
        "r2_median": float(r2_median),
        "imax_median": float(imax_median),
        "score": float(_candidate_score(nroy_frac, r2_median, imax_median)),
        "selected_attempt": int(selected_attempt),
        "selected_metrics": {
            "nroy_pct": float(nroy_frac),
            "r2_median": float(r2_median),
            "imax_median": float(imax_median),
        },
        "attempts_digest": attempts_digest,
    }
    confirm_path = outdir / "04_master" / "confirmatory_summary.json"
    if confirm_path.exists():
        payload = json.loads(confirm_path.read_text(encoding="utf-8"))
        result["confirm_nroy_delta_pp"] = float(payload.get("nroy_delta_pp", float("nan")))
    return result


def _run_quick_tiebreak_confirm(
    root: Path,
    outdir: Path,
    campaign_id: str,
    sigma_step: float,
    ev_q: float,
    improb_base: float,
    confirm_seed_offset_base: int,
) -> float:
    quick_dir = outdir / "quick_confirm"
    cmd = [
        _python_cmd(),
        "scripts/run_research_core.py",
        "--mode",
        "full",
        "--profile",
        "nightly_v3",
        "--outdir",
        str(quick_dir),
        "--campaign-id",
        campaign_id,
        "--wave-min",
        "2",
        "--wave-max",
        "3",
        "--adaptive-waves",
        "true",
        "--improb-threshold-base",
        str(float(improb_base)),
        "--sigma-tuning-step",
        str(sigma_step),
        "--sigma-tuning-max-iters",
        "3",
        "--ev-default-quantile",
        str(ev_q),
        "--wave-n",
        "120",
        "--wave-steps",
        "120",
        "--report-n",
        "2",
        "--report-steps",
        "120",
        "--min-r2-for-history-matching",
        "0.30",
        "--credit-rejection-rate-mean-max",
        "0.80",
        "--final-wave-alias",
        "final",
        "--confirm-seed-offset-base",
        str(int(confirm_seed_offset_base)),
    ]
    _run(cmd, cwd=root)
    confirm = quick_dir / "04_master" / "confirmatory_summary.json"
    if not confirm.exists():
        return float("inf")
    payload = json.loads(confirm.read_text(encoding="utf-8"))
    return abs(float(payload.get("nroy_delta_pp", float("inf"))))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run 12h evidence campaign (Stage A/B/C).")
    p.add_argument("--root", type=str, default=".")
    p.add_argument("--outdir", type=str, default="output_evidence_campaign")
    p.add_argument("--campaign-id", type=str, default="")
    p.add_argument("--mini-confirm-max-delta-pp", type=float, default=10.0)
    p.add_argument("--confirm-seed-offset-base", type=int, default=31)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.root).resolve()
    outdir = (root / args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    campaign_id = args.campaign_id.strip() or f"evidence_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"

    frozen = {
        "campaign_id": campaign_id,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "goal": "Core PASS evidence campaign with 2/3 replication",
        "core_pass_blocking_only": True,
        "acceptance": {"min_pass_runs": 2, "n_runs": 3},
        "consistency": {"nroy_tol_pp": 8.0, "r2_tol": 0.08},
        "fixed_wave_bounds": {"wave_min": 2, "wave_max": 3},
        "preflight_retry_policy": "nroy_safe",
        "stage_a_wave_n": 160,
        "mini_confirm_max_delta_pp": float(args.mini_confirm_max_delta_pp),
        "confirm_seed_offset_base": int(args.confirm_seed_offset_base),
    }
    (outdir / "frozen_protocol.json").write_text(json.dumps(frozen, ensure_ascii=False, indent=2), encoding="utf-8")

    candidates = [
        Candidate(name="A1", improb_base=2.60, sigma_step=0.08, ev_q=0.80),
        Candidate(name="A2", improb_base=2.55, sigma_step=0.08, ev_q=0.80),
        Candidate(name="A3", improb_base=2.60, sigma_step=0.10, ev_q=0.75),
        Candidate(name="A4", improb_base=2.55, sigma_step=0.10, ev_q=0.75),
    ]

    stage_a: List[Dict[str, object]] = []
    for c in candidates:
        c_dir = outdir / "stage_a" / c.name
        c_dir.mkdir(parents=True, exist_ok=True)
        res = _run_core(
            root=root,
            outdir=c_dir,
            campaign_id=campaign_id,
            mode="preflight",
            improb_base=c.improb_base,
            sigma_step=c.sigma_step,
            ev_q=c.ev_q,
            seed_offset=0,
            wave_n=160,
            confirm_seed_offset_base=int(args.confirm_seed_offset_base),
        )
        res.update({"candidate": c.name, "improb_base": c.improb_base, "sigma_step": c.sigma_step, "ev_default_quantile": c.ev_q})
        stage_a.append(res)

    stage_a_pass = [x for x in stage_a if bool(x["core_pass"])]
    if not stage_a_pass:
        summary = {
            "campaign_id": campaign_id,
            "status": "FAIL",
            "reason": "stage_a_no_core_pass_candidates",
            "stage_a": stage_a,
            "stage_b": [],
        }
        (outdir / "campaign_evidence_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print("[DONE] Campaign finished with FAIL: no Stage A PASS candidates.")
        return

    stage_a_eligible: List[Dict[str, object]] = []
    for row in stage_a_pass:
        delta = _run_quick_tiebreak_confirm(
            root=root,
            outdir=outdir / "stage_a" / str(row["candidate"]),
            campaign_id=campaign_id,
            sigma_step=float(row["sigma_step"]),
            ev_q=float(row["ev_default_quantile"]),
            improb_base=float(row["improb_base"]),
            confirm_seed_offset_base=int(args.confirm_seed_offset_base),
        )
        row["quick_confirm_nroy_delta_abs_pp"] = float(delta)
        if float(delta) <= float(args.mini_confirm_max_delta_pp):
            stage_a_eligible.append(row)

    if not stage_a_eligible:
        summary = {
            "campaign_id": campaign_id,
            "status": "FAIL",
            "reason": "stage_a_no_mini_confirm_stable_candidates",
            "stage_a": stage_a,
            "stage_b": [],
            "mini_confirm_max_delta_pp": float(args.mini_confirm_max_delta_pp),
        }
        (outdir / "campaign_evidence_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print("[DONE] Campaign finished with FAIL: no mini-confirm stable Stage A candidates.")
        return

    best_score = max(float(x["score"]) for x in stage_a_eligible)
    leaders = [x for x in stage_a_eligible if abs(float(x["score"]) - best_score) < 1e-12]
    winner = sorted(leaders, key=lambda x: float(x.get("quick_confirm_nroy_delta_abs_pp", float("inf"))))[0]

    stage_b: List[Dict[str, object]] = []
    offsets = [0, 200, 400]
    for idx, off in enumerate(offsets, start=1):
        rid = f"R{idx}"
        r_dir = outdir / "stage_b" / rid
        r_dir.mkdir(parents=True, exist_ok=True)
        res = _run_core(
            root=root,
            outdir=r_dir,
            campaign_id=campaign_id,
            mode="full",
            improb_base=float(winner["improb_base"]),
            sigma_step=float(winner["sigma_step"]),
            ev_q=float(winner["ev_default_quantile"]),
            seed_offset=int(off),
            wave_n=350,
            confirm_seed_offset_base=int(args.confirm_seed_offset_base),
        )
        res.update({"run_id": rid, "seed_offset": off})
        stage_b.append(res)

    pass_runs = [r for r in stage_b if bool(r["core_pass"])]
    consistency = _consistency_eval(pass_runs, nroy_tol_pp=8.0, r2_tol=0.08)
    success = len(pass_runs) >= 2 and bool(consistency["ok"])
    summary = {
        "campaign_id": campaign_id,
        "status": "SUCCESS" if success else "FAIL",
        "acceptance_rule": "core_pass_at_least_2_of_3_and_consistency_ok",
        "winner": winner,
        "stage_a": stage_a,
        "stage_b": stage_b,
        "pass_runs_count": len(pass_runs),
        "consistency": consistency,
    }
    (outdir / "campaign_evidence_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame(stage_a).to_csv(outdir / "stage_a_summary.csv", index=False)
    pd.DataFrame(stage_b).to_csv(outdir / "stage_b_summary.csv", index=False)
    print(f"[DONE] Campaign finished: {summary['status']}. Summary: {outdir / 'campaign_evidence_summary.json'}")


if __name__ == "__main__":
    main()
