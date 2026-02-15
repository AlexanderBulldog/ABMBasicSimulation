from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
for p in (str(ROOT), str(SCRIPTS)):
    if p not in sys.path:
        sys.path.insert(0, p)

from run_evidence_campaign import _candidate_score, _consistency_eval, _core_pass_eval  # noqa: E402


def test_core_pass_eval_blocking_only() -> None:
    gates = pd.DataFrame(
        [
            {"check": "a", "pass": True, "blocking": True},
            {"check": "b", "pass": True, "blocking": True},
            {"check": "legacy", "pass": False, "blocking": False},
        ]
    )
    assert _core_pass_eval(gates) is True


def test_core_pass_eval_fails_when_any_blocking_fail() -> None:
    gates = pd.DataFrame(
        [
            {"check": "a", "pass": True, "blocking": True},
            {"check": "b", "pass": False, "blocking": True},
        ]
    )
    assert _core_pass_eval(gates) is False


def test_consistency_eval_passes_within_thresholds() -> None:
    runs = [
        {"nroy_pct": 0.58, "r2_median": 0.64},
        {"nroy_pct": 0.62, "r2_median": 0.60},
    ]
    out = _consistency_eval(runs, nroy_tol_pp=8.0, r2_tol=0.08)
    assert out["ok"] is True


def test_consistency_eval_fails_when_nroy_spread_too_large() -> None:
    runs = [
        {"nroy_pct": 0.52, "r2_median": 0.64},
        {"nroy_pct": 0.63, "r2_median": 0.62},
    ]
    out = _consistency_eval(runs, nroy_tol_pp=8.0, r2_tol=0.08)
    assert out["ok"] is False


def test_campaign_acceptance_2_of_3_logic() -> None:
    stage_b = [
        {"core_pass": True, "nroy_pct": 0.58, "r2_median": 0.64},
        {"core_pass": True, "nroy_pct": 0.60, "r2_median": 0.61},
        {"core_pass": False, "nroy_pct": 0.72, "r2_median": 0.52},
    ]
    pass_runs = [r for r in stage_b if bool(r["core_pass"])]
    consistency = _consistency_eval(pass_runs, nroy_tol_pp=8.0, r2_tol=0.08)
    success = len(pass_runs) >= 2 and bool(consistency["ok"])
    assert success is True


def test_candidate_selection_prefers_pass_attempt_even_if_not_last() -> None:
    attempt1_score = _candidate_score(0.58, 0.62, 2.45)
    attempt2_score = _candidate_score(0.70, 0.65, 2.53)
    assert attempt1_score > 0
    assert attempt2_score > 0
    # Simulates ranking tuple used in run_evidence_campaign: (pass_flag, score)
    rank1 = (1, attempt1_score)
    rank2 = (0, attempt2_score)
    assert rank1 > rank2
