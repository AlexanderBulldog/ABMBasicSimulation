from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
for p in (str(ROOT), str(SCRIPTS)):
    if p not in sys.path:
        sys.path.insert(0, p)

from make_research_master_report import _blocking_gates, _legacy_gate_eval  # noqa: E402
from run_research_core import _evaluate_go_no_go, _preflight_improb_thresholds, _sigma_tuning_decision  # noqa: E402


def test_sigma_tuning_decision_tightens_when_nroy_above_band() -> None:
    decision = _sigma_tuning_decision(nroy=90.0, nroy_min=25.0, nroy_max=65.0, step=0.12)
    assert decision is not None
    factor, reason = decision
    assert factor < 1.0
    assert reason == "nroy_above_max_tighten_sigma_model"


def test_preflight_thresholds_has_max_one_retry() -> None:
    assert _preflight_improb_thresholds(base=2.6, bump=0.15, max_value=2.9) == [2.6, 2.75]


def test_go_no_go_uses_blocking_only(tmp_path: Path) -> None:
    gates = pd.DataFrame(
        [
            {"check": "blocking pass", "pass": True, "value": "ok", "blocking": True, "reason_code": "ok"},
            {"check": "legacy fail", "pass": False, "value": "ref", "blocking": False, "reason_code": "legacy"},
        ]
    )
    p = tmp_path / "quality_gates.csv"
    gates.to_csv(p, index=False)
    out = _evaluate_go_no_go(p)
    assert out["pass"] is True
    assert out["n_blocking"] == 1


def test_gate_tables_have_required_columns() -> None:
    hm = pd.DataFrame({"nroy": [True, False, True], "I_max": [2.2, 3.4, 2.8], "metrics_used": [4, 4, 4]})
    lhs = pd.DataFrame({"bad_run": [False, False, True], "PriceDispersion_mean": [0.1, 0.08, 0.09], "InventoryGap_mean": [0.1, -0.2, 0.4], "CreditRejections_mean": [10.0, 20.0, 30.0]})
    emu = pd.DataFrame({"skipped": [False, False, True], "gpr_r2_cv_mean": [0.7, 0.4, -0.1]})
    rep = pd.DataFrame({"Employment_mean": [60.0, 62.0], "Output_mean": [80.0, 82.0], "Consumption_mean": [78.0, 81.0], "BankFailed_share": [0.0, 0.0], "BalanceOK_share": [1.0, 1.0], "CreditRejections_mean": [10.0, 12.0]})
    sa = pd.DataFrame({"component": ["EV", "OU", "MD", "CU"] * 4, "rank_at_max_reduction": [1, 2, 3, 4] * 4, "rank_by_mean": [1, 2, 3, 4] * 4})
    cfg = {
        "nroy_wave2_min_pct": 25.0,
        "nroy_wave2_max_pct": 65.0,
        "legacy_nroy_wave2_min_pct": 25.0,
        "legacy_nroy_wave2_max_pct": 60.0,
        "imax_wave2_median_max": 3.0,
        "imax_wave2_p95_max": 4.5,
        "rep_employment_min": 45.0,
        "rep_output_min": 50.0,
        "rep_consumption_min": 50.0,
        "bad_run_pct_w2_max": 5.0,
        "credit_rejections_p90_max": 30.0,
        "emu_r2_cv_median_min": 0.60,
        "emu_r2_share_threshold": 0.30,
        "emu_r2_share_min": 0.65,
    }
    legacy = _legacy_gate_eval(hm, cfg)
    blocking = _blocking_gates(hm, lhs, emu, rep, sa, cfg, confirm_stats=None)
    for col in ["contour", "blocking", "gate_group", "reason_code"]:
        assert col in legacy.columns
        assert col in blocking.columns


def test_go_no_go_json_shape(tmp_path: Path) -> None:
    gates = pd.DataFrame([
        {"check": "c1", "pass": True, "value": "v", "blocking": True, "reason_code": "ok"},
        {"check": "c2", "pass": False, "value": "v", "blocking": True, "reason_code": "x"},
    ])
    p = tmp_path / "q.csv"
    gates.to_csv(p, index=False)
    out = _evaluate_go_no_go(p)
    assert out["pass"] is False
    payload = {"attempts": [{"checks": out["checks"]}]}
    text = json.dumps(payload)
    assert "checks" in text
