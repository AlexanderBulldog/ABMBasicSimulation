from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from synpop import EconomyModel  # noqa: E402
from synpop.bank import Bank  # noqa: E402


def test_price_block_respects_max_step() -> None:
    m = EconomyModel(seed=1, max_price_step=0.05, price_stickiness=0.0, price_adjust_speed=0.8)
    f = m.firms[0]
    f.price = 1.0
    f.inventory = 1000.0
    f.expected_sales_ewma = 10.0
    f.set_price(m.wage)
    assert abs(f.price - 1.0) <= 0.05 + 1e-9


def test_credit_underwriting_tightens_with_high_debt() -> None:
    b = Bank(loan_rate=0.03, deposit_rate=0.0, credit_multiplier=10.0, init_equity=20.0)
    loan_low_debt = b.grant_loan(
        requested=5.0,
        borrower_type="household",
        expected_income=10.0,
        current_debt=0.5,
        max_dsr=0.3,
    )
    loan_high_debt = b.grant_loan(
        requested=5.0,
        borrower_type="household",
        expected_income=10.0,
        current_debt=100.0,
        max_dsr=0.3,
    )
    assert loan_high_debt <= loan_low_debt


def test_firm_reentry_after_default_downtime() -> None:
    m = EconomyModel(seed=2, firm_reentry_lag=2, firm_reentry_cash_fraction=0.5)
    f = m.firms[0]
    f.debt = 1e6
    f.last_revenue = 0.0
    f.last_demand = 0.1
    assert f.maybe_default(max_debt_revenue=m.firm_debt_cap_multiplier)
    assert f.downtime_remaining == 2
    for _ in range(2):
        f.begin_step()
    assert f.is_active()
    assert f.cash >= 0.0


def test_backward_compat_core_columns_present() -> None:
    m = EconomyModel(seed=3)
    m.run_model(steps=3)
    df = m.results_dataframe()
    core = [
        "Employment",
        "Output",
        "Consumption",
        "AvgPrice",
        "Bank_Equity",
        "Defaults",
        "BalanceOK",
    ]
    for col in core:
        assert col in df.columns


def test_new_structural_columns_present() -> None:
    m = EconomyModel(seed=4)
    m.run_model(steps=3)
    df = m.results_dataframe()
    cols = ["AvgMarkup", "PriceDispersion", "InventoryGap", "CreditRejections", "FirmDowntimeShare", "ReentryCount"]
    for col in cols:
        assert col in df.columns
    assert np.isfinite(df[cols].to_numpy(dtype=float)).all()
