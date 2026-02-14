from __future__ import annotations

from dataclasses import dataclass
from typing import List

from .utils import clamp


@dataclass
class BankState:
    loans_firms: float = 0.0
    loans_hh: float = 0.0
    deposits_hh: float = 0.0
    deposits_firms: float = 0.0
    equity: float = 5.0
    reserves: float = 5.0


class Bank:
    """Minimal stock-flow consistent bank balance sheet."""

    def __init__(
        self,
        loan_rate: float,
        deposit_rate: float,
        credit_multiplier: float = 6.0,
        init_equity: float = 5.0,
        recap_ratio: float = 0.06,
        allow_bailout: bool = True,
        bailout_min_equity: float = 1.0,
    ):
        self.loan_rate = loan_rate
        self.deposit_rate = deposit_rate
        self.credit_multiplier = credit_multiplier
        self.recap_ratio = clamp(recap_ratio, 0.0, 0.5)
        self.allow_bailout = bool(allow_bailout)
        self.bailout_min_equity = max(0.0, float(bailout_min_equity))
        self.state = BankState(equity=init_equity, reserves=init_equity)
        self.failed = False
        self.last_resolution_amount = 0.0
        self.last_resolution_haircut = 0.0
        self.last_resolved = False
        self.last_bailout_amount = 0.0
        self.last_bailed_out = False
        self.last_credit_requests = 0
        self.last_credit_rejections = 0

    def begin_step(self) -> None:
        self.last_credit_requests = 0
        self.last_credit_rejections = 0

    def accrue_interest(self, hh_list, firm_list) -> None:
        """Apply interest to deposits and loans; adjust equity by net margin."""
        total_loan_int = 0.0

        for h in hh_list:
            if h.debt > 0:
                interest = h.debt * self.loan_rate
                h.debt += interest
                service = min(h.deposit, interest)
                h.deposit -= service
                h.debt -= service
                total_loan_int += service

        for f in firm_list:
            if f.debt > 0:
                interest = f.debt * self.loan_rate
                f.debt += interest
                service = min(f.cash, interest)
                f.cash -= service
                f.debt -= service
                total_loan_int += service

        total_deposits = sum(max(0.0, h.deposit) for h in hh_list) + sum(max(0.0, f.cash) for f in firm_list)
        deposit_rate = self.deposit_rate if self.state.equity > 0 else 0.0
        if total_deposits > 0 and deposit_rate > 0:
            desired = deposit_rate * total_deposits
            # Keep deposit interest within realized loan-interest cashflow; deposit_rate acts as a cap.
            if desired > 0 and total_loan_int < desired:
                deposit_rate *= total_loan_int / desired

        total_deposit_int = 0.0
        if deposit_rate > 0:
            for h in hh_list:
                if h.deposit > 0:
                    interest = h.deposit * deposit_rate
                    h.deposit += interest
                    total_deposit_int += interest
            for f in firm_list:
                if f.cash > 0:
                    interest = f.cash * deposit_rate
                    f.cash += interest
                    total_deposit_int += interest

        net = total_loan_int - total_deposit_int
        self.state.equity += net

    def absorb_loss(self, amount: float) -> None:
        """Reduce bank equity by realized credit losses."""
        if amount <= 0:
            return
        self.state.equity -= amount

    def available_credit(self) -> float:
        if self.failed:
            return 0.0
        cap = self.credit_multiplier * max(self.state.equity, 0.0)
        used = self.state.loans_firms + self.state.loans_hh
        return max(0.0, cap - used)

    def grant_loan(
        self,
        requested: float,
        borrower_type: str = "generic",
        expected_income: float | None = None,
        expected_revenue: float | None = None,
        cash_buffer: float | None = None,
        current_debt: float | None = None,
        max_dsr: float | None = None,
    ) -> float:
        if self.failed or requested <= 0:
            return 0.0
        self.last_credit_requests += 1
        cap = self.credit_multiplier * max(self.state.equity, 0.0)
        used = self.state.loans_firms + self.state.loans_hh
        available = max(0.0, cap - used)
        base = min(requested, available)
        # Smooth prudential taper: keep normal lending in mid-utilization regimes,
        # then gradually ration near the balance-sheet cap.
        if cap > 0:
            util = clamp(used / cap, 0.0, 1.5)
            if util <= 0.70:
                prudential_factor = 1.0
            else:
                # At cap utilization, keep a small positive flow instead of hard shutdown.
                prudential_factor = clamp(1.0 - (util - 0.70) / 0.60, 0.35, 1.0)
        else:
            prudential_factor = 1.0
        loan = base * prudential_factor

        debt_now = max(0.0, float(current_debt) if current_debt is not None else 0.0)
        dsr_cap = max(0.0, float(max_dsr) if max_dsr is not None else 0.0)
        if dsr_cap > 0:
            if borrower_type == "household":
                income = max(1e-9, float(expected_income) if expected_income is not None else 0.0)
                max_debt_stock = dsr_cap * income / max(self.loan_rate, 1e-6)
                # Allow limited rollover of existing debt to avoid unnecessary hard denials.
                borrower_limit = max(0.0, max_debt_stock - 0.90 * debt_now)
                loan = min(loan, borrower_limit)
            elif borrower_type == "firm":
                revenue = max(0.0, float(expected_revenue) if expected_revenue is not None else 0.0)
                buffer_cash = max(0.0, float(cash_buffer) if cash_buffer is not None else 0.0)
                # Allow part of liquid buffer to support debt service for short-term funding gaps.
                debt_service_capacity = revenue + 0.50 * buffer_cash
                max_debt_stock = dsr_cap * debt_service_capacity / max(self.loan_rate, 1e-6)
                # Firms face lumpy cash needs; permit conservative refinancing of outstanding debt.
                borrower_limit = max(0.0, max_debt_stock - 0.85 * debt_now)
                loan = min(loan, borrower_limit)

        # Rejection is counted only for full denials; partial grants remain approved-but-rationed.
        if loan <= 1e-12:
            self.last_credit_rejections += 1
        return max(0.0, loan)

    def update_balance_sheet(self, hh_list, firm_list) -> None:
        self.last_resolution_amount = 0.0
        self.last_resolution_haircut = 0.0
        self.last_resolved = False
        self.last_bailout_amount = 0.0
        self.last_bailed_out = False

        self.state.deposits_hh = sum(h.deposit for h in hh_list)
        self.state.deposits_firms = sum(f.cash for f in firm_list)
        self.state.loans_hh = sum(h.debt for h in hh_list)
        self.state.loans_firms = sum(f.debt for f in firm_list)

        if self.state.equity < 0:
            self._resolve_insolvency(hh_list, firm_list)
            self.state.deposits_hh = sum(h.deposit for h in hh_list)
            self.state.deposits_firms = sum(f.cash for f in firm_list)
            self.state.loans_hh = sum(h.debt for h in hh_list)
            self.state.loans_firms = sum(f.debt for f in firm_list)

        liabilities = self.state.deposits_hh + self.state.deposits_firms + self.state.equity
        assets = self.state.loans_firms + self.state.loans_hh
        self.state.reserves = liabilities - assets
        self.failed = self.state.equity < 0

    def _resolve_insolvency(self, hh_list, firm_list) -> None:
        """Bail-in deposits to restore solvency, targeting a small positive equity buffer."""
        total_deposits = sum(max(0.0, h.deposit) for h in hh_list) + sum(max(0.0, f.cash) for f in firm_list)
        if total_deposits <= 0:
            if self.allow_bailout and self.bailout_min_equity > 0:
                self.last_bailout_amount = float(-self.state.equity + self.bailout_min_equity)
                self.last_bailed_out = True
                self.state.equity += self.last_bailout_amount
                return
            self.failed = True
            return

        equity = self.state.equity
        r = self.recap_ratio
        # Choose bail-in X so that: equity' = equity + X and equity' = r * deposits', deposits' = total - X.
        needed = (r * total_deposits - equity) / (1.0 + r)
        needed = max(0.0, min(needed, total_deposits))
        haircut = needed / total_deposits if total_deposits > 0 else 1.0

        if haircut > 0:
            factor = 1.0 - haircut
            for h in hh_list:
                if h.deposit > 0:
                    h.deposit *= factor
            for f in firm_list:
                if f.cash > 0:
                    f.cash *= factor
            self.state.equity += needed
            self.last_resolution_amount = float(needed)
            self.last_resolution_haircut = float(haircut)
            self.last_resolved = True

        if self.state.equity < 0 and self.allow_bailout and self.bailout_min_equity > 0:
            bailout = -self.state.equity + self.bailout_min_equity
            self.state.equity += bailout
            self.last_bailout_amount = float(bailout)
            self.last_bailed_out = True
