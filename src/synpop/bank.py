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
    ):
        self.loan_rate = loan_rate
        self.deposit_rate = deposit_rate
        self.credit_multiplier = credit_multiplier
        self.state = BankState(equity=init_equity, reserves=init_equity)
        self.failed = False

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
        self.failed = self.state.equity < 0

    def absorb_loss(self, amount: float) -> None:
        """Reduce bank equity by realized credit losses."""
        if amount <= 0:
            return
        self.state.equity -= amount
        self.failed = self.state.equity < 0

    def available_credit(self) -> float:
        if self.failed:
            return 0.0
        cap = self.credit_multiplier * max(self.state.equity, 0.0)
        used = self.state.loans_firms + self.state.loans_hh
        return max(0.0, cap - used)

    def grant_loan(self, requested: float) -> float:
        if self.failed or requested <= 0:
            return 0.0
        cap = self.credit_multiplier * max(self.state.equity, 0.0)
        used = self.state.loans_firms + self.state.loans_hh
        available = max(0.0, cap - used)
        base = min(requested, available)
        prudential_factor = clamp(1.0 - (used / cap) if cap > 0 else 1.0, 0.0, 1.0)
        return base * prudential_factor

    def update_balance_sheet(self, hh_list, firm_list) -> None:
        self.state.deposits_hh = sum(h.deposit for h in hh_list)
        self.state.deposits_firms = sum(f.cash for f in firm_list)
        self.state.loans_hh = sum(h.debt for h in hh_list)
        self.state.loans_firms = sum(f.debt for f in firm_list)
        liabilities = self.state.deposits_hh + self.state.deposits_firms + self.state.equity
        assets = self.state.loans_firms + self.state.loans_hh
        self.state.reserves = liabilities - assets
