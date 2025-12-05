from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
from mesa import Agent

from .bank import Bank
from .utils import clamp


class Household(Agent):
    def __init__(
        self,
        model: "EconomyModel",
        alpha: float,
        initial_savings: float,
        reserve_wage: Optional[float] = None,
        skill: Optional[float] = None,
        reserve_window: int = 4,
    ) -> None:
        super().__init__(model)
        self.alpha = alpha
        self.deposit = initial_savings
        self.debt = 0.0
        self.employed = False
        self.employer_id: Optional[int] = None
        self.last_income = 0.0
        self.reserve_wage = reserve_wage or model.wage
        self.skill = skill
        self._wage_history: List[float] = []
        self.reserve_window = reserve_window
        self.defaulted = False

    def begin_step(self) -> None:
        self.last_income = 0.0
        self.defaulted = False

    def receive_wage(self, amount: float, employer_id: int) -> None:
        self.last_income += amount
        self.deposit += amount
        self.employed = True
        self.employer_id = employer_id
        self._wage_history.append(amount)
        if len(self._wage_history) > self.reserve_window:
            self._wage_history.pop(0)
        self.reserve_wage = np.mean(self._wage_history) if self._wage_history else self.reserve_wage

    def decide_consumption(self, bank: Bank, repay_fraction: float) -> float:
        available = self.deposit

        if self.debt > 0 and available > 0:
            repayment = min(available * repay_fraction, self.debt)
            self.debt -= repayment
            available -= repayment
            self.deposit -= repayment

        income_expectation = np.mean(self._wage_history) if self._wage_history else self.last_income
        desired_base = available + income_expectation
        desired = clamp(self.alpha * desired_base, 0.0, float("inf"))

        if desired > available:
            gap = desired - available
            credit_cap = self.model.hh_debt_cap_multiplier * max(self.last_income, 1e-6)
            allowed = max(0.0, credit_cap - self.debt)
            loan = min(gap, allowed, bank.available_credit())
            if loan > 0:
                self.debt += loan
                bank.state.loans_hh += loan
                available += loan
                self.deposit += loan

        consumption = min(desired, available)
        self.deposit -= consumption
        if self.deposit < 0:
            self.deposit = 0.0
        return consumption

    def maybe_default(self, max_debt_income: float) -> bool:
        if self.debt <= 0:
            return False
        income_ref = max(self.last_income, np.mean(self._wage_history) if self._wage_history else 0.0, 1e-6)
        if self.debt > max_debt_income * income_ref:
            self.defaulted = True
            self.debt = 0.0
            return True
        return False


class Firm(Agent):
    def __init__(
        self,
        model: "EconomyModel",
        productivity: float,
        initial_cash: float,
        market_share: Optional[float] = None,
        base_markup: float = 0.1,
    ) -> None:
        super().__init__(model)
        self.productivity = productivity
        self.cash = initial_cash
        self.debt = 0.0
        self.inventory = 0.0
        self.workers: List[int] = []
        self.last_demand = model.initial_demand_share()
        self.last_production = 0.0
        self.market_share = market_share
        self.price = model.base_price * (1 + base_markup)
        self.base_markup = base_markup
        self.defaulted = False

    def begin_step(self) -> None:
        self.last_production = 0.0
        self.defaulted = False

    def target_workers(self, adapt_rate: float, base_wage: float) -> int:
        desired_output = self.last_demand
        desired_workers = int(round(desired_output / max(self.productivity, 1e-6)))
        current = len(self.workers)
        diff = desired_workers - current
        adjust = 0
        if abs(diff) >= 1:
            step = max(1, int(np.ceil(abs(diff) * adapt_rate)))
            adjust = step if diff > 0 else -step
        target = max(0, current + adjust)
        affordable = int((self.cash / max(base_wage, 1e-6)))
        return min(target, affordable) if self.model.enable_credit == False else target

    def hire(self, available_workers: List[Household], needed: int) -> List[Household]:
        hires: List[Household] = []
        if needed <= 0 or not available_workers:
            return hires
        sorted_workers = sorted(
            available_workers,
            key=lambda h: (h.skill if h.skill is not None else 1.0),
            reverse=True,
        )
        count = min(needed, len(sorted_workers))
        hires = sorted_workers[:count]
        for h in hires:
            h.employed = True
            h.employer_id = self.unique_id
        current_ids = set(self.workers)
        for h in hires:
            current_ids.add(h.unique_id)
        self.workers = list(current_ids)
        return hires

    def set_price(self, base_wage: float) -> None:
        unit_cost = base_wage / max(self.productivity, 1e-6)
        inventory_signal = 0.0
        if self.inventory > self.last_demand:
            inventory_signal = -0.05
        elif self.inventory < 0.5 * self.last_demand:
            inventory_signal = 0.05
        markup = clamp(self.base_markup + inventory_signal, 0.0, 0.5)
        self.price = max(0.1, unit_cost * (1 + markup))

    def produce_and_pay(
        self,
        wages: List[float],
        bank: Bank,
    ) -> Tuple[float, float]:
        wage_bill = sum(wages)
        if wage_bill > self.cash and self.model.enable_credit:
            gap = wage_bill - self.cash
            loan = bank.grant_loan(gap)
            self.debt += loan
            bank.state.loans_firms += loan
            self.cash += loan
        paid = min(self.cash, wage_bill)
        self.cash -= paid
        effective_labor = sum(self.model.effective_skill(self._worker_by_id(wid)) for wid in self.workers)
        output = self.productivity * effective_labor
        self.inventory += output
        self.last_production = output
        return paid, output

    def _worker_by_id(self, uid: int) -> Optional[Household]:
        for h in self.model.households:
            if h.unique_id == uid:
                return h
        return None

    def maybe_default(self, max_debt_revenue: float) -> bool:
        revenue_ref = max(self.last_demand, 1e-6)
        if self.debt > max_debt_revenue * revenue_ref and self.cash < 0:
            self.defaulted = True
            self.debt = 0.0
            self.inventory = 0.0
            self.cash = 0.0
            for uid in self.workers:
                hh = self._worker_by_id(uid)
                if hh:
                    hh.employed = False
                    hh.employer_id = None
            self.workers = []
            return True
        return False
