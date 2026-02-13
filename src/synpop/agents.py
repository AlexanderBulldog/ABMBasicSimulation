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
        self.income_expectation = float(model.wage) * float(model.initial_employment_rate)

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

    def _expected_income(self) -> float:
        raw = np.mean(self._wage_history) if self._wage_history else self.last_income
        mem = clamp(getattr(self.model, "consumption_memory", 0.35), 0.0, 1.0)
        self.income_expectation = (1.0 - mem) * self.income_expectation + mem * float(raw)
        return max(0.0, self.income_expectation)

    def decide_consumption(self, bank: Bank, repay_fraction: float) -> float:
        available = self.deposit
        debt_service_paid = 0.0
        if self.debt > 0 and available > 0:
            repayment = min(available * repay_fraction, self.debt)
            self.debt -= repayment
            available -= repayment
            self.deposit -= repayment
            debt_service_paid = repayment

        expected_income = self._expected_income()
        debt_service_burden = debt_service_paid / max(expected_income, 1e-6)
        precautionary = clamp(getattr(self.model, "precautionary_saving", 0.1), 0.0, 0.9)
        burden_penalty = clamp(1.0 - precautionary * debt_service_burden, 0.2, 1.0)

        if not self.employed:
            u_penalty = clamp(getattr(self.model, "unemployed_consumption_penalty", 0.2), 0.0, 0.8)
            burden_penalty *= (1.0 - u_penalty)

        desired_base = available + expected_income
        desired_raw = clamp(self.alpha * desired_base * burden_penalty, 0.0, float("inf"))
        c_floor = clamp(getattr(self.model, "consumption_floor_prop", 0.0), 0.0, 1.0) * expected_income
        c_ceil = clamp(getattr(self.model, "consumption_ceiling_prop", 2.0), 0.2, 4.0) * desired_base
        desired = clamp(desired_raw, c_floor, c_ceil)

        if desired > available:
            gap = desired - available
            income_ref = max(expected_income, self.last_income, 1e-6)
            credit_cap = self.model.hh_debt_cap_multiplier * income_ref
            allowed = max(0.0, credit_cap - self.debt)
            loan = bank.grant_loan(
                requested=min(gap, allowed),
                borrower_type="household",
                expected_income=income_ref,
                current_debt=self.debt,
                max_dsr=getattr(self.model, "hh_max_dsr", 0.35),
            )
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
        threshold = max_debt_income * income_ref * getattr(self.model, "hh_default_trigger_multiplier", 1.0)
        if self.debt > threshold:
            self.defaulted = True
            lgd = getattr(self.model, "hh_loss_given_default", 1.0)
            loss = self.debt * float(lgd)
            self.debt = 0.0
            self.deposit = 0.0
            if loss > 0:
                self.model.bank.absorb_loss(loss)
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
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.debt = 0.0
        self.inventory = 0.0
        self.workers: List[int] = []
        self.last_demand = model.initial_demand_share()
        self.last_revenue = 0.0
        self.last_production = 0.0
        self.market_share = market_share
        self.price = model.base_price * (1 + base_markup)
        self.base_markup = base_markup
        self.defaulted = False
        self.markup = float(base_markup)
        self.expected_sales_ewma = max(1e-6, model.initial_demand_share())
        self.last_sales_units = 0.0
        self.last_forecast_error = 0.0
        self.downtime_remaining = 0
        self.reentry_pending_cash = 0.0
        self.reentered_this_step = False

    def begin_step(self) -> None:
        self.last_production = 0.0
        self.last_revenue = 0.0
        self.defaulted = False
        self.reentered_this_step = False
        if self.downtime_remaining > 0:
            self.downtime_remaining -= 1
            for uid in self.workers:
                hh = self._worker_by_id(uid)
                if hh:
                    hh.employed = False
                    hh.employer_id = None
            self.workers = []
            if self.downtime_remaining == 0:
                self._reenter()

    def is_active(self) -> bool:
        return self.downtime_remaining == 0

    def _reenter(self) -> None:
        self.cash = max(self.cash, self.reentry_pending_cash)
        self.inventory = 0.0
        self.last_revenue = 0.0
        self.last_production = 0.0
        self.last_demand = max(1e-6, self.expected_sales_ewma)
        self.reentry_pending_cash = 0.0
        self.reentered_this_step = True

    def _ml_state(self, base_wage: float) -> np.ndarray:
        bank_credit = self.model.bank.available_credit()
        return np.array(
            [
                self.inventory,
                self.last_demand,
                self.cash,
                self.debt,
                self.price,
                len(self.workers),
                bank_credit,
                base_wage,
            ],
            dtype=float,
        )

    def _ml_state_dict(self, base_wage: float) -> dict[str, float]:
        bank_credit = self.model.bank.available_credit()
        return {
            "inventory": float(self.inventory),
            "last_demand": float(self.last_demand),
            "cash": float(self.cash),
            "debt": float(self.debt),
            "price": float(self.price),
            "n_workers": float(len(self.workers)),
            "bank_available_credit": float(bank_credit),
            "wage": float(base_wage),
        }

    def target_workers(self, adapt_rate: float, base_wage: float) -> int:
        if not self.is_active():
            return 0
        planned_sales = max(1e-6, self.expected_sales_ewma)
        target_inventory = max(0.0, float(self.model.inventory_target_days) * planned_sales)
        inventory_replenishment = max(0.0, target_inventory - self.inventory)
        planned_output = planned_sales + inventory_replenishment
        desired_workers = int(round(planned_output / max(self.productivity, 1e-6)))

        current = len(self.workers)
        diff = desired_workers - current
        if abs(diff) >= 1:
            step = max(1, int(np.ceil(abs(diff) * adapt_rate)))
            if diff > 0:
                step = min(step, int(getattr(self.model, "max_hire_per_step", 8)))
            else:
                step = min(step, int(getattr(self.model, "max_fire_per_step", 8)))
            adjust = step if diff > 0 else -step
        else:
            adjust = 0
        target = max(0, current + adjust)
        affordable = int(self.cash / max(base_wage, 1e-6))
        return min(target, affordable) if not self.model.enable_credit else target

    def hire(self, available_workers: List[Household], needed: int) -> List[Household]:
        hires: List[Household] = []
        if needed <= 0 or not available_workers or not self.is_active():
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
        if self.model.use_ml_policy and self.model.firm_policy is not None:
            state_dict = self._ml_state_dict(base_wage)
            features = getattr(self.model.firm_policy, "feature_order_", None)
            if features:
                try:
                    import pandas as pd
                except ImportError:
                    state = np.array([state_dict[f] for f in features], dtype=float)
                    markup = float(self.model.firm_policy.predict(state.reshape(1, -1))[0])
                else:
                    frame = pd.DataFrame([[state_dict[f] for f in features]], columns=features)
                    markup = float(self.model.firm_policy.predict(frame)[0])
            else:
                state = self._ml_state(base_wage)
                markup = float(self.model.firm_policy.predict(state.reshape(1, -1))[0])
            self.markup = clamp(markup, 0.0, 0.6)
            self.price = max(0.1, unit_cost * (1.0 + self.markup))
            return

        expected_sales = max(1e-6, self.expected_sales_ewma)
        target_inventory = max(1e-6, float(self.model.inventory_target_days) * expected_sales)
        inventory_gap = (self.inventory - target_inventory) / target_inventory

        current_margin = (self.price / max(unit_cost, 1e-6)) - 1.0
        margin_gap = self.base_markup - current_margin
        credit_stress = self.debt / max(self.cash + expected_sales * self.price, 1e-6)

        speed = clamp(float(self.model.price_adjust_speed), 0.0, 1.0)
        k_inv = float(getattr(self.model, "markup_inventory_sensitivity", 0.15))
        k_mar = float(getattr(self.model, "markup_margin_sensitivity", 0.25))
        k_cr = float(getattr(self.model, "markup_credit_sensitivity", 0.05))
        d_markup = speed * (-k_inv * inventory_gap + k_mar * margin_gap + k_cr * credit_stress)

        self.markup = clamp(self.markup + d_markup, 0.0, 0.6)
        desired_price = max(0.1, unit_cost * (1.0 + self.markup))
        stickiness = clamp(float(self.model.price_stickiness), 0.0, 0.99)
        blended = stickiness * self.price + (1.0 - stickiness) * desired_price

        max_step = max(1e-6, float(self.model.max_price_step) * max(self.price, 0.1))
        delta = clamp(blended - self.price, -max_step, max_step)
        self.price = max(0.1, self.price + delta)
        self.markup = clamp(self.price / max(unit_cost, 1e-6) - 1.0, 0.0, 0.6)

    def produce_and_pay(
        self,
        wages: List[float],
        bank: Bank,
    ) -> Tuple[float, float, float]:
        if not self.is_active():
            return 0.0, 0.0, 0.0

        wage_bill = sum(wages)
        if wage_bill > self.cash and self.model.enable_credit:
            gap = wage_bill - self.cash
            loan = bank.grant_loan(
                requested=gap,
                borrower_type="firm",
                expected_revenue=max(self.last_revenue, self.expected_sales_ewma * self.price, 0.0),
                cash_buffer=self.cash,
                current_debt=self.debt,
                max_dsr=getattr(self.model, "firm_max_dsr", 1.2),
            )
            self.debt += loan
            bank.state.loans_firms += loan
            self.cash += loan
        paid = min(self.cash, wage_bill)
        self.cash -= paid
        overhead_rate = clamp(self.base_markup, 0.02, 0.2)
        overhead = overhead_rate * wage_bill
        self.cash -= overhead
        if self.cash < 0:
            overdraft = -self.cash
            if overdraft > 0 and self.model.enable_credit:
                loan = bank.grant_loan(
                    requested=overdraft,
                    borrower_type="firm",
                    expected_revenue=max(self.last_revenue, self.expected_sales_ewma * self.price, 0.0),
                    cash_buffer=0.0,
                    current_debt=self.debt,
                    max_dsr=getattr(self.model, "firm_max_dsr", 1.2),
                )
                self.debt += loan
                bank.state.loans_firms += loan
                self.cash += loan
            if self.cash < 0:
                self.cash = 0.0
        effective_labor = sum(self.model.effective_skill(self._worker_by_id(wid)) for wid in self.workers)
        output = self.productivity * effective_labor
        self.inventory += output
        self.last_production = output
        return paid, output, float(overhead)

    def _worker_by_id(self, uid: int) -> Optional[Household]:
        for h in self.model.households:
            if h.unique_id == uid:
                return h
        return None

    def maybe_default(self, max_debt_revenue: float) -> bool:
        revenue_ref = max(self.last_revenue, self.last_demand * self.price, 1e-6)
        threshold = max_debt_revenue * revenue_ref * getattr(self.model, "firm_default_trigger_multiplier", 1.0)
        if self.debt > threshold:
            self.defaulted = True
            lgd = getattr(self.model, "firm_loss_given_default", 1.0)
            loss = self.debt * float(lgd)
            self.debt = 0.0
            self.inventory = 0.0
            self.cash = 0.0
            self.last_revenue = 0.0
            for uid in self.workers:
                hh = self._worker_by_id(uid)
                if hh:
                    hh.employed = False
                    hh.employer_id = None
            self.workers = []
            if loss > 0:
                self.model.bank.absorb_loss(loss)

            self.downtime_remaining = max(0, int(getattr(self.model, "firm_reentry_lag", 2)))
            self.reentry_pending_cash = self.initial_cash * clamp(
                float(getattr(self.model, "firm_reentry_cash_fraction", 0.5)),
                0.0,
                1.0,
            )
            if self.downtime_remaining == 0:
                self._reenter()
            return True
        return False
