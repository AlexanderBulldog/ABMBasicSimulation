from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
from mesa import Model
from mesa.datacollection import DataCollector

from .agents import Firm, Household
from .bank import Bank
from .builder import PopulationBuilder
from .utils import clamp


class EconomyModel(Model):
    def __init__(
        self,
        n_households: int = 100,
        n_firms: int = 10,
        wage: float = 1.0,
        productivity: float = 1.0,
        alpha_mean: float = 0.85,
        alpha_std: float = 0.05,
        initial_savings_range: tuple[float, float] = (0.0, 2.0),
        firm_initial_cash: float = 5.0,
        loan_rate: float = 0.02,
        deposit_rate: float = 0.005,
        adaptation_rate: float = 0.5,
        initial_employment_rate: float = 0.8,
        repayment_fraction: float = 0.1,
        seed: Optional[int] = None,
        enable_credit: bool = True,
        household_config: Optional[Dict] = None,
        firm_config: Optional[Dict] = None,
        controls: Optional[Dict] = None,
        skill_wage_weight: float = 0.0,
        hh_debt_cap_multiplier: float = 4.0,
        firm_debt_cap_multiplier: float = 2.0,
        hh_default_trigger_multiplier: float = 1.5,
        firm_default_trigger_multiplier: float = 1.5,
        hh_loss_given_default: float = 0.6,
        firm_loss_given_default: float = 0.6,
        base_price: float = 1.0,
        price_elasticity: float = 2.0,
        quality_weight: float = 0.0,
        bank_credit_multiplier: float = 6.0,
        balance_tolerance: float = 1e-6,
        log_balance_warnings: bool = True,
        demand_smoothing: float = 0.1,
        demand_floor: Optional[float] = None,
        # Iteration 1: pricing block.
        price_adjust_speed: float = 0.2,
        price_stickiness: float = 0.65,
        max_price_step: float = 0.12,
        inventory_target_days: float = 1.5,
        markup_inventory_sensitivity: float = 0.15,
        markup_margin_sensitivity: float = 0.25,
        markup_credit_sensitivity: float = 0.05,
        # Iteration 2: labor and expectations.
        demand_expectation_memory: float = 0.35,
        max_hire_per_step: int = 8,
        max_fire_per_step: int = 8,
        # Iteration 3: underwriting and resolution.
        hh_max_dsr: float = 0.35,
        firm_max_dsr: float = 1.2,
        firm_reentry_lag: int = 2,
        firm_reentry_cash_fraction: float = 0.5,
        # Iteration 4: household smoothing.
        consumption_memory: float = 0.35,
        precautionary_saving: float = 0.1,
        consumption_floor_prop: float = 0.0,
        consumption_ceiling_prop: float = 2.0,
        unemployed_consumption_penalty: float = 0.2,
        unemployment_transfer: float = 0.0,
    ) -> None:
        super().__init__(seed=seed)
        self.n_households = n_households
        self.n_firms = n_firms
        self.wage = wage
        self.productivity = productivity
        self.loan_rate = loan_rate
        self.deposit_rate = deposit_rate
        self.adaptation_rate = adaptation_rate
        self.initial_employment_rate = initial_employment_rate
        self.repayment_fraction = repayment_fraction
        self.enable_credit = enable_credit
        self.skill_wage_weight = skill_wage_weight
        self.hh_debt_cap_multiplier = hh_debt_cap_multiplier
        self.firm_debt_cap_multiplier = firm_debt_cap_multiplier
        self.hh_default_trigger_multiplier = clamp(hh_default_trigger_multiplier, 1.0, 10.0)
        self.firm_default_trigger_multiplier = clamp(firm_default_trigger_multiplier, 1.0, 10.0)
        self.hh_loss_given_default = clamp(hh_loss_given_default, 0.0, 1.0)
        self.firm_loss_given_default = clamp(firm_loss_given_default, 0.0, 1.0)
        self.base_price = base_price
        self.price_elasticity = price_elasticity
        self.quality_weight = quality_weight
        self.balance_tolerance = balance_tolerance
        self.log_balance_warnings = log_balance_warnings
        self.demand_smoothing = clamp(demand_smoothing, 0.0, 1.0)
        self.demand_floor = demand_floor
        self.use_ml_policy = False
        self.firm_policy = None

        self.price_adjust_speed = clamp(price_adjust_speed, 0.0, 1.0)
        self.price_stickiness = clamp(price_stickiness, 0.0, 0.99)
        self.max_price_step = clamp(max_price_step, 0.001, 0.8)
        self.inventory_target_days = clamp(inventory_target_days, 0.2, 6.0)
        self.markup_inventory_sensitivity = max(0.0, float(markup_inventory_sensitivity))
        self.markup_margin_sensitivity = max(0.0, float(markup_margin_sensitivity))
        self.markup_credit_sensitivity = max(0.0, float(markup_credit_sensitivity))

        self.demand_expectation_memory = clamp(demand_expectation_memory, 0.0, 1.0)
        self.max_hire_per_step = max(1, int(max_hire_per_step))
        self.max_fire_per_step = max(1, int(max_fire_per_step))

        self.hh_max_dsr = clamp(hh_max_dsr, 0.01, 1.5)
        self.firm_max_dsr = clamp(firm_max_dsr, 0.05, 5.0)
        self.firm_reentry_lag = max(0, int(firm_reentry_lag))
        self.firm_reentry_cash_fraction = clamp(firm_reentry_cash_fraction, 0.0, 1.0)

        self.consumption_memory = clamp(consumption_memory, 0.0, 1.0)
        self.precautionary_saving = clamp(precautionary_saving, 0.0, 0.9)
        self.consumption_floor_prop = clamp(consumption_floor_prop, 0.0, 1.0)
        self.consumption_ceiling_prop = clamp(consumption_ceiling_prop, 0.2, 4.0)
        self.unemployed_consumption_penalty = clamp(unemployed_consumption_penalty, 0.0, 0.8)
        self.unemployment_transfer = max(0.0, float(unemployment_transfer))

        self.rng = np.random.default_rng(seed)
        controls = controls or {}
        builder = PopulationBuilder(self.rng)

        self.bank = Bank(
            loan_rate=loan_rate,
            deposit_rate=deposit_rate,
            credit_multiplier=bank_credit_multiplier,
            init_equity=firm_initial_cash * max(n_firms, 1),
        )

        self.households: List[Household] = []
        self.firms: List[Firm] = []

        hh_specs = builder.households(
            n_households,
            alpha_mean=alpha_mean,
            alpha_std=alpha_std,
            savings_range=initial_savings_range,
            config=household_config,
            controls=controls.get("households"),
        )
        for spec in hh_specs:
            h = Household(
                self,
                spec["alpha"],
                spec["savings"],
                reserve_wage=spec.get("reserve_wage"),
                skill=spec.get("skill"),
            )
            self.households.append(h)

        self._avg_alpha = float(np.mean([h.alpha for h in self.households])) if self.households else 0.0

        firm_specs = builder.firms(
            n_firms,
            base_productivity=productivity,
            base_cash=firm_initial_cash,
            config=firm_config,
            controls=controls.get("firms"),
        )
        for spec in firm_specs:
            f = Firm(
                self,
                spec["productivity"],
                spec["cash"],
                market_share=spec.get("market_share"),
            )
            f.last_demand = spec.get("expected_demand") or self.initial_demand_share()
            f.expected_sales_ewma = max(1e-6, f.last_demand)
            self.firms.append(f)

        self._seed_initial_employment()

        self.datacollector = DataCollector(
            model_reporters={
                "Employment": lambda m: sum(1 for h in m.households if h.employed),
                "UnemploymentRate": lambda m: 1
                - (sum(1 for h in m.households if h.employed) / max(m.n_households, 1)),
                "WageBill": lambda m: m._last_wage_bill,
                "Output": lambda m: m._last_output,
                "Production": lambda m: m._last_production,
                "Consumption": lambda m: m._last_consumption,
                "Transfers": lambda m: m._last_transfers,
                "UnemploymentTransfers": lambda m: m._last_unemployment_transfers,
                "HH_Deposit": lambda m: sum(h.deposit for h in m.households),
                "HH_Debt": lambda m: sum(h.debt for h in m.households),
                "Firm_Debt": lambda m: sum(f.debt for f in m.firms),
                "Firm_Cash": lambda m: sum(f.cash for f in m.firms),
                "Inventories": lambda m: sum(f.inventory for f in m.firms),
                "AvgPrice": lambda m: np.mean([f.price for f in m.firms]) if m.firms else 0,
                "AvgMarkup": lambda m: m._last_avg_markup,
                "PriceDispersion": lambda m: m._last_price_dispersion,
                "InventoryGap": lambda m: m._last_inventory_gap,
                "SalesForecastError": lambda m: m._last_sales_forecast_error,
                "InventoryTurnover": lambda m: m._last_inventory_turnover,
                "CreditRequests": lambda m: m._last_credit_requests,
                "CreditRejections": lambda m: m._last_credit_rejections,
                "CreditRejectionRate": lambda m: m._last_credit_rejection_rate,
                "FirmDowntimeShare": lambda m: m._last_firm_downtime_share,
                "ReentryCount": lambda m: m._last_reentry_count,
                "Bank_Equity": lambda m: m.bank.state.equity,
                "Bank_Loans": lambda m: m.bank.state.loans_firms + m.bank.state.loans_hh,
                "Bank_Deposits": lambda m: m.bank.state.deposits_firms + m.bank.state.deposits_hh,
                "BankFailed": lambda m: bool(m.bank.failed),
                "BankResolved": lambda m: bool(m.bank.last_resolved),
                "BankResolutionAmount": lambda m: float(m.bank.last_resolution_amount),
                "BankResolutionHaircut": lambda m: float(m.bank.last_resolution_haircut),
                "BankBailedOut": lambda m: bool(m.bank.last_bailed_out),
                "BankBailoutAmount": lambda m: float(m.bank.last_bailout_amount),
                "Defaults": lambda m: m._last_defaults,
                "DefaultsHH": lambda m: m._last_defaults_hh,
                "DefaultsFirm": lambda m: m._last_defaults_firm,
                "BalanceOK": lambda m: m._last_balance_ok,
            }
        )

        self._last_output = 0.0
        self._last_production = 0.0
        self._last_consumption = 0.0
        self._last_wage_bill = 0.0
        self._last_transfers = 0.0
        self._last_unemployment_transfers = 0.0
        self._last_defaults = 0
        self._last_defaults_hh = 0
        self._last_defaults_firm = 0
        self._last_balance_ok = True
        self._last_avg_markup = 0.0
        self._last_price_dispersion = 0.0
        self._last_inventory_gap = 0.0
        self._last_sales_forecast_error = 0.0
        self._last_inventory_turnover = 0.0
        self._last_credit_requests = 0.0
        self._last_credit_rejections = 0.0
        self._last_credit_rejection_rate = 0.0
        self._last_firm_downtime_share = 0.0
        self._last_reentry_count = 0.0
        self._demand_floor_value = self.demand_floor if self.demand_floor is not None else 0.0

    def initial_demand_share(self) -> float:
        employment_rate = clamp(self.initial_employment_rate, 0.0, 1.0)
        expected_wage_bill = (self.n_households * employment_rate) * self.wage
        expected_consumption = expected_wage_bill * clamp(self._avg_alpha, 0.0, 1.0)
        unit_price = max(self.base_price, 1e-6)
        total_units = expected_consumption / unit_price
        return total_units / max(self.n_firms, 1)

    def load_firm_policy(self, path: str) -> None:
        import joblib

        self.firm_policy = joblib.load(path)
        self.use_ml_policy = True

    def _seed_initial_employment(self) -> None:
        target_jobs = int(self.initial_employment_rate * self.n_households)
        eligible = [h for h in self.households if (h.reserve_wage is None or self.wage >= h.reserve_wage)]
        job_pool = self.random.sample(eligible, min(target_jobs, len(eligible)))
        per_firm = max(1, target_jobs // max(self.n_firms, 1)) if target_jobs > 0 else 0

        for f in self.firms:
            hires = job_pool[:per_firm]
            job_pool = job_pool[per_firm:]
            for h in hires:
                h.employed = True
                h.employer_id = f.unique_id
            f.workers = [h.unique_id for h in hires]

    def step(self) -> None:
        self.bank.begin_step()
        for h in self.households:
            h.begin_step()
        for f in self.firms:
            f.begin_step()

        self.bank.accrue_interest(self.households, self.firms)
        self._labor_market()
        self._production_and_wages()
        self._consumption_market()
        self._handle_defaults()
        self.bank.update_balance_sheet(self.households, self.firms)
        self._check_balance()
        self._last_credit_requests = float(self.bank.last_credit_requests)
        self._last_credit_rejections = float(self.bank.last_credit_rejections)
        self._last_credit_rejection_rate = (
            float(self.bank.last_credit_rejections) / max(float(self.bank.last_credit_requests), 1.0)
        )
        self._last_firm_downtime_share = float(
            sum(1 for f in self.firms if getattr(f, "downtime_remaining", 0) > 0) / max(len(self.firms), 1)
        )
        self._last_reentry_count = float(sum(1 for f in self.firms if getattr(f, "reentered_this_step", False)))

        self.datacollector.collect(self)

    def _labor_market(self) -> None:
        unemployed = [
            h
            for h in self.households
            if not h.employed and (h.reserve_wage is None or self.wage >= h.reserve_wage)
        ]
        self.random.shuffle(unemployed)

        for firm in self.firms:
            target = firm.target_workers(self.adaptation_rate, self.wage)
            if target < len(firm.workers):
                n_fire = len(firm.workers) - target
                to_fire_ids = self.random.sample(firm.workers, n_fire)
                for uid in to_fire_ids:
                    hh = self._household_by_id(uid)
                    if hh:
                        hh.employed = False
                        hh.employer_id = None
                firm.workers = [uid for uid in firm.workers if uid not in to_fire_ids]

            if target > len(firm.workers):
                hires_needed = target - len(firm.workers)
                hires = firm.hire(unemployed, hires_needed)
                unemployed = [h for h in unemployed if h not in hires]

    def _production_and_wages(self) -> None:
        total_production = 0.0
        total_wage_bill = 0.0
        total_overhead = 0.0
        for firm in self.firms:
            workers = [self._household_by_id(wid) for wid in firm.workers]
            workers = [w for w in workers if w is not None]
            wages = [self._wage_for_worker(w) for w in workers]

            firm.set_price(self.wage)
            paid, output, overhead = firm.produce_and_pay(wages, self.bank)
            total_wage_bill += paid
            total_production += output
            total_overhead += overhead

            wage_bill = float(sum(wages))
            pay_ratio = 1.0 if wage_bill <= 0 else max(0.0, min(1.0, paid / wage_bill))
            for worker, wage_amt in zip(workers, wages):
                worker.receive_wage(wage_amt * pay_ratio, firm.unique_id)

        if total_overhead > 0 and self.households:
            per_hh = total_overhead / max(len(self.households), 1)
            for h in self.households:
                h.deposit += per_hh
        self._last_transfers = total_overhead

        prices = np.array([f.price for f in self.firms], dtype=float) if self.firms else np.array([], dtype=float)
        markups = np.array([f.markup for f in self.firms], dtype=float) if self.firms else np.array([], dtype=float)
        self._last_avg_markup = float(markups.mean()) if markups.size else 0.0
        self._last_price_dispersion = float(prices.std()) if prices.size else 0.0

        self._last_production = total_production
        self._last_wage_bill = total_wage_bill

    def _consumption_market(self) -> None:
        total_consumption = 0.0
        total_sold_units = 0.0

        if self.unemployment_transfer > 0:
            unemployed = [h for h in self.households if not h.employed]
            if unemployed:
                for h in unemployed:
                    h.deposit += self.unemployment_transfer
                self._last_unemployment_transfers = self.unemployment_transfer * len(unemployed)
            else:
                self._last_unemployment_transfers = 0.0
        else:
            self._last_unemployment_transfers = 0.0

        for h in self.households:
            total_consumption += h.decide_consumption(self.bank, repay_fraction=self.repayment_fraction)

        if not self.firms:
            self._last_consumption = total_consumption
            self._last_output = 0.0
            return

        prices = np.array([f.price for f in self.firms])
        qualities = np.array([f.productivity for f in self.firms])
        beta = max(self.price_elasticity, 1e-6)
        gamma = self.quality_weight
        utility = -beta * (prices / max(prices.mean(), 1e-6)) + gamma * (qualities / max(qualities.mean(), 1e-6))
        weights_raw = np.exp(utility - utility.max())
        weights = weights_raw / max(weights_raw.sum(), 1e-12)

        forecast_errors: List[float] = []
        turnover_vals: List[float] = []
        inv_gaps: List[float] = []

        for f, w in zip(self.firms, weights):
            demand = total_consumption * w / max(f.price, 1e-6)
            inv_start = f.inventory
            sold = min(f.inventory, demand)
            revenue = sold * f.price
            f.inventory -= sold
            if f.inventory < 0:
                f.inventory = 0.0
            f.cash += revenue
            f.last_revenue = revenue
            total_sold_units += sold

            smoothed = (1 - self.demand_smoothing) * f.last_demand + self.demand_smoothing * demand
            f.last_demand = max(self._demand_floor_value, smoothed)

            mem = self.demand_expectation_memory
            pred = max(1e-6, f.expected_sales_ewma)
            f.expected_sales_ewma = (1.0 - mem) * pred + mem * sold
            f.last_sales_units = sold
            f.last_forecast_error = abs(sold - pred) / max(pred, 1e-6)
            forecast_errors.append(f.last_forecast_error)

            avg_inv = max(1e-6, 0.5 * (inv_start + f.inventory))
            turnover_vals.append(sold / avg_inv)

            target_inventory = max(1e-6, self.inventory_target_days * max(1e-6, f.expected_sales_ewma))
            inv_gaps.append((f.inventory - target_inventory) / target_inventory)

            if f.debt > 0 and f.cash > 0:
                repay = min(f.cash * self.repayment_fraction, f.debt)
                f.debt -= repay
                f.cash -= repay

        self._last_sales_forecast_error = float(np.mean(forecast_errors)) if forecast_errors else 0.0
        self._last_inventory_turnover = float(np.mean(turnover_vals)) if turnover_vals else 0.0
        self._last_inventory_gap = float(np.mean(inv_gaps)) if inv_gaps else 0.0
        self._last_consumption = total_consumption
        self._last_output = total_sold_units

    def _handle_defaults(self) -> None:
        defaults = 0
        defaults_hh = 0
        defaults_firm = 0
        for h in self.households:
            if h.maybe_default(max_debt_income=self.hh_debt_cap_multiplier):
                defaults += 1
                defaults_hh += 1
        for f in self.firms:
            if f.maybe_default(max_debt_revenue=self.firm_debt_cap_multiplier):
                defaults += 1
                defaults_firm += 1
        self._last_defaults = defaults
        self._last_defaults_hh = defaults_hh
        self._last_defaults_firm = defaults_firm

    def _check_balance(self) -> None:
        assets = self.bank.state.loans_firms + self.bank.state.loans_hh + self.bank.state.reserves
        liabilities = self.bank.state.deposits_firms + self.bank.state.deposits_hh + self.bank.state.equity
        ok = abs(assets - liabilities) <= self.balance_tolerance
        self._last_balance_ok = ok
        if not ok and self.log_balance_warnings:
            print(f"[WARN] Bank balance mismatch: assets={assets:.4f}, liabilities={liabilities:.4f}")

    def run_model(self, steps: int = 50) -> None:
        for _ in range(steps):
            self.step()

    def results_dataframe(self):
        return self.datacollector.get_model_vars_dataframe()

    def _household_by_id(self, uid: int) -> Optional[Household]:
        for h in self.households:
            if h.unique_id == uid:
                return h
        return None

    def effective_skill(self, h: Household) -> float:
        return max(0.1, h.skill if h.skill is not None else 1.0)

    def _wage_for_worker(self, h: Household) -> float:
        skill_factor = (h.skill if h.skill is not None else 1.0) - 1.0
        multiplier = 1.0 + self.skill_wage_weight * skill_factor
        multiplier = clamp(multiplier, 0.5, 5.0)
        return self.wage * multiplier
