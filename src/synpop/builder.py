from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np

from .utils import clamp


class PopulationBuilder:
    """Synthetic population builder (synpop-inspired) for households and firms."""

    def __init__(self, rng: np.random.Generator):
        self.rng = rng

    def households(
        self,
        n: int,
        alpha_mean: float,
        alpha_std: float,
        savings_range: Sequence[float],
        config: Optional[Dict] = None,
        controls: Optional[Dict] = None,
    ) -> List[Dict]:
        cfg = config or {}
        alpha_low, alpha_high = cfg.get("alpha_bounds", (0.4, 0.99))
        sav_cfg = cfg.get("savings", {})
        sav_mode = sav_cfg.get("mode", "uniform")

        specs: List[Dict] = []
        for _ in range(n):
            alpha = clamp(self.rng.normal(alpha_mean, alpha_std), alpha_low, alpha_high)

            if sav_mode == "lognormal":
                mu = sav_cfg.get("mu", 0.5)
                sigma = sav_cfg.get("sigma", 0.5)
                scale = sav_cfg.get("scale", 1.0)
                savings = float(self.rng.lognormal(mu, sigma) * scale)
            else:
                low, high = savings_range
                savings = float(self.rng.uniform(low, high))

            reserve_wage = None
            if "reserve_wage" in cfg:
                rw_cfg = cfg["reserve_wage"]
                base = rw_cfg.get("base", 0.5)
                spread = rw_cfg.get("spread", 0.2)
                reserve_wage = clamp(self.rng.normal(base, spread), 0.1, 2.0)

            skill = None
            if "skill" in cfg:
                sk_cfg = cfg["skill"]
                mu = sk_cfg.get("mu", 1.0)
                sigma = sk_cfg.get("sigma", 0.1)
                skill = clamp(self.rng.normal(mu, sigma), 0.1, 3.0)

            specs.append(
                {
                    "alpha": alpha,
                    "savings": savings,
                    "reserve_wage": reserve_wage,
                    "skill": skill,
                }
            )

        specs = self._apply_household_controls(specs, controls or {})
        return specs

    def firms(
        self,
        n: int,
        base_productivity: float,
        base_cash: float,
        config: Optional[Dict] = None,
        controls: Optional[Dict] = None,
    ) -> List[Dict]:
        cfg = config or {}
        prod_cfg = cfg.get("productivity", {})
        prod_mode = prod_cfg.get("mode", "normal")
        cash_cfg = cfg.get("cash", {})
        cash_mode = cash_cfg.get("mode", "uniform")
        market_shares = cfg.get("market_shares")

        specs: List[Dict] = []
        for _ in range(n):
            if prod_mode == "lognormal":
                mu = prod_cfg.get("mu", 0.0)
                sigma = prod_cfg.get("sigma", 0.1)
                scale = prod_cfg.get("scale", base_productivity)
                productivity = float(self.rng.lognormal(mu, sigma) * scale)
            else:
                sigma = prod_cfg.get("sigma", 0.1)
                productivity = float(clamp(self.rng.normal(base_productivity, sigma), 0.1, 10.0))

            if cash_mode == "lognormal":
                mu = cash_cfg.get("mu", 1.0)
                sigma = cash_cfg.get("sigma", 0.5)
                scale = cash_cfg.get("scale", base_cash)
                cash = float(self.rng.lognormal(mu, sigma) * scale)
            else:
                low, high = cash_cfg.get("range", (0.5 * base_cash, 2 * base_cash))
                cash = float(self.rng.uniform(low, high))

            expected_demand = cfg.get("expected_demand")

            specs.append(
                {
                    "productivity": productivity,
                    "cash": cash,
                    "market_share": None,
                    "expected_demand": expected_demand,
                }
            )

        if market_shares:
            total = sum(market_shares[:n])
            norm = [ms / total for ms in market_shares[:n]] if total > 0 else [1 / n] * n
            for spec, share in zip(specs, norm):
                spec["market_share"] = share

        specs = self._apply_firm_controls(specs, controls or {})
        return specs

    def _apply_household_controls(self, specs: List[Dict], controls: Dict) -> List[Dict]:
        target_savings = controls.get("total_savings")
        if target_savings is not None:
            total = sum(s["savings"] for s in specs) or 1e-9
            scale = target_savings / total
            for s in specs:
                s["savings"] *= scale
        return specs

    def _apply_firm_controls(self, specs: List[Dict], controls: Dict) -> List[Dict]:
        target_cash = controls.get("total_cash")
        if target_cash is not None:
            total = sum(s["cash"] for s in specs) or 1e-9
            scale = target_cash / total
            for s in specs:
                s["cash"] *= scale
        return specs
