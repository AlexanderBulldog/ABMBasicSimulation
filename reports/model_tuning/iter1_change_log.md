# Iter1 Change Log

## Scope
Итерация 1-4 в коде ядра (структурные изменения) с обратной совместимостью колонок.

## Implemented
1. `src/synpop/agents.py`
- Adaptive continuous pricing rule: markup update по `inventory_gap + margin_gap + credit_stress`.
- Price stickiness + max price step limiter.
- Inventory-target based labor demand (`inventory_target_days`) + anti-whiplash (`max_hire_per_step`, `max_fire_per_step`).
- Firm default changed from instant restart to downtime/reentry regime (`firm_reentry_lag`, `firm_reentry_cash_fraction`).
- Household consumption smoothing (income EWMA, precautionary saving, floor/ceiling, unemployment penalty).
- Borrower-aware loan requests for HH/Firms.

2. `src/synpop/bank.py`
- Borrower-specific underwriting in `grant_loan(...)` via DSR-like caps.
- Credit request/rejection counters per step.

3. `src/synpop/model.py`
- New model parameters for pricing/planning/underwriting/reentry/consumption smoothing.
- New diagnostics in DataCollector: `AvgMarkup`, `PriceDispersion`, `InventoryGap`, `SalesForecastError`, `InventoryTurnover`, `CreditRejections`, `FirmDowntimeShare`, `ReentryCount`, `UnemploymentTransfers`.

4. `scripts/run_operator.py`
- Extended `PARAM_BOUNDS` with conservative ranges for new structural parameters.

5. `scripts/targets_report.json`
- Added structural targets (`enabled_for_sa=false` initially).

6. `scripts/make_research_master_report.py`
- Added contour-B structural diagnostics and `quality_gates_structural.csv`.

7. `scripts/compare_model_iters.py`
- Added iteration comparator and composite score for retain/revert decisions.

8. Tests
- Added `tests/test_structural_model.py` (5 checks passed).

## Verification
- `python -m compileall src scripts tests` passed.
- `pytest tests/test_structural_model.py` passed (5/5).
- `run_research_core --mode dry` completed in `output_model_tune_iter1_dry`.

## Note
`iter1_full` started (`output_model_tune_iter1_full`), but full completion is computationally heavy with expanded parameter space; dry contour completed fully and is used for immediate comparison.
