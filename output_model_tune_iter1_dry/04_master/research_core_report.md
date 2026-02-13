# Research Core Report

## ABM Input A
- Baseline branch excluded from core analysis.
- Parameter bounds: `scripts/run_operator.py:PARAM_BOUNDS`.
- Wave1 settings: `{'n': 20, 'seed': 0, 'seeds': '0,1,2', 'steps': 60, 'window': 20, 'balance_ok_threshold': 0.6}`.
- Wave2 settings: `{'n': 20, 'seed': 1, 'seeds': '0,1,2', 'steps': 60, 'window': 20, 'balance_ok_threshold': 0.6, 'bounds_kind': 'p05p95'}`.
- HM/SA settings: `{'targets': 'scripts/targets_report.json', 'min_r2_for_history_matching': 0.25, 'improb_threshold': 3.0, 'ev_mode': 'seed_replicates', 'ev_default_quantile': 0.9, 'sigma_calibration_mode_wave1': 'wave1_empirical', 'sigma_calibration_mode_wave2': 'fixed', 'sa_enable': True, 'sa_domain': 'nroy', 'sa_reductions': '0.1,0.2,0.3,0.4'}`.
- Representative settings: `{'n': 2, 'steps': 120, 'window': 30, 'seeds': '0,1,2', 'min_employment': 45, 'min_output': 50, 'min_consumption': 50, 'min_hh_deposit': 0.2, 'max_bank_resolved_share': 0.06, 'max_bank_bailedout_share': 0.02, 'max_haircut_mean': 0.03, 'max_defaults_hh_rate': 0.01, 'max_defaults_firm_rate': 0.01, 'balance_ok_threshold': 0.98, 'relax_if_needed': True}`.

## Emulator/HM B
| Wave | Trained metrics | Skipped | GPR CV mean | GPR CV median | NROY % | I_max median | I_max p95 | I_max max | metrics_used mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| wave1 | 28 | 7 | 0.756 | 0.884 | 68.42 | 2.294 | 6.688 | 7.638 | 15.00 |
| wave2 | 26 | 9 | 0.739 | 0.791 | 75.00 | 2.542 | 5.583 | 5.762 | 14.00 |

## Improvement C (Wave2 vs Wave1)
| Metric | Wave1 | Wave2 | Delta |
|---|---:|---:|---:|
| NROY_pct | 68.4211 | 75.0000 | 6.5789 |
| Imax_median | 2.2936 | 2.5415 | 0.2479 |
| RI_shrink_mean | 9.1291 | 16.0412 | 6.9121 |
| bad_run_pct | 5.0000 | 0.0000 | -5.0000 |

## Interpretation G (SA, wave2)
| Component | Share@max reduction | Share mean | Rank@max | Rank mean |
|---|---:|---:|---:|---:|
| MD | 0.3333 | 0.1167 | 1 | 1 |
| CU | 0.0000 | 0.0000 | 2 | 2 |
| EV | 0.0000 | 0.0000 | 2 | 2 |
| OU | 0.0000 | 0.0000 | 2 | 2 |

## Representative Runs
- Selected points: `2`
- Long-run realizations: `6` (expected n_points x n_seeds)
- Employment_mean min: `89.3000`
- Output_mean min: `115.5273`
- Consumption_mean min: `100.2355`
- BankFailed_share max: `0.000000`
- BalanceOK_share min: `1.000000`

## Final Scientific Verdict
- Overall quality gate: `FAIL`
| Check | Pass | Value |
|---|---|---|
| NROY wave2 in [25,60]% | FAIL | 75.00% |
| I_max median wave2 < 3.2 | PASS | 2.5415 |
| Representative Employment_mean >= 45 | PASS | min=89.3000 |
| Representative Output_mean >= 50 | PASS | min=115.5273 |
| Representative Consumption_mean >= 50 | PASS | min=100.2355 |
| Representative BankFailed_share == 0 | PASS | max=0.000000 |
| Representative BalanceOK_share == 1 | PASS | min=1.000000 |
| SA components EV/OU/MD/CU present | PASS | components=['CU', 'EV', 'MD', 'OU'] |
| SA has >=16 rows | PASS | rows=16 |
| SA ranks present | PASS | rank_cols=True |
| Confirmatory stability (|NROY delta| <= 5.0pp and SA top2 stable) | PASS | nroy_delta_pp=0.00; sa_top2_stable=True |

## Model-Structure Diagnostics
- Structural gate (contour B): `FAIL`
| Check | Pass | Value |
|---|---|---|
| Structural: PriceDispersion median in [0.03,0.30] | PASS | median=0.0528 |
| Structural: |InventoryGap| p95 <= 1.5 | FAIL | p95_abs=10.9593 |
| Structural: CreditRejections p90 <= 10 | FAIL | p90=106.2800 |
| Structural: FirmDowntimeShare max <= 0.20 | PASS | max=0.0000 |

## Confirmatory Stability
- confirmatory_nroy_pct: `75.00`
- main_nroy_pct: `75.00`
- nroy_delta_pp: `0.00`
- SA top2 stable: `True`

### Failure Reasons & Corrective Actions
- `NROY wave2 in [25,60]%` failed (`75.00%`).
- Suggested actions: retune targets sigma_model/sigma_obs, adjust EV quantile, or refine wave2 bounds.
