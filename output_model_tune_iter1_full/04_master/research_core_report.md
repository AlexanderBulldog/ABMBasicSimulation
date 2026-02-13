# Research Core Report

## ABM Input A
- Baseline branch excluded from core analysis.
- Parameter bounds: `scripts/run_operator.py:PARAM_BOUNDS`.
- Wave1 settings: `{'n': 400, 'seed': 0, 'seeds': '0,1,2', 'steps': 180, 'window': 30, 'balance_ok_threshold': 0.6}`.
- Wave2 settings: `{'n': 400, 'seed': 1, 'seeds': '0,1,2', 'steps': 180, 'window': 30, 'balance_ok_threshold': 0.6, 'bounds_kind': 'p05p95'}`.
- HM/SA settings: `{'targets': 'scripts/targets_report.json', 'min_r2_for_history_matching': 0.25, 'improb_threshold': 3.0, 'ev_mode': 'seed_replicates', 'ev_default_quantile': 0.9, 'sigma_calibration_mode_wave1': 'wave1_empirical', 'sigma_calibration_mode_wave2': 'fixed', 'sa_enable': True, 'sa_domain': 'nroy', 'sa_reductions': '0.1,0.2,0.3,0.4'}`.
- Representative settings: `{'n': 8, 'steps': 400, 'window': 60, 'seeds': '0,1,2', 'min_employment': 45, 'min_output': 50, 'min_consumption': 50, 'min_hh_deposit': 0.2, 'max_bank_resolved_share': 0.06, 'max_bank_bailedout_share': 0.02, 'max_haircut_mean': 0.03, 'max_defaults_hh_rate': 0.01, 'max_defaults_firm_rate': 0.01, 'balance_ok_threshold': 0.98, 'relax_if_needed': True}`.

## Emulator/HM B
| Wave | Trained metrics | Skipped | GPR CV mean | GPR CV median | NROY % | I_max median | I_max p95 | I_max max | metrics_used mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| wave1 | 27 | 8 | 0.614 | 0.735 | 97.44 | 1.479 | 2.485 | 6.095 | 14.00 |
| wave2 | 27 | 8 | -18.402 | 0.729 | 98.75 | 1.838 | 2.668 | 8.608 | 14.00 |

## Improvement C (Wave2 vs Wave1)
| Metric | Wave1 | Wave2 | Delta |
|---|---:|---:|---:|
| NROY_pct | 97.4381 | 98.7458 | 1.3077 |
| Imax_median | 1.4791 | 1.8382 | 0.3591 |
| RI_shrink_mean | 0.2630 | 10.5568 | 10.2938 |
| bad_run_pct | 2.4167 | 0.3333 | -2.0833 |

## Interpretation G (SA, wave2)
| Component | Share@max reduction | Share mean | Rank@max | Rank mean |
|---|---:|---:|---:|---:|
| MD | 0.0771 | 0.0366 | 1 | 1 |
| OU | 0.0135 | 0.0055 | 2 | 2 |
| CU | 0.0051 | 0.0032 | 3 | 3 |
| EV | 0.0000 | 0.0000 | 4 | 4 |

## Representative Runs
- Selected points: `8`
- Long-run realizations: `24` (expected n_points x n_seeds)
- Employment_mean min: `93.2667`
- Output_mean min: `100.9099`
- Consumption_mean min: `94.1543`
- BankFailed_share max: `0.000000`
- BalanceOK_share min: `1.000000`

## Final Scientific Verdict
- Overall quality gate: `FAIL`
| Check | Pass | Value |
|---|---|---|
| NROY wave2 in [25,60]% | FAIL | 98.75% |
| I_max median wave2 < 3.2 | PASS | 1.8382 |
| Representative Employment_mean >= 45 | PASS | min=93.2667 |
| Representative Output_mean >= 50 | PASS | min=100.9099 |
| Representative Consumption_mean >= 50 | PASS | min=94.1543 |
| Representative BankFailed_share == 0 | PASS | max=0.000000 |
| Representative BalanceOK_share == 1 | PASS | min=1.000000 |
| SA components EV/OU/MD/CU present | PASS | components=['CU', 'EV', 'MD', 'OU'] |
| SA has >=16 rows | PASS | rows=16 |
| SA ranks present | PASS | rank_cols=True |
| Confirmatory stability (|NROY delta| <= 5.0pp and SA top2 stable) | PASS | nroy_delta_pp=-1.83; sa_top2_stable=True |

## Model-Structure Diagnostics
- Structural gate (contour B): `FAIL`
| Check | Pass | Value |
|---|---|---|
| Structural: PriceDispersion median in [0.03,0.30] | PASS | median=0.0658 |
| Structural: |InventoryGap| p95 <= 1.5 | PASS | p95_abs=0.2686 |
| Structural: CreditRejections p90 <= 10 | FAIL | p90=105.0000 |
| Structural: FirmDowntimeShare max <= 0.20 | PASS | max=0.0000 |

## Confirmatory Stability
- confirmatory_nroy_pct: `96.91`
- main_nroy_pct: `98.75`
- nroy_delta_pp: `-1.83`
- SA top2 stable: `True`

### Failure Reasons & Corrective Actions
- `NROY wave2 in [25,60]%` failed (`98.75%`).
- Suggested actions: retune targets sigma_model/sigma_obs, adjust EV quantile, or refine wave2 bounds.
