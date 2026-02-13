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
| wave1 | 22 | 4 | 0.486 | 0.591 | 93.74 | 1.483 | 3.161 | 5.060 | 12.00 |
| wave2 | 22 | 4 | 0.511 | 0.578 | 54.91 | 2.873 | 5.258 | 6.453 | 12.00 |

## Improvement C (Wave2 vs Wave1)
| Metric | Wave1 | Wave2 | Delta |
|---|---:|---:|---:|
| NROY_pct | 93.7428 | 54.9063 | -38.8365 |
| Imax_median | 1.4829 | 2.8735 | 1.3906 |
| RI_shrink_mean | 0.4169 | 12.3450 | 11.9281 |
| bad_run_pct | 28.0833 | 24.4167 | -3.6667 |

## Interpretation G (SA, wave2)
| Component | Share@max reduction | Share mean | Rank@max | Rank mean |
|---|---:|---:|---:|---:|
| MD | 0.2229 | 0.1300 | 1 | 1 |
| OU | 0.1265 | 0.0868 | 2 | 2 |
| EV | 0.0120 | 0.0045 | 3 | 3 |
| CU | 0.0060 | 0.0030 | 4 | 4 |

## Representative Runs
- Selected points: `8`
- Long-run realizations: `24` (expected n_points x n_seeds)
- Employment_mean min: `48.1500`
- Output_mean min: `55.9153`
- Consumption_mean min: `68.1805`
- BankFailed_share max: `0.000000`
- BalanceOK_share min: `1.000000`

## Final Scientific Verdict
- Overall quality gate: `PASS`
| Check | Pass | Value |
|---|---|---|
| NROY wave2 in [25,60]% | PASS | 54.91% |
| I_max median wave2 < 3.2 | PASS | 2.8735 |
| Representative Employment_mean >= 45 | PASS | min=48.1500 |
| Representative Output_mean >= 50 | PASS | min=55.9153 |
| Representative Consumption_mean >= 50 | PASS | min=68.1805 |
| Representative BankFailed_share == 0 | PASS | max=0.000000 |
| Representative BalanceOK_share == 1 | PASS | min=1.000000 |
| SA components EV/OU/MD/CU present | PASS | components=['CU', 'EV', 'MD', 'OU'] |
| SA has >=16 rows | PASS | rows=16 |
| SA ranks present | PASS | rank_cols=True |
| Confirmatory stability (|NROY delta| <= 5.0pp and SA top2 stable) | PASS | nroy_delta_pp=4.77; sa_top2_stable=True |

## Model-Structure Diagnostics
- Structural gates: `n/a` (metrics absent in this run).

## Confirmatory Stability
- confirmatory_nroy_pct: `59.68`
- main_nroy_pct: `54.91`
- nroy_delta_pp: `4.77`
- SA top2 stable: `True`

