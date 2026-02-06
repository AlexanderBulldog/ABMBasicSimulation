# Research Core Report

## ABM Input A
- Baseline branch excluded from core analysis.
- Parameter bounds: `scripts/run_operator.py:PARAM_BOUNDS`.
- Wave1 settings: `{'n': 20, 'seed': 0, 'seeds': '0,1,2', 'steps': 60, 'window': 20, 'balance_ok_threshold': 0.6}`.
- Wave2 settings: `{'n': 20, 'seed': 1, 'seeds': '0,1,2', 'steps': 60, 'window': 20, 'balance_ok_threshold': 0.6, 'bounds_kind': 'p05p95'}`.
- HM/SA settings: `{'targets': 'scripts/targets_report.json', 'min_r2_for_history_matching': 0.25, 'improb_threshold': 3.0, 'ev_mode': 'seed_replicates', 'ev_default_quantile': 0.9, 'sa_enable': True, 'sa_domain': 'nroy', 'sa_reductions': '0.1,0.2,0.3,0.4'}`.
- Representative settings: `{'n': 2, 'steps': 120, 'window': 30, 'seeds': '0,1,2', 'min_employment': 55, 'min_output': 60, 'min_consumption': 60, 'min_hh_deposit': 0.2, 'max_bank_resolved_share': 0.06, 'max_bank_bailedout_share': 0.02, 'max_haircut_mean': 0.03, 'max_defaults_hh_rate': 0.01, 'max_defaults_firm_rate': 0.01, 'balance_ok_threshold': 0.98, 'relax_if_needed': True}`.

## Emulator/HM B
| Wave | Trained metrics | Skipped | GPR CV mean | GPR CV median | NROY % | I_max median | I_max p95 | I_max max | metrics_used mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| wave1 | 20 | 6 | 0.652 | 0.851 | 16.07 | 4.029 | 6.047 | 6.103 | 12.00 |
| wave2 | 15 | 11 | 0.595 | 0.818 | 100.00 | 1.487 | 2.106 | 2.136 | 8.00 |

## Improvement C (Wave2 vs Wave1)
| Metric | Wave1 | Wave2 | Delta |
|---|---:|---:|---:|
| NROY_pct | 16.0714 | 100.0000 | 83.9286 |
| Imax_median | 4.0295 | 1.4869 | -2.5426 |
| RI_shrink_mean | 56.7698 | 58.4932 | 1.7233 |
| bad_run_pct | 6.6667 | 0.0000 | -6.6667 |

## Interpretation G (SA, wave2)
| Component | Share@max reduction | Share mean | Rank@max | Rank mean |
|---|---:|---:|---:|---:|
| CU | 0.0000 | 0.0000 | 1 | 1 |
| EV | 0.0000 | 0.0000 | 1 | 1 |
| MD | 0.0000 | 0.0000 | 1 | 1 |
| OU | 0.0000 | 0.0000 | 1 | 1 |

## Representative Runs
- Selected points: `2`
- Long-run realizations: `6` (expected n_points x n_seeds)
- Employment_mean min: `56.0000`
- Output_mean min: `63.4322`
- Consumption_mean min: `69.7861`
- BankFailed_share max: `0.000000`
- BalanceOK_share min: `1.000000`

## Final Scientific Verdict
- Overall quality gate: `FAIL`
| Check | Pass | Value |
|---|---|---|
| NROY wave2 in [25,60]% | FAIL | 100.00% |
| I_max median wave2 < 3.2 | PASS | 1.4869 |
| Representative Employment_mean > 0 | PASS | min=56.0000 |
| Representative Output_mean > 0 | PASS | min=63.4322 |
| Representative Consumption_mean > 0 | PASS | min=69.7861 |
| Representative BankFailed_share == 0 | PASS | max=0.000000 |
| Representative BalanceOK_share == 1 | PASS | min=1.000000 |
| SA components EV/OU/MD/CU present | PASS | components=['CU', 'EV', 'MD', 'OU'] |
| SA has >=16 rows | PASS | rows=16 |
| SA ranks present | PASS | rank_cols=True |

### Failure Reasons & Corrective Actions
- `NROY wave2 in [25,60]%` failed (`100.00%`).
- Suggested actions: retune targets sigma_model/sigma_obs, adjust EV quantile, or refine wave2 bounds.
