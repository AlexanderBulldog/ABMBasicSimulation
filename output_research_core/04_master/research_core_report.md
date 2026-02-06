# Research Core Report

## ABM Input A
- Baseline branch excluded from core analysis.
- Parameter bounds: `scripts/run_operator.py:PARAM_BOUNDS`.
- Wave1 settings: `{'n': 400, 'seed': 0, 'seeds': '0,1,2', 'steps': 180, 'window': 30, 'balance_ok_threshold': 0.6}`.
- Wave2 settings: `{'n': 400, 'seed': 1, 'seeds': '0,1,2', 'steps': 180, 'window': 30, 'balance_ok_threshold': 0.6, 'bounds_kind': 'p05p95'}`.
- HM/SA settings: `{'targets': 'scripts/targets_report.json', 'min_r2_for_history_matching': 0.25, 'improb_threshold': 3.0, 'ev_mode': 'seed_replicates', 'ev_default_quantile': 0.9, 'sa_enable': True, 'sa_domain': 'nroy', 'sa_reductions': '0.1,0.2,0.3,0.4'}`.
- Representative settings: `{'n': 8, 'steps': 400, 'window': 60, 'seeds': '0,1,2', 'min_employment': 55, 'min_output': 60, 'min_consumption': 60, 'min_hh_deposit': 0.2, 'max_bank_resolved_share': 0.06, 'max_bank_bailedout_share': 0.02, 'max_haircut_mean': 0.03, 'max_defaults_hh_rate': 0.01, 'max_defaults_firm_rate': 0.01, 'balance_ok_threshold': 0.98, 'relax_if_needed': True}`.

## Emulator/HM B
| Wave | Trained metrics | Skipped | GPR CV mean | GPR CV median | NROY % | I_max median | I_max p95 | I_max max | metrics_used mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| wave1 | 22 | 4 | 0.486 | 0.591 | 74.28 | 2.321 | 4.133 | 5.276 | 12.00 |
| wave2 | 22 | 4 | 0.502 | 0.653 | 83.50 | 2.158 | 3.703 | 4.238 | 12.00 |

## Improvement C (Wave2 vs Wave1)
| Metric | Wave1 | Wave2 | Delta |
|---|---:|---:|---:|
| NROY_pct | 74.2758 | 83.5006 | 9.2248 |
| Imax_median | 2.3212 | 2.1580 | -0.1632 |
| RI_shrink_mean | 0.7948 | 12.1277 | 11.3329 |
| bad_run_pct | 28.0833 | 25.2500 | -2.8333 |

## Interpretation G (SA, wave2)
| Component | Share@max reduction | Share mean | Rank@max | Rank mean |
|---|---:|---:|---:|---:|
| MD | 0.1602 | 0.0938 | 1 | 1 |
| OU | 0.0374 | 0.0264 | 2 | 2 |
| EV | 0.0093 | 0.0067 | 3 | 3 |
| CU | 0.0067 | 0.0053 | 4 | 4 |

## Representative Runs
- Selected points: `8`
- Long-run realizations: `24` (expected n_points x n_seeds)
- Employment_mean min: `10.0000`
- Output_mean min: `15.1209`
- Consumption_mean min: `12.5128`
- BankFailed_share max: `0.000000`
- BalanceOK_share min: `1.000000`

## Final Scientific Verdict
- Overall quality gate: `FAIL`
| Check | Pass | Value |
|---|---|---|
| NROY wave2 in [25,60]% | FAIL | 83.50% |
| I_max median wave2 < 3.2 | PASS | 2.1580 |
| Representative Employment_mean > 0 | PASS | min=10.0000 |
| Representative Output_mean > 0 | PASS | min=15.1209 |
| Representative Consumption_mean > 0 | PASS | min=12.5128 |
| Representative BankFailed_share == 0 | PASS | max=0.000000 |
| Representative BalanceOK_share == 1 | PASS | min=1.000000 |
| SA components EV/OU/MD/CU present | PASS | components=['CU', 'EV', 'MD', 'OU'] |
| SA has >=16 rows | PASS | rows=16 |
| SA ranks present | PASS | rank_cols=True |

### Failure Reasons & Corrective Actions
- `NROY wave2 in [25,60]%` failed (`83.50%`).
- Suggested actions: retune targets sigma_model/sigma_obs, adjust EV quantile, or refine wave2 bounds.
