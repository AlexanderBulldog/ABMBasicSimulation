# Research Core Report

## Scientific Verdict v3 (blocking)
- overall_gate_pass_v3: `FAIL`
- legacy_gate_pass(reference): `PASS`

| Check | Pass | Value | Group | Reason |
|---|---|---|---|---|
| NROY wave2 in [25,65]% | PASS | 53.57% | emulator_hm | ok |
| I_max median wave2 < 3.0 | PASS | 2.6654 | emulator_hm | ok |
| I_max p95 wave2 < 4.5 | PASS | 3.7626 | emulator_hm | ok |
| Emulator CV-R2 median >= 0.60 | FAIL | 0.5965 | emulator_hm | emu_median_below_min |
| Emulator CV-R2 share>=0.30 >= 0.65 | PASS | 0.9474 | emulator_hm | ok |
| bad_run_pct_w2 <= 5% | PASS | 0.00% | abm | ok |
| Representative Employment_mean >= 45 | PASS | min=92.2500 | interpretability | ok |
| Representative Output_mean >= 50 | PASS | min=101.1746 | interpretability | ok |
| Representative Consumption_mean >= 50 | PASS | min=102.1440 | interpretability | ok |
| Representative BankFailed_share == 0 | PASS | max=0.000000 | abm | ok |
| Representative BalanceOK_share == 1 | PASS | min=1.000000 | abm | ok |
| SA components EV/OU/MD/CU present | PASS | components=['CU', 'EV', 'MD', 'OU'] | confirmatory | ok |
| SA has >=16 rows | PASS | rows=16 | confirmatory | ok |
| SA ranks present | PASS | rank_cols=True | confirmatory | ok |
| Structural: PriceDispersion median in [0.03,0.30] | PASS | median=0.0883 | interpretability | ok |
| Structural: |InventoryGap| p95 <= 1.5 | PASS | p95_abs=0.3465 | interpretability | ok |
| Structural: CreditRejectionRate mean <= 0.80 | PASS | mean=0.6652; p90=0.9090 | interpretability | ok |

## Wave-by-wave Convergence
```
 wave                                                                                        wave_dir   n  steps  nroy_pct  Imax_median  Imax_p95  gpr_r2_cv_median  gpr_r2_cv_share_ge_threshold  shrink_mean  shrink_delta  wave_blocking_pass  stable_candidate
    1      C:\Users\Alexander\Desktop\Education\Aspa\code\output_research_core_v5_preflight1\01_wave1 140    120 47.174447     2.793005  4.949705          0.402259                      0.655172     3.222799           NaN               False             False
    2      C:\Users\Alexander\Desktop\Education\Aspa\code\output_research_core_v5_preflight1\02_wave2 140    120 54.285714     2.620888  4.202458          0.642203                      0.862069    14.909779     11.686980                True             False
    3 C:\Users\Alexander\Desktop\Education\Aspa\code\output_research_core_v5_preflight1\waves\wave_03 140    120 53.571429     2.665411  3.762629          0.389568                      0.600000    24.370904      9.461125               False             False
```

- stopping_diagnostics: `{'stop_reason': 'max_waves_reached', 'final_wave': 3, 'waves_run': 3, 'wave_min': 2, 'wave_max': 3, 'adaptive_waves': True, 'stability_shrink_delta_pp_max': 5.0}`

## Economic Interpretability
- Representative points: `4`
- Employment_mean min: `92.2500`
- Output_mean min: `101.1746`
- Consumption_mean min: `102.1440`
- CreditRejections p90: `78.1050`

## Legacy Contour (reference)
| Check | Pass | Value |
|---|---|---|
| NROY wave2 in [25,60]% | PASS | 53.57% |

