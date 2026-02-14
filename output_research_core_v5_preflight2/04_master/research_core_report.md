# Research Core Report

## Scientific Verdict v3 (blocking)
- overall_gate_pass_v3: `PASS`
- legacy_gate_pass(reference): `PASS`

| Check | Pass | Value | Group | Reason |
|---|---|---|---|---|
| NROY wave2 in [25,65]% | PASS | 58.57% | emulator_hm | ok |
| I_max median wave2 < 3.0 | PASS | 2.3729 | emulator_hm | ok |
| I_max p95 wave2 < 4.5 | PASS | 3.3152 | emulator_hm | ok |
| Emulator CV-R2 median >= 0.60 | PASS | 0.6193 | emulator_hm | ok |
| Emulator CV-R2 share>=0.30 >= 0.65 | PASS | 1.0000 | emulator_hm | ok |
| bad_run_pct_w2 <= 5% | PASS | 0.00% | abm | ok |
| Representative Employment_mean >= 45 | PASS | min=90.0750 | interpretability | ok |
| Representative Output_mean >= 50 | PASS | min=98.8993 | interpretability | ok |
| Representative Consumption_mean >= 50 | PASS | min=103.2140 | interpretability | ok |
| Representative BankFailed_share == 0 | PASS | max=0.000000 | abm | ok |
| Representative BalanceOK_share == 1 | PASS | min=1.000000 | abm | ok |
| SA components EV/OU/MD/CU present | PASS | components=['CU', 'EV', 'MD', 'OU'] | confirmatory | ok |
| SA has >=16 rows | PASS | rows=16 | confirmatory | ok |
| SA ranks present | PASS | rank_cols=True | confirmatory | ok |
| Structural: PriceDispersion median in [0.03,0.30] | PASS | median=0.0882 | interpretability | ok |
| Structural: |InventoryGap| p95 <= 1.5 | PASS | p95_abs=0.3327 | interpretability | ok |
| Structural: CreditRejectionRate mean <= 0.80 | PASS | mean=0.7005; p90=0.9722 | interpretability | ok |

## Wave-by-wave Convergence
```
 wave                                                                                        wave_dir   n  steps  nroy_pct  Imax_median  Imax_p95  gpr_r2_cv_median  gpr_r2_cv_share_ge_threshold  shrink_mean  shrink_delta  wave_blocking_pass  stable_candidate
    1      C:\Users\Alexander\Desktop\Education\Aspa\code\output_research_core_v5_preflight2\01_wave1 140    120 44.963145     2.793005  4.949705          0.628247                           1.0     3.222799           NaN               False             False
    2      C:\Users\Alexander\Desktop\Education\Aspa\code\output_research_core_v5_preflight2\02_wave2 140    120 26.491647     2.841620  3.730536          0.692882                           1.0    17.806930     14.584131                True             False
    3 C:\Users\Alexander\Desktop\Education\Aspa\code\output_research_core_v5_preflight2\waves\wave_03 140    120 58.571429     2.372916  3.315194          0.619282                           1.0    25.686944      7.880014                True             False
```

- stopping_diagnostics: `{'stop_reason': 'max_waves_reached', 'final_wave': 3, 'waves_run': 3, 'wave_min': 2, 'wave_max': 3, 'adaptive_waves': True, 'stability_shrink_delta_pp_max': 5.0}`

## Economic Interpretability
- Representative points: `4`
- Employment_mean min: `90.0750`
- Output_mean min: `98.8993`
- Consumption_mean min: `103.2140`
- CreditRejections p90: `89.4925`

## Legacy Contour (reference)
| Check | Pass | Value |
|---|---|---|
| NROY wave2 in [25,60]% | PASS | 58.57% |

