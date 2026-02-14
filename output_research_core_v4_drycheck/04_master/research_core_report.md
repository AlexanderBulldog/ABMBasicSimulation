# Research Core Report

## Scientific Verdict v3 (blocking)
- overall_gate_pass_v3: `FAIL`
- legacy_gate_pass(reference): `PASS`

| Check | Pass | Value | Group | Reason |
|---|---|---|---|---|
| NROY wave2 in [25,65]% | PASS | 55.00% | emulator_hm | ok |
| I_max median wave2 < 3.0 | PASS | 2.5144 | emulator_hm | ok |
| I_max p95 wave2 < 4.5 | PASS | 3.7365 | emulator_hm | ok |
| Emulator CV-R2 median >= 0.60 | FAIL | 0.4793 | emulator_hm | emu_median_below_min |
| Emulator CV-R2 share>=0.30 >= 0.65 | PASS | 0.8276 | emulator_hm | ok |
| bad_run_pct_w2 <= 5% | PASS | 0.00% | abm | ok |
| Representative Employment_mean >= 45 | PASS | min=98.0000 | interpretability | ok |
| Representative Output_mean >= 50 | PASS | min=113.8729 | interpretability | ok |
| Representative Consumption_mean >= 50 | PASS | min=119.3684 | interpretability | ok |
| Representative BankFailed_share == 0 | PASS | max=0.000000 | abm | ok |
| Representative BalanceOK_share == 1 | PASS | min=1.000000 | abm | ok |
| SA components EV/OU/MD/CU present | PASS | components=['CU', 'EV', 'MD', 'OU'] | confirmatory | ok |
| SA has >=16 rows | PASS | rows=16 | confirmatory | ok |
| SA ranks present | PASS | rank_cols=True | confirmatory | ok |
| Structural: PriceDispersion median in [0.03,0.30] | PASS | median=0.0862 | interpretability | ok |
| Structural: |InventoryGap| p95 <= 1.5 | PASS | p95_abs=1.3332 | interpretability | ok |
| Structural: CreditRejectionRate p90 <= 0.85 | FAIL | p90=1.0000 | interpretability | credit_rejection_rate_too_high |

## Wave-by-wave Convergence
```
 wave                                                                                      wave_dir  n  steps  nroy_pct  Imax_median  Imax_p95  gpr_r2_cv_median  gpr_r2_cv_share_ge_threshold  shrink_mean  shrink_delta  wave_blocking_pass  stable_candidate
    1      C:\Users\Alexander\Desktop\Education\Aspa\code\output_research_core_v4_drycheck\01_wave1 20     60 21.052632     3.408758  6.706910          0.722454                      0.827586    34.625943           NaN               False             False
    2      C:\Users\Alexander\Desktop\Education\Aspa\code\output_research_core_v4_drycheck\02_wave2 20     60 35.000000     2.755312  4.347112          0.450871                      0.689655    50.593687     15.967744               False             False
    3 C:\Users\Alexander\Desktop\Education\Aspa\code\output_research_core_v4_drycheck\waves\wave_03 20     60 75.000000     2.034309  3.043706          0.304568                      0.517241    55.008730      4.415043               False             False
    4 C:\Users\Alexander\Desktop\Education\Aspa\code\output_research_core_v4_drycheck\waves\wave_04 20     60 85.000000     2.225207  3.416301          0.396843                      0.677419    59.018745      4.010015               False             False
    5 C:\Users\Alexander\Desktop\Education\Aspa\code\output_research_core_v4_drycheck\waves\wave_05 20     60 55.000000     2.514353  3.736543          0.479345                      0.827586    66.892484      7.873739               False             False
```

- stopping_diagnostics: `{'stop_reason': 'max_waves_reached', 'final_wave': 5, 'waves_run': 5, 'wave_min': 2, 'wave_max': 5, 'adaptive_waves': True, 'stability_shrink_delta_pp_max': 5.0}`

## Economic Interpretability
- Representative points: `2`
- Employment_mean min: `98.0000`
- Output_mean min: `113.8729`
- Consumption_mean min: `119.3684`
- CreditRejections p90: `103.7333`

## Legacy Contour (reference)
| Check | Pass | Value |
|---|---|---|
| NROY wave2 in [25,60]% | PASS | 55.00% |

