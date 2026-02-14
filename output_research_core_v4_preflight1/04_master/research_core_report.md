# Research Core Report

## Scientific Verdict v3 (blocking)
- overall_gate_pass_v3: `FAIL`
- legacy_gate_pass(reference): `PASS`

| Check | Pass | Value | Group | Reason |
|---|---|---|---|---|
| NROY wave2 in [25,65]% | PASS | 26.00% | emulator_hm | ok |
| I_max median wave2 < 3.0 | PASS | 2.8871 | emulator_hm | ok |
| I_max p95 wave2 < 4.5 | PASS | 3.1928 | emulator_hm | ok |
| Emulator CV-R2 median >= 0.60 | FAIL | 0.3367 | emulator_hm | emu_median_below_min |
| Emulator CV-R2 share>=0.30 >= 0.65 | FAIL | 0.6296 | emulator_hm | emu_share_below_min |
| bad_run_pct_w2 <= 5% | PASS | 0.00% | abm | ok |
| Representative Employment_mean >= 45 | PASS | min=88.2750 | interpretability | ok |
| Representative Output_mean >= 50 | PASS | min=102.7465 | interpretability | ok |
| Representative Consumption_mean >= 50 | PASS | min=109.3989 | interpretability | ok |
| Representative BankFailed_share == 0 | PASS | max=0.000000 | abm | ok |
| Representative BalanceOK_share == 1 | PASS | min=1.000000 | abm | ok |
| SA components EV/OU/MD/CU present | PASS | components=['CU', 'EV', 'MD', 'OU'] | confirmatory | ok |
| SA has >=16 rows | PASS | rows=16 | confirmatory | ok |
| SA ranks present | PASS | rank_cols=True | confirmatory | ok |
| Structural: PriceDispersion median in [0.03,0.30] | PASS | median=0.0973 | interpretability | ok |
| Structural: |InventoryGap| p95 <= 1.5 | PASS | p95_abs=0.3399 | interpretability | ok |
| Structural: CreditRejections p90 <= 90 | FAIL | p90=103.0050 | interpretability | credit_rejections_abs_too_high |
| Structural: CreditRejections p90 improvement vs baseline >= 15% | FAIL | improve=1.72% (baseline=104.805, p90=103.0050) | interpretability | credit_rejections_improvement_too_low |

## Wave-by-wave Convergence
```
 wave                                                                                        wave_dir   n  steps  nroy_pct  Imax_median  Imax_p95  gpr_r2_cv_median  gpr_r2_cv_share_ge_threshold  shrink_mean  shrink_delta  wave_blocking_pass  stable_candidate
    1      C:\Users\Alexander\Desktop\Education\Aspa\code\output_research_core_v4_preflight1\01_wave1 220    150  3.588144     3.592810  5.476044          0.670811                      0.928571    22.441901           NaN               False             False
    2      C:\Users\Alexander\Desktop\Education\Aspa\code\output_research_core_v4_preflight1\02_wave2 220    150 55.454545     2.570697  3.219190          0.492330                      0.740741    23.499983      1.058083               False             False
    3 C:\Users\Alexander\Desktop\Education\Aspa\code\output_research_core_v4_preflight1\waves\wave_03 220    150 29.090909     2.860281  3.071818          0.361147                      0.666667    33.113755      9.613772               False             False
    4 C:\Users\Alexander\Desktop\Education\Aspa\code\output_research_core_v4_preflight1\waves\wave_04 250    150 90.800000     2.234876  2.673654          0.309865                      0.592593    40.208398      7.094644               False             False
    5 C:\Users\Alexander\Desktop\Education\Aspa\code\output_research_core_v4_preflight1\waves\wave_05 250    150 26.000000     2.887075  3.192829          0.336687                      0.629630    47.910487      7.702088               False             False
```

- stopping_diagnostics: `{'stop_reason': 'max_waves_reached', 'final_wave': 5, 'waves_run': 5, 'wave_min': 2, 'wave_max': 5, 'adaptive_waves': True, 'stability_shrink_delta_pp_max': 5.0}`

## Economic Interpretability
- Representative points: `4`
- Employment_mean min: `88.2750`
- Output_mean min: `102.7465`
- Consumption_mean min: `109.3989`
- CreditRejections p90: `104.6225`

## Legacy Contour (reference)
| Check | Pass | Value |
|---|---|---|
| NROY wave2 in [25,60]% | PASS | 26.00% |

