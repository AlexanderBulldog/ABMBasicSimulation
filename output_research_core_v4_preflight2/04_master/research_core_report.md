# Research Core Report

## Scientific Verdict v3 (blocking)
- overall_gate_pass_v3: `FAIL`
- legacy_gate_pass(reference): `FAIL`

| Check | Pass | Value | Group | Reason |
|---|---|---|---|---|
| NROY wave2 in [25,65]% | FAIL | 79.44% | emulator_hm | nroy_out_of_range |
| I_max median wave2 < 3.0 | PASS | 2.4059 | emulator_hm | ok |
| I_max p95 wave2 < 4.5 | PASS | 3.1117 | emulator_hm | ok |
| Emulator CV-R2 median >= 0.60 | FAIL | 0.3321 | emulator_hm | emu_median_below_min |
| Emulator CV-R2 share>=0.30 >= 0.65 | FAIL | 0.5926 | emulator_hm | emu_share_below_min |
| bad_run_pct_w2 <= 5% | PASS | 0.00% | abm | ok |
| Representative Employment_mean >= 45 | PASS | min=90.3250 | interpretability | ok |
| Representative Output_mean >= 50 | PASS | min=106.8910 | interpretability | ok |
| Representative Consumption_mean >= 50 | PASS | min=109.2046 | interpretability | ok |
| Representative BankFailed_share == 0 | PASS | max=0.000000 | abm | ok |
| Representative BalanceOK_share == 1 | PASS | min=1.000000 | abm | ok |
| SA components EV/OU/MD/CU present | PASS | components=['CU', 'EV', 'MD', 'OU'] | confirmatory | ok |
| SA has >=16 rows | PASS | rows=16 | confirmatory | ok |
| SA ranks present | PASS | rank_cols=True | confirmatory | ok |
| Structural: PriceDispersion median in [0.03,0.30] | PASS | median=0.0936 | interpretability | ok |
| Structural: |InventoryGap| p95 <= 1.5 | PASS | p95_abs=0.3363 | interpretability | ok |
| Structural: CreditRejections p90 <= 90 | FAIL | p90=103.3050 | interpretability | credit_rejections_abs_too_high |
| Structural: CreditRejections p90 improvement vs baseline >= 15% | FAIL | improve=1.43% (baseline=104.805, p90=103.3050) | interpretability | credit_rejections_improvement_too_low |

## Wave-by-wave Convergence
```
 wave                                                                                        wave_dir   n  steps  nroy_pct  Imax_median  Imax_p95  gpr_r2_cv_median  gpr_r2_cv_share_ge_threshold  shrink_mean  shrink_delta  wave_blocking_pass  stable_candidate
    1      C:\Users\Alexander\Desktop\Education\Aspa\code\output_research_core_v4_preflight2\01_wave1 180    140 11.666667     3.209996  3.650614          0.535346                      0.814815    40.861441           NaN               False             False
    2      C:\Users\Alexander\Desktop\Education\Aspa\code\output_research_core_v4_preflight2\02_wave2 180    140 42.222222     2.829363  3.114772          0.333638                      0.629630    48.941360      8.079919               False             False
    3 C:\Users\Alexander\Desktop\Education\Aspa\code\output_research_core_v4_preflight2\waves\wave_03 180    140 79.444444     2.405933  3.111671          0.332110                      0.592593    53.809578      4.868218               False             False
```

- stopping_diagnostics: `{'stop_reason': 'max_waves_reached', 'final_wave': 3, 'waves_run': 3, 'wave_min': 2, 'wave_max': 3, 'adaptive_waves': True, 'stability_shrink_delta_pp_max': 5.0}`

## Economic Interpretability
- Representative points: `4`
- Employment_mean min: `90.3250`
- Output_mean min: `106.8910`
- Consumption_mean min: `109.2046`
- CreditRejections p90: `100.4600`

## Legacy Contour (reference)
| Check | Pass | Value |
|---|---|---|
| NROY wave2 in [25,60]% | FAIL | 79.44% |

