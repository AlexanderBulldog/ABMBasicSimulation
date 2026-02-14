# Research Core Report

## Scientific Verdict v3 (blocking)
- overall_gate_pass_v3: `FAIL`
- legacy_gate_pass(reference): `PASS`

| Check | Pass | Value | Group | Reason |
|---|---|---|---|---|
| NROY wave2 in [25,65]% | PASS | 32.80% | emulator_hm | ok |
| I_max median wave2 < 3.0 | PASS | 2.6524 | emulator_hm | ok |
| I_max p95 wave2 < 4.5 | PASS | 3.0894 | emulator_hm | ok |
| Emulator CV-R2 median >= 0.60 | FAIL | 0.4294 | emulator_hm | emu_median_below_min |
| Emulator CV-R2 share>=0.30 >= 0.65 | PASS | 0.6800 | emulator_hm | ok |
| bad_run_pct_w2 <= 5% | PASS | 0.00% | abm | ok |
| Representative Employment_mean >= 45 | PASS | min=93.5750 | interpretability | ok |
| Representative Output_mean >= 50 | PASS | min=108.7533 | interpretability | ok |
| Representative Consumption_mean >= 50 | PASS | min=111.6325 | interpretability | ok |
| Representative BankFailed_share == 0 | PASS | max=0.000000 | abm | ok |
| Representative BalanceOK_share == 1 | PASS | min=1.000000 | abm | ok |
| SA components EV/OU/MD/CU present | PASS | components=['CU', 'EV', 'MD', 'OU'] | confirmatory | ok |
| SA has >=16 rows | PASS | rows=16 | confirmatory | ok |
| SA ranks present | PASS | rank_cols=True | confirmatory | ok |
| Structural: PriceDispersion median in [0.03,0.30] | PASS | median=0.0768 | interpretability | ok |
| Structural: |InventoryGap| p95 <= 1.5 | PASS | p95_abs=0.2383 | interpretability | ok |
| Structural: CreditRejections p90 <= 30 | FAIL | p90=87.0050 | interpretability | credit_rejections_too_high |

## Wave-by-wave Convergence
```
 wave                                                                                    wave_dir   n  steps  nroy_pct  Imax_median  Imax_p95  gpr_r2_cv_median  gpr_r2_cv_share_ge_threshold  shrink_mean  shrink_delta  wave_blocking_pass  stable_candidate
    1      C:\Users\Alexander\Desktop\Education\Aspa\code\output_research_core_v3_tuned2\01_wave1 120    120 10.495627     3.558902  5.379670          0.540071                      0.785714    13.240345           NaN               False             False
    2      C:\Users\Alexander\Desktop\Education\Aspa\code\output_research_core_v3_tuned2\02_wave2 120    120 20.281690     3.165051  3.532229          0.429700                      0.740741    23.781067     10.540722               False             False
    3 C:\Users\Alexander\Desktop\Education\Aspa\code\output_research_core_v3_tuned2\waves\wave_03 120    120 22.500000     3.209864  3.508736          0.522153                      0.666667    35.193950     11.412884               False             False
    4 C:\Users\Alexander\Desktop\Education\Aspa\code\output_research_core_v3_tuned2\waves\wave_04 250    120 55.600000     2.550376  3.188407          0.262669                      0.481481    40.704689      5.510739               False             False
    5 C:\Users\Alexander\Desktop\Education\Aspa\code\output_research_core_v3_tuned2\waves\wave_05 250    120 32.800000     2.652386  3.089390          0.429376                      0.680000    48.014328      7.309639               False             False
```

- stopping_diagnostics: `{'stop_reason': 'max_waves_reached', 'final_wave': 5, 'waves_run': 5, 'wave_min': 2, 'wave_max': 5, 'adaptive_waves': True, 'stability_shrink_delta_pp_max': 5.0}`

## Economic Interpretability
- Representative points: `4`
- Employment_mean min: `93.5750`
- Output_mean min: `108.7533`
- Consumption_mean min: `111.6325`
- CreditRejections p90: `80.4325`

## Legacy Contour (reference)
| Check | Pass | Value |
|---|---|---|
| NROY wave2 in [25,60]% | PASS | 32.80% |

