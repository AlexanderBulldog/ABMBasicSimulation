# Research Core Report

## Scientific Verdict v3 (blocking)
- overall_gate_pass_v3: `FAIL`
- legacy_gate_pass(reference): `PASS`

| Check | Pass | Value | Group | Reason |
|---|---|---|---|---|
| NROY wave2 in [25,65]% | PASS | 40.40% | emulator_hm | ok |
| I_max median wave2 < 3.0 | PASS | 2.7848 | emulator_hm | ok |
| I_max p95 wave2 < 4.5 | PASS | 2.9531 | emulator_hm | ok |
| Emulator CV-R2 median >= 0.60 | FAIL | 0.4929 | emulator_hm | emu_median_below_min |
| Emulator CV-R2 share>=0.30 >= 0.65 | FAIL | 0.6296 | emulator_hm | emu_share_below_min |
| bad_run_pct_w2 <= 5% | PASS | 0.00% | abm | ok |
| Representative Employment_mean >= 45 | PASS | min=93.5750 | interpretability | ok |
| Representative Output_mean >= 50 | PASS | min=111.2364 | interpretability | ok |
| Representative Consumption_mean >= 50 | PASS | min=110.4857 | interpretability | ok |
| Representative BankFailed_share == 0 | PASS | max=0.000000 | abm | ok |
| Representative BalanceOK_share == 1 | PASS | min=1.000000 | abm | ok |
| SA components EV/OU/MD/CU present | PASS | components=['CU', 'EV', 'MD', 'OU'] | confirmatory | ok |
| SA has >=16 rows | PASS | rows=16 | confirmatory | ok |
| SA ranks present | PASS | rank_cols=True | confirmatory | ok |
| Structural: PriceDispersion median in [0.03,0.30] | PASS | median=0.0949 | interpretability | ok |
| Structural: |InventoryGap| p95 <= 1.5 | PASS | p95_abs=0.3374 | interpretability | ok |
| Structural: CreditRejections p90 <= 30 | FAIL | p90=98.7550 | interpretability | credit_rejections_too_high |

## Wave-by-wave Convergence
```
 wave                                                                                    wave_dir   n  steps  nroy_pct  Imax_median  Imax_p95  gpr_r2_cv_median  gpr_r2_cv_share_ge_threshold  shrink_mean  shrink_delta  wave_blocking_pass  stable_candidate
    1      C:\Users\Alexander\Desktop\Education\Aspa\code\output_research_core_v3_tuned1\01_wave1 120    120 14.868805     3.558902  5.379670          0.540071                      0.785714    10.899841           NaN               False             False
    2      C:\Users\Alexander\Desktop\Education\Aspa\code\output_research_core_v3_tuned1\02_wave2 120    120 43.333333     2.850839  3.999187          0.663058                      0.814815    21.413488     10.513647                True             False
    3 C:\Users\Alexander\Desktop\Education\Aspa\code\output_research_core_v3_tuned1\waves\wave_03 120    120 60.833333     2.670687  3.850996          0.555810                      0.666667    29.312445      7.898957               False             False
    4 C:\Users\Alexander\Desktop\Education\Aspa\code\output_research_core_v3_tuned1\waves\wave_04 250    120 31.600000     2.899300  3.347527          0.487999                      0.629630    37.378095      8.065650               False             False
    5 C:\Users\Alexander\Desktop\Education\Aspa\code\output_research_core_v3_tuned1\waves\wave_05 250    120 40.400000     2.784783  2.953069          0.492919                      0.629630    44.982828      7.604732               False             False
```

- stopping_diagnostics: `{'stop_reason': 'max_waves_reached', 'final_wave': 5, 'waves_run': 5, 'wave_min': 2, 'wave_max': 5, 'adaptive_waves': True, 'stability_shrink_delta_pp_max': 5.0}`

## Economic Interpretability
- Representative points: `4`
- Employment_mean min: `93.5750`
- Output_mean min: `111.2364`
- Consumption_mean min: `110.4857`
- CreditRejections p90: `95.2000`

## Legacy Contour (reference)
| Check | Pass | Value |
|---|---|---|
| NROY wave2 in [25,60]% | PASS | 40.40% |

