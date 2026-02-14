# Research Core Report

## Scientific Verdict v3 (blocking)
- overall_gate_pass_v3: `FAIL`
- legacy_gate_pass(reference): `PASS`

| Check | Pass | Value | Group | Reason |
|---|---|---|---|---|
| NROY wave2 in [25,65]% | PASS | 56.80% | emulator_hm | ok |
| I_max median wave2 < 3.0 | PASS | 2.7246 | emulator_hm | ok |
| I_max p95 wave2 < 4.5 | PASS | 2.8756 | emulator_hm | ok |
| Emulator CV-R2 median >= 0.60 | FAIL | 0.2854 | emulator_hm | emu_median_below_min |
| Emulator CV-R2 share>=0.30 >= 0.65 | FAIL | 0.4815 | emulator_hm | emu_share_below_min |
| bad_run_pct_w2 <= 5% | PASS | 0.00% | abm | ok |
| Representative Employment_mean >= 45 | PASS | min=94.8000 | interpretability | ok |
| Representative Output_mean >= 50 | PASS | min=98.2243 | interpretability | ok |
| Representative Consumption_mean >= 50 | PASS | min=111.5508 | interpretability | ok |
| Representative BankFailed_share == 0 | PASS | max=0.000000 | abm | ok |
| Representative BalanceOK_share == 1 | PASS | min=1.000000 | abm | ok |
| SA components EV/OU/MD/CU present | PASS | components=['CU', 'EV', 'MD', 'OU'] | confirmatory | ok |
| SA has >=16 rows | PASS | rows=16 | confirmatory | ok |
| SA ranks present | PASS | rank_cols=True | confirmatory | ok |
| Structural: PriceDispersion median in [0.03,0.30] | PASS | median=0.1016 | interpretability | ok |
| Structural: |InventoryGap| p95 <= 1.5 | PASS | p95_abs=0.2879 | interpretability | ok |
| Structural: CreditRejections p90 <= 30 | FAIL | p90=97.7500 | interpretability | credit_rejections_too_high |

## Wave-by-wave Convergence
```
 wave                                                                                                                        wave_dir   n  steps  nroy_pct  Imax_median  Imax_p95  gpr_r2_cv_median  gpr_r2_cv_share_ge_threshold  shrink_mean  shrink_delta  wave_blocking_pass  stable_candidate
    1      C:\Users\Alexander\Desktop\Education\Aspa\code\output_quality_campaign_night_20260214_021850\exp01_baseline_dense\01_wave1 180    140  7.899807     3.735055  5.265155          0.633692                      0.935484    13.550890           NaN               False             False
    2      C:\Users\Alexander\Desktop\Education\Aspa\code\output_quality_campaign_night_20260214_021850\exp01_baseline_dense\02_wave2 180    140 35.150376     2.944813  4.856765          0.657382                      0.866667    18.527093      4.976203               False             False
    3 C:\Users\Alexander\Desktop\Education\Aspa\code\output_quality_campaign_night_20260214_021850\exp01_baseline_dense\waves\wave_03 180    140 46.666667     2.788699  3.895917          0.579739                      0.740741    28.516396      9.989303               False             False
    4 C:\Users\Alexander\Desktop\Education\Aspa\code\output_quality_campaign_night_20260214_021850\exp01_baseline_dense\waves\wave_04 250    140 34.400000     2.864070  2.990459          0.687812                      0.703704    36.994015      8.477619                True             False
    5 C:\Users\Alexander\Desktop\Education\Aspa\code\output_quality_campaign_night_20260214_021850\exp01_baseline_dense\waves\wave_05 250    140 56.800000     2.724592  2.875606          0.285397                      0.481481    43.512600      6.518585               False             False
```

- stopping_diagnostics: `{'stop_reason': 'max_waves_reached', 'final_wave': 5, 'waves_run': 5, 'wave_min': 2, 'wave_max': 5, 'adaptive_waves': True, 'stability_shrink_delta_pp_max': 5.0}`

## Economic Interpretability
- Representative points: `4`
- Employment_mean min: `94.8000`
- Output_mean min: `98.2243`
- Consumption_mean min: `111.5508`
- CreditRejections p90: `87.9300`

## Legacy Contour (reference)
| Check | Pass | Value |
|---|---|---|
| NROY wave2 in [25,60]% | PASS | 56.80% |

