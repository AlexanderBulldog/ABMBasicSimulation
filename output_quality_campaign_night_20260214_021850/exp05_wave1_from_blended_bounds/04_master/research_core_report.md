# Research Core Report

## Scientific Verdict v3 (blocking)
- overall_gate_pass_v3: `FAIL`
- legacy_gate_pass(reference): `PASS`

| Check | Pass | Value | Group | Reason |
|---|---|---|---|---|
| NROY wave2 in [25,65]% | PASS | 55.60% | emulator_hm | ok |
| I_max median wave2 < 3.0 | PASS | 2.7337 | emulator_hm | ok |
| I_max p95 wave2 < 4.5 | PASS | 2.8867 | emulator_hm | ok |
| Emulator CV-R2 median >= 0.60 | FAIL | -0.0014 | emulator_hm | emu_median_below_min |
| Emulator CV-R2 share>=0.30 >= 0.65 | FAIL | 0.1667 | emulator_hm | emu_share_below_min |
| bad_run_pct_w2 <= 5% | PASS | 0.00% | abm | ok |
| Representative Employment_mean >= 45 | PASS | min=95.5250 | interpretability | ok |
| Representative Output_mean >= 50 | PASS | min=112.5502 | interpretability | ok |
| Representative Consumption_mean >= 50 | PASS | min=112.6706 | interpretability | ok |
| Representative BankFailed_share == 0 | PASS | max=0.000000 | abm | ok |
| Representative BalanceOK_share == 1 | PASS | min=1.000000 | abm | ok |
| SA components EV/OU/MD/CU present | PASS | components=['CU', 'EV', 'MD', 'OU'] | confirmatory | ok |
| SA has >=16 rows | PASS | rows=16 | confirmatory | ok |
| SA ranks present | PASS | rank_cols=True | confirmatory | ok |
| Structural: PriceDispersion median in [0.03,0.30] | PASS | median=0.0943 | interpretability | ok |
| Structural: |InventoryGap| p95 <= 1.5 | PASS | p95_abs=0.1227 | interpretability | ok |
| Structural: CreditRejections p90 <= 30 | FAIL | p90=96.0500 | interpretability | credit_rejections_too_high |

## Wave-by-wave Convergence
```
 wave                                                                                                                                   wave_dir   n  steps  nroy_pct  Imax_median  Imax_p95  gpr_r2_cv_median  gpr_r2_cv_share_ge_threshold  shrink_mean  shrink_delta  wave_blocking_pass  stable_candidate
    1      C:\Users\Alexander\Desktop\Education\Aspa\code\output_quality_campaign_night_20260214_021850\exp05_wave1_from_blended_bounds\01_wave1 220    150 25.000000     3.087396  3.928861          0.209852                      0.440000    58.143598           NaN               False             False
    2      C:\Users\Alexander\Desktop\Education\Aspa\code\output_quality_campaign_night_20260214_021850\exp05_wave1_from_blended_bounds\02_wave2 220    150 31.363636     2.931816  3.528162          0.197252                      0.360000    63.241990      5.098392               False             False
    3 C:\Users\Alexander\Desktop\Education\Aspa\code\output_quality_campaign_night_20260214_021850\exp05_wave1_from_blended_bounds\waves\wave_03 220    150 62.727273     2.694451  2.953036          0.096758                      0.291667    67.044758      3.802768               False             False
    4 C:\Users\Alexander\Desktop\Education\Aspa\code\output_quality_campaign_night_20260214_021850\exp05_wave1_from_blended_bounds\waves\wave_04 250    150 57.600000     2.715473  2.921523          0.020545                      0.250000    70.748472      3.703714               False             False
    5 C:\Users\Alexander\Desktop\Education\Aspa\code\output_quality_campaign_night_20260214_021850\exp05_wave1_from_blended_bounds\waves\wave_05 250    150 55.600000     2.733655  2.886672         -0.001356                      0.166667    73.957298      3.208826               False             False
```

- stopping_diagnostics: `{'stop_reason': 'max_waves_reached', 'final_wave': 5, 'waves_run': 5, 'wave_min': 2, 'wave_max': 5, 'adaptive_waves': True, 'stability_shrink_delta_pp_max': 5.0}`

## Economic Interpretability
- Representative points: `4`
- Employment_mean min: `95.5250`
- Output_mean min: `112.5502`
- Consumption_mean min: `112.6706`
- CreditRejections p90: `95.7625`

## Legacy Contour (reference)
| Check | Pass | Value |
|---|---|---|
| NROY wave2 in [25,60]% | PASS | 55.60% |

