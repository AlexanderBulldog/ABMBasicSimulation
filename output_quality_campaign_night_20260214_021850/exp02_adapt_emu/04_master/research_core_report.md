# Research Core Report

## Scientific Verdict v3 (blocking)
- overall_gate_pass_v3: `FAIL`
- legacy_gate_pass(reference): `PASS`

| Check | Pass | Value | Group | Reason |
|---|---|---|---|---|
| NROY wave2 in [25,65]% | PASS | 54.00% | emulator_hm | ok |
| I_max median wave2 < 3.0 | PASS | 2.6833 | emulator_hm | ok |
| I_max p95 wave2 < 4.5 | PASS | 3.5450 | emulator_hm | ok |
| Emulator CV-R2 median >= 0.60 | FAIL | 0.3781 | emulator_hm | emu_median_below_min |
| Emulator CV-R2 share>=0.30 >= 0.65 | FAIL | 0.5185 | emulator_hm | emu_share_below_min |
| bad_run_pct_w2 <= 5% | PASS | 0.00% | abm | ok |
| Representative Employment_mean >= 45 | PASS | min=84.4500 | interpretability | ok |
| Representative Output_mean >= 50 | PASS | min=98.1696 | interpretability | ok |
| Representative Consumption_mean >= 50 | PASS | min=105.6702 | interpretability | ok |
| Representative BankFailed_share == 0 | PASS | max=0.000000 | abm | ok |
| Representative BalanceOK_share == 1 | PASS | min=1.000000 | abm | ok |
| SA components EV/OU/MD/CU present | PASS | components=['CU', 'EV', 'MD', 'OU'] | confirmatory | ok |
| SA has >=16 rows | PASS | rows=16 | confirmatory | ok |
| SA ranks present | PASS | rank_cols=True | confirmatory | ok |
| Structural: PriceDispersion median in [0.03,0.30] | PASS | median=0.1022 | interpretability | ok |
| Structural: |InventoryGap| p95 <= 1.5 | PASS | p95_abs=0.2228 | interpretability | ok |
| Structural: CreditRejections p90 <= 30 | FAIL | p90=97.8050 | interpretability | credit_rejections_too_high |

## Wave-by-wave Convergence
```
 wave                                                                                                                   wave_dir   n  steps  nroy_pct  Imax_median  Imax_p95  gpr_r2_cv_median  gpr_r2_cv_share_ge_threshold  shrink_mean  shrink_delta  wave_blocking_pass  stable_candidate
    1      C:\Users\Alexander\Desktop\Education\Aspa\code\output_quality_campaign_night_20260214_021850\exp02_adapt_emu\01_wave1 220    150  9.905660     3.457786  5.365782          0.556271                      0.838710    12.215470           NaN               False             False
    2      C:\Users\Alexander\Desktop\Education\Aspa\code\output_quality_campaign_night_20260214_021850\exp02_adapt_emu\02_wave2 220    150 30.898021     3.015848  4.494818          0.618362                      0.741935    22.446220     10.230750               False             False
    3 C:\Users\Alexander\Desktop\Education\Aspa\code\output_quality_campaign_night_20260214_021850\exp02_adapt_emu\waves\wave_03 220    150 32.776935     2.825689  3.963474          0.519419                      0.666667    31.751787      9.305567               False             False
    4 C:\Users\Alexander\Desktop\Education\Aspa\code\output_quality_campaign_night_20260214_021850\exp02_adapt_emu\waves\wave_04 250    150 50.800000     2.748209  3.400073          0.540330                      0.518519    38.790689      7.038902               False             False
    5 C:\Users\Alexander\Desktop\Education\Aspa\code\output_quality_campaign_night_20260214_021850\exp02_adapt_emu\waves\wave_05 250    150 54.000000     2.683262  3.545044          0.378129                      0.518519    45.513982      6.723292               False             False
```

- stopping_diagnostics: `{'stop_reason': 'max_waves_reached', 'final_wave': 5, 'waves_run': 5, 'wave_min': 2, 'wave_max': 5, 'adaptive_waves': True, 'stability_shrink_delta_pp_max': 5.0}`

## Economic Interpretability
- Representative points: `4`
- Employment_mean min: `84.4500`
- Output_mean min: `98.1696`
- Consumption_mean min: `105.6702`
- CreditRejections p90: `90.7425`

## Legacy Contour (reference)
| Check | Pass | Value |
|---|---|---|
| NROY wave2 in [25,60]% | PASS | 54.00% |

