# Research Core Report

## Scientific Verdict v3 (blocking)
- overall_gate_pass_v3: `FAIL`
- legacy_gate_pass(reference): `PASS`

| Check | Pass | Value | Group | Reason |
|---|---|---|---|---|
| NROY wave2 in [25,65]% | PASS | 55.20% | emulator_hm | ok |
| I_max median wave2 < 3.0 | PASS | 2.7334 | emulator_hm | ok |
| I_max p95 wave2 < 4.5 | PASS | 2.9248 | emulator_hm | ok |
| Emulator CV-R2 median >= 0.60 | FAIL | 0.1074 | emulator_hm | emu_median_below_min |
| Emulator CV-R2 share>=0.30 >= 0.65 | FAIL | 0.2000 | emulator_hm | emu_share_below_min |
| bad_run_pct_w2 <= 5% | PASS | 0.00% | abm | ok |
| Representative Employment_mean >= 45 | PASS | min=98.3000 | interpretability | ok |
| Representative Output_mean >= 50 | PASS | min=114.1152 | interpretability | ok |
| Representative Consumption_mean >= 50 | PASS | min=129.9785 | interpretability | ok |
| Representative BankFailed_share == 0 | PASS | max=0.000000 | abm | ok |
| Representative BalanceOK_share == 1 | PASS | min=1.000000 | abm | ok |
| SA components EV/OU/MD/CU present | PASS | components=['CU', 'EV', 'MD', 'OU'] | confirmatory | ok |
| SA has >=16 rows | PASS | rows=16 | confirmatory | ok |
| SA ranks present | PASS | rank_cols=True | confirmatory | ok |
| Structural: PriceDispersion median in [0.03,0.30] | PASS | median=0.1076 | interpretability | ok |
| Structural: |InventoryGap| p95 <= 1.5 | PASS | p95_abs=0.2654 | interpretability | ok |
| Structural: CreditRejections p90 <= 30 | FAIL | p90=81.1500 | interpretability | credit_rejections_too_high |

## Wave-by-wave Convergence
```
 wave                                                                                                                                  wave_dir   n  steps  nroy_pct  Imax_median  Imax_p95  gpr_r2_cv_median  gpr_r2_cv_share_ge_threshold  shrink_mean  shrink_delta  wave_blocking_pass  stable_candidate
    1      C:\Users\Alexander\Desktop\Education\Aspa\code\output_quality_campaign_night_20260214_021850\exp04_wave1_from_credit_bounds\01_wave1 200    140       3.0     3.459423  3.827375          0.377940                      0.555556    64.777443           NaN               False             False
    2      C:\Users\Alexander\Desktop\Education\Aspa\code\output_quality_campaign_night_20260214_021850\exp04_wave1_from_credit_bounds\02_wave2 200    140      39.0     2.816948  3.032145          0.317304                      0.576923    65.421813      0.644370               False             False
    3 C:\Users\Alexander\Desktop\Education\Aspa\code\output_quality_campaign_night_20260214_021850\exp04_wave1_from_credit_bounds\waves\wave_03 200    140      35.0     2.831402  3.180502          0.253997                      0.346154    69.886003      4.464190               False             False
    4 C:\Users\Alexander\Desktop\Education\Aspa\code\output_quality_campaign_night_20260214_021850\exp04_wave1_from_credit_bounds\waves\wave_04 250    140      27.2     2.841179  3.188222          0.171081                      0.240000    73.378178      3.492175               False             False
    5 C:\Users\Alexander\Desktop\Education\Aspa\code\output_quality_campaign_night_20260214_021850\exp04_wave1_from_credit_bounds\waves\wave_05 250    140      55.2     2.733373  2.924752          0.107429                      0.200000    76.345493      2.967316               False             False
```

- stopping_diagnostics: `{'stop_reason': 'max_waves_reached', 'final_wave': 5, 'waves_run': 5, 'wave_min': 2, 'wave_max': 5, 'adaptive_waves': True, 'stability_shrink_delta_pp_max': 5.0}`

## Economic Interpretability
- Representative points: `4`
- Employment_mean min: `98.3000`
- Output_mean min: `114.1152`
- Consumption_mean min: `129.9785`
- CreditRejections p90: `78.8825`

## Legacy Contour (reference)
| Check | Pass | Value |
|---|---|---|
| NROY wave2 in [25,60]% | PASS | 55.20% |

