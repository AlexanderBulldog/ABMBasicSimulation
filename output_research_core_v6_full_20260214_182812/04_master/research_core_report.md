# Research Core Report

## Scientific Verdict v3 (blocking)
- overall_gate_pass_v3: `FAIL`
- legacy_gate_pass(reference): `FAIL`

| Check | Pass | Value | Group | Reason |
|---|---|---|---|---|
| NROY wave2 in [25,65]% | FAIL | 68.40% | emulator_hm | nroy_out_of_range |
| I_max median wave2 < 3.0 | PASS | 2.4292 | emulator_hm | ok |
| I_max p95 wave2 < 4.5 | PASS | 2.9273 | emulator_hm | ok |
| Emulator CV-R2 median >= 0.60 | FAIL | 0.5570 | emulator_hm | emu_median_below_min |
| Emulator CV-R2 share>=0.30 >= 0.65 | PASS | 1.0000 | emulator_hm | ok |
| bad_run_pct_w2 <= 5% | PASS | 0.00% | abm | ok |
| Representative Employment_mean >= 45 | PASS | min=91.0333 | interpretability | ok |
| Representative Output_mean >= 50 | PASS | min=97.5970 | interpretability | ok |
| Representative Consumption_mean >= 50 | PASS | min=98.9246 | interpretability | ok |
| Representative BankFailed_share == 0 | PASS | max=0.000000 | abm | ok |
| Representative BalanceOK_share == 1 | PASS | min=1.000000 | abm | ok |
| SA components EV/OU/MD/CU present | PASS | components=['CU', 'EV', 'MD', 'OU'] | confirmatory | ok |
| SA has >=16 rows | PASS | rows=16 | confirmatory | ok |
| SA ranks present | PASS | rank_cols=True | confirmatory | ok |
| Structural: PriceDispersion median in [0.03,0.30] | PASS | median=0.0975 | interpretability | ok |
| Structural: |InventoryGap| p95 <= 1.5 | PASS | p95_abs=0.1413 | interpretability | ok |
| Structural: CreditRejectionRate mean <= 0.80 | PASS | mean=0.6832; p90=0.8573 | interpretability | ok |
| Confirmatory stability (|NROY delta| <= 5.0pp and SA top2 stable) | FAIL | nroy_delta_pp=15.31; sa_top2_stable=True | confirmatory | confirmatory_unstable |

## Wave-by-wave Convergence
```
 wave                                                                                                  wave_dir   n  steps  nroy_pct  Imax_median  Imax_p95  gpr_r2_cv_median  gpr_r2_cv_share_ge_threshold  shrink_mean  shrink_delta  wave_blocking_pass  stable_candidate
    1      C:\Users\Alexander\Desktop\Education\Aspa\code\output_research_core_v6_full_20260214_182812\01_wave1 350    220 35.380117     2.971244  4.851146          0.699152                           1.0     2.488867           NaN               False             False
    2      C:\Users\Alexander\Desktop\Education\Aspa\code\output_research_core_v6_full_20260214_182812\02_wave2 350    220 58.571429     2.494049  3.357796          0.647314                           1.0    13.910420     11.421553                True             False
    3 C:\Users\Alexander\Desktop\Education\Aspa\code\output_research_core_v6_full_20260214_182812\waves\wave_03 350    220 58.000000     2.482310  3.141448          0.699581                           1.0    23.787751      9.877331                True             False
    4 C:\Users\Alexander\Desktop\Education\Aspa\code\output_research_core_v6_full_20260214_182812\waves\wave_04 250    220 79.600000     2.337360  2.830318          0.516234                           1.0    31.982567      8.194816               False             False
    5 C:\Users\Alexander\Desktop\Education\Aspa\code\output_research_core_v6_full_20260214_182812\waves\wave_05 250    220 68.400000     2.429162  2.927263          0.557002                           1.0    39.483583      7.501016               False             False
```

- stopping_diagnostics: `{'stop_reason': 'max_waves_reached', 'final_wave': 5, 'waves_run': 5, 'wave_min': 2, 'wave_max': 5, 'adaptive_waves': True, 'stability_shrink_delta_pp_max': 5.0}`

## Economic Interpretability
- Representative points: `10`
- Employment_mean min: `91.0333`
- Output_mean min: `97.5970`
- Consumption_mean min: `98.9246`
- CreditRejections p90: `76.4883`

## Legacy Contour (reference)
| Check | Pass | Value |
|---|---|---|
| NROY wave2 in [25,60]% | FAIL | 68.40% |

