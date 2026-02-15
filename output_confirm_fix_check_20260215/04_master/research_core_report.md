# Research Core Report

## Scientific Verdict v3 (blocking)
- overall_gate_pass_v3: `FAIL`
- legacy_gate_pass(reference): `PASS`

| Check | Pass | Value | Group | Reason |
|---|---|---|---|---|
| NROY wave2 in [25,65]% | PASS | 54.09% | emulator_hm | ok |
| I_max median wave2 < 3.0 | PASS | 2.5872 | emulator_hm | ok |
| I_max p95 wave2 < 4.5 | PASS | 2.9387 | emulator_hm | ok |
| Emulator CV-R2 median >= 0.60 | FAIL | 0.5990 | emulator_hm | emu_median_below_min |
| Emulator CV-R2 share>=0.30 >= 0.65 | PASS | 1.0000 | emulator_hm | ok |
| bad_run_pct_w2 <= 5% | PASS | 0.00% | abm | ok |
| Representative Employment_mean >= 45 | FAIL | min=11.3833 | interpretability | employment_below_floor |
| Representative Output_mean >= 50 | FAIL | min=16.0439 | interpretability | output_below_floor |
| Representative Consumption_mean >= 50 | FAIL | min=15.7961 | interpretability | consumption_below_floor |
| Representative BankFailed_share == 0 | PASS | max=0.000000 | abm | ok |
| Representative BalanceOK_share == 1 | PASS | min=1.000000 | abm | ok |
| SA components EV/OU/MD/CU present | PASS | components=['CU', 'EV', 'MD', 'OU'] | confirmatory | ok |
| SA has >=16 rows | PASS | rows=16 | confirmatory | ok |
| SA ranks present | PASS | rank_cols=True | confirmatory | ok |
| Structural: PriceDispersion median in [0.03,0.30] | PASS | median=0.0863 | interpretability | ok |
| Structural: |InventoryGap| p95 <= 1.5 | PASS | p95_abs=0.2292 | interpretability | ok |
| Structural: CreditRejectionRate mean <= 0.80 | PASS | mean=0.6773; p90=0.8788 | interpretability | ok |
| Confirmatory stability (|NROY delta| <= 5.0pp and SA top2 stable) | FAIL | nroy_delta_pp=20.91; sa_top2_stable=False | confirmatory | confirmatory_unstable |

## Wave-by-wave Convergence
```
 wave                                                                                       wave_dir   n  steps  nroy_pct  Imax_median  Imax_p95  gpr_r2_cv_median  gpr_r2_cv_share_ge_threshold  shrink_mean  shrink_delta  wave_blocking_pass  stable_candidate
    1      C:\Users\Alexander\Desktop\Education\Aspa\code\output_confirm_fix_check_20260215\01_wave1 220    220 13.798450     3.415101  5.205762          0.765513                           1.0     6.630440           NaN               False             False
    2 C:\Users\Alexander\Desktop\Education\Aspa\code\output_confirm_fix_check_20260215\02_final_wave 220    220 28.614916     2.813808  4.086864          0.646475                           1.0    16.134897      9.504458                True             False
    3 C:\Users\Alexander\Desktop\Education\Aspa\code\output_confirm_fix_check_20260215\waves\wave_03 220    220 54.090909     2.587176  2.938687          0.598964                           1.0    25.940721      9.805824               False             False
```

- stopping_diagnostics: `{'stop_reason': 'max_waves_reached', 'final_wave': 3, 'waves_run': 3, 'wave_min': 2, 'wave_max': 3, 'adaptive_waves': True, 'stability_shrink_delta_pp_max': 5.0}`

## Economic Interpretability
- Representative points: `6`
- Employment_mean min: `11.3833`
- Output_mean min: `16.0439`
- Consumption_mean min: `15.7961`
- CreditRejections p90: `82.8017`

## Legacy Contour (reference)
| Check | Pass | Value |
|---|---|---|
| NROY wave2 in [25,60]% | PASS | 54.09% |

