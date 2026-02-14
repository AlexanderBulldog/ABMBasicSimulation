# Research Core Report

## Scientific Verdict v3 (blocking)
- overall_gate_pass_v3: `FAIL`
- legacy_gate_pass(reference): `PASS`

| Check | Pass | Value | Group | Reason |
|---|---|---|---|---|
| NROY wave2 in [25,65]% | PASS | 57.60% | emulator_hm | ok |
| I_max median wave2 < 3.0 | PASS | 2.5866 | emulator_hm | ok |
| I_max p95 wave2 < 4.5 | PASS | 2.8018 | emulator_hm | ok |
| Emulator CV-R2 median >= 0.60 | FAIL | 0.5630 | emulator_hm | emu_median_below_min |
| Emulator CV-R2 share>=0.30 >= 0.65 | PASS | 0.8889 | emulator_hm | ok |
| bad_run_pct_w2 <= 5% | PASS | 0.00% | abm | ok |
| Representative Employment_mean >= 45 | PASS | min=94.2250 | interpretability | ok |
| Representative Output_mean >= 50 | PASS | min=113.4317 | interpretability | ok |
| Representative Consumption_mean >= 50 | PASS | min=106.8787 | interpretability | ok |
| Representative BankFailed_share == 0 | PASS | max=0.000000 | abm | ok |
| Representative BalanceOK_share == 1 | PASS | min=1.000000 | abm | ok |
| SA components EV/OU/MD/CU present | PASS | components=['CU', 'EV', 'MD', 'OU'] | confirmatory | ok |
| SA has >=16 rows | PASS | rows=16 | confirmatory | ok |
| SA ranks present | PASS | rank_cols=True | confirmatory | ok |
| Structural: PriceDispersion median in [0.03,0.30] | PASS | median=0.0822 | interpretability | ok |
| Structural: |InventoryGap| p95 <= 1.5 | PASS | p95_abs=0.3292 | interpretability | ok |
| Structural: CreditRejections p90 <= 30 | FAIL | p90=104.8050 | interpretability | credit_rejections_too_high |

## Wave-by-wave Convergence
```
 wave                                                                             wave_dir   n  steps  nroy_pct  Imax_median  Imax_p95  gpr_r2_cv_median  gpr_r2_cv_share_ge_threshold  shrink_mean  shrink_delta  wave_blocking_pass  stable_candidate
    1      C:\Users\Alexander\Desktop\Education\Aspa\code\output_research_core_v3\01_wave1 120    120 89.714286     1.569378  3.526953          0.688121                      0.888889     1.122465           NaN               False             False
    2      C:\Users\Alexander\Desktop\Education\Aspa\code\output_research_core_v3\02_wave2 120    120 64.166667     2.292237  3.840950          0.690844                      0.851852    13.012187     11.889722                True             False
    3 C:\Users\Alexander\Desktop\Education\Aspa\code\output_research_core_v3\waves\wave_03 120    120 48.333333     2.614556  3.416940          0.649515                      0.888889    23.445183     10.432996                True             False
    4 C:\Users\Alexander\Desktop\Education\Aspa\code\output_research_core_v3\waves\wave_04 250    120 66.800000     2.561396  3.095655          0.586145                      0.888889    31.489549      8.044366               False             False
    5 C:\Users\Alexander\Desktop\Education\Aspa\code\output_research_core_v3\waves\wave_05 250    120 57.600000     2.586551  2.801805          0.563022                      0.888889    38.896227      7.406678               False             False
```

- stopping_diagnostics: `{'stop_reason': 'max_waves_reached', 'final_wave': 5, 'waves_run': 5, 'wave_min': 2, 'wave_max': 5, 'adaptive_waves': True, 'stability_shrink_delta_pp_max': 5.0}`

## Economic Interpretability
- Representative points: `4`
- Employment_mean min: `94.2250`
- Output_mean min: `113.4317`
- Consumption_mean min: `106.8787`
- CreditRejections p90: `101.9600`

## Legacy Contour (reference)
| Check | Pass | Value |
|---|---|---|
| NROY wave2 in [25,60]% | PASS | 57.60% |

