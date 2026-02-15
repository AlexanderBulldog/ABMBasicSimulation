# Research Core Report

## Scientific Verdict v3 (blocking)
- overall_gate_pass_v3: `FAIL`
- legacy_gate_pass(reference): `PASS`

| Check | Pass | Value | Group | Reason |
|---|---|---|---|---|
| NROY wave2 in [25,65]% | PASS | 44.57% | emulator_hm | ok |
| I_max median wave2 < 3.0 | PASS | 2.6064 | emulator_hm | ok |
| I_max p95 wave2 < 4.5 | PASS | 3.2574 | emulator_hm | ok |
| Emulator CV-R2 median >= 0.60 | PASS | 0.6878 | emulator_hm | ok |
| Emulator CV-R2 share>=0.30 >= 0.65 | PASS | 1.0000 | emulator_hm | ok |
| bad_run_pct_w2 <= 5% | PASS | 0.00% | abm | ok |
| Representative Employment_mean >= 45 | FAIL | min=23.1500 | interpretability | employment_below_floor |
| Representative Output_mean >= 50 | FAIL | min=23.7580 | interpretability | output_below_floor |
| Representative Consumption_mean >= 50 | FAIL | min=26.8851 | interpretability | consumption_below_floor |
| Representative BankFailed_share == 0 | PASS | max=0.000000 | abm | ok |
| Representative BalanceOK_share == 1 | PASS | min=1.000000 | abm | ok |
| SA components EV/OU/MD/CU present | PASS | components=['CU', 'EV', 'MD', 'OU'] | confirmatory | ok |
| SA has >=16 rows | PASS | rows=16 | confirmatory | ok |
| SA ranks present | PASS | rank_cols=True | confirmatory | ok |
| Structural: PriceDispersion median in [0.03,0.30] | PASS | median=0.0946 | interpretability | ok |
| Structural: |InventoryGap| p95 <= 1.5 | PASS | p95_abs=0.1749 | interpretability | ok |
| Structural: CreditRejectionRate mean <= 0.80 | PASS | mean=0.6492; p90=0.8462 | interpretability | ok |
| Confirmatory stability (|NROY delta| <= 5.0pp and SA top2 stable) | FAIL | nroy_delta_pp=31.43; sa_top2_stable=True | confirmatory | confirmatory_unstable |

## Wave-by-wave Convergence
```
 wave                                                                                                     wave_dir   n  steps  nroy_pct  Imax_median  Imax_p95  gpr_r2_cv_median  gpr_r2_cv_share_ge_threshold  shrink_mean  shrink_delta  wave_blocking_pass  stable_candidate
    1      C:\Users\Alexander\Desktop\Education\Aspa\code\output_evidence_campaign_v3_20260215\stage_b\R2\01_wave1 350    220 28.205128     3.204011  4.953704          0.672430                           1.0     3.610090           NaN               False             False
    2 C:\Users\Alexander\Desktop\Education\Aspa\code\output_evidence_campaign_v3_20260215\stage_b\R2\02_final_wave 350    220 78.000000     2.168148  3.311989          0.563711                           1.0    13.190548      9.580458               False             False
    3 C:\Users\Alexander\Desktop\Education\Aspa\code\output_evidence_campaign_v3_20260215\stage_b\R2\waves\wave_03 350    220 44.571429     2.606412  3.257367          0.687822                           1.0    23.422257     10.231708                True             False
```

- stopping_diagnostics: `{'stop_reason': 'max_waves_reached', 'final_wave': 3, 'waves_run': 3, 'wave_min': 2, 'wave_max': 3, 'adaptive_waves': True, 'stability_shrink_delta_pp_max': 5.0}`

## Economic Interpretability
- Representative points: `10`
- Employment_mean min: `23.1500`
- Output_mean min: `23.7580`
- Consumption_mean min: `26.8851`
- CreditRejections p90: `78.1650`

## Legacy Contour (reference)
| Check | Pass | Value |
|---|---|---|
| NROY wave2 in [25,60]% | PASS | 44.57% |

