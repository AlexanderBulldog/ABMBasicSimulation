# Research Core Report

## Scientific Verdict v3 (blocking)
- overall_gate_pass_v3: `FAIL`
- legacy_gate_pass(reference): `PASS`

| Check | Pass | Value | Group | Reason |
|---|---|---|---|---|
| NROY wave2 in [25,65]% | PASS | 34.17% | emulator_hm | ok |
| I_max median wave2 < 3.0 | PASS | 2.7374 | emulator_hm | ok |
| I_max p95 wave2 < 4.5 | PASS | 3.0380 | emulator_hm | ok |
| Emulator CV-R2 median >= 0.60 | PASS | 0.7104 | emulator_hm | ok |
| Emulator CV-R2 share>=0.30 >= 0.65 | PASS | 1.0000 | emulator_hm | ok |
| bad_run_pct_w2 <= 5% | PASS | 0.00% | abm | ok |
| Representative Employment_mean >= 45 | PASS | min=81.9000 | interpretability | ok |
| Representative Output_mean >= 50 | PASS | min=91.5760 | interpretability | ok |
| Representative Consumption_mean >= 50 | PASS | min=99.8603 | interpretability | ok |
| Representative BankFailed_share == 0 | PASS | max=0.000000 | abm | ok |
| Representative BalanceOK_share == 1 | PASS | min=1.000000 | abm | ok |
| SA components EV/OU/MD/CU present | PASS | components=['CU', 'EV', 'MD', 'OU'] | confirmatory | ok |
| SA has >=16 rows | PASS | rows=16 | confirmatory | ok |
| SA ranks present | PASS | rank_cols=True | confirmatory | ok |
| Structural: PriceDispersion median in [0.03,0.30] | PASS | median=0.0934 | interpretability | ok |
| Structural: |InventoryGap| p95 <= 1.5 | PASS | p95_abs=0.4497 | interpretability | ok |
| Structural: CreditRejectionRate mean <= 0.80 | PASS | mean=0.6427; p90=0.8915 | interpretability | ok |
| Confirmatory stability (|NROY delta| <= 5.0pp and SA top2 stable) | FAIL | nroy_delta_pp=53.33; sa_top2_stable=False | confirmatory | confirmatory_unstable |

## Wave-by-wave Convergence
```
 wave                                                                                                                   wave_dir   n  steps  nroy_pct  Imax_median  Imax_p95  gpr_r2_cv_median  gpr_r2_cv_share_ge_threshold  shrink_mean  shrink_delta  wave_blocking_pass  stable_candidate
    1      C:\Users\Alexander\Desktop\Education\Aspa\code\output_evidence_campaign_v3_20260215\stage_a\A3\quick_confirm\01_wave1 120    120 21.839080     3.372382  5.147915          0.629603                           1.0     8.584727           NaN               False             False
    2 C:\Users\Alexander\Desktop\Education\Aspa\code\output_evidence_campaign_v3_20260215\stage_a\A3\quick_confirm\02_final_wave 120    120 56.666667     2.441844  3.895580          0.674948                           1.0    18.759526     10.174799                True             False
    3 C:\Users\Alexander\Desktop\Education\Aspa\code\output_evidence_campaign_v3_20260215\stage_a\A3\quick_confirm\waves\wave_03 120    120 34.166667     2.737405  3.038040          0.710397                           1.0    29.887762     11.128235                True             False
```

- stopping_diagnostics: `{'stop_reason': 'max_waves_reached', 'final_wave': 3, 'waves_run': 3, 'wave_min': 2, 'wave_max': 3, 'adaptive_waves': True, 'stability_shrink_delta_pp_max': 5.0}`

## Economic Interpretability
- Representative points: `2`
- Employment_mean min: `81.9000`
- Output_mean min: `91.5760`
- Consumption_mean min: `99.8603`
- CreditRejections p90: `98.4250`

## Legacy Contour (reference)
| Check | Pass | Value |
|---|---|---|
| NROY wave2 in [25,60]% | PASS | 34.17% |

