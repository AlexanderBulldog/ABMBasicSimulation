# Research Core Report

## Scientific Verdict v3 (blocking)
- overall_gate_pass_v3: `FAIL`
- legacy_gate_pass(reference): `PASS`

| Check | Pass | Value | Group | Reason |
|---|---|---|---|---|
| NROY wave2 in [25,65]% | PASS | 35.71% | emulator_hm | ok |
| I_max median wave2 < 3.0 | PASS | 2.6172 | emulator_hm | ok |
| I_max p95 wave2 < 4.5 | PASS | 2.8196 | emulator_hm | ok |
| Emulator CV-R2 median >= 0.60 | PASS | 0.6070 | emulator_hm | ok |
| Emulator CV-R2 share>=0.30 >= 0.65 | PASS | 1.0000 | emulator_hm | ok |
| bad_run_pct_w2 <= 5% | PASS | 0.00% | abm | ok |
| Representative Employment_mean >= 45 | FAIL | min=4.0167 | interpretability | employment_below_floor |
| Representative Output_mean >= 50 | FAIL | min=5.0267 | interpretability | output_below_floor |
| Representative Consumption_mean >= 50 | FAIL | min=9.5488 | interpretability | consumption_below_floor |
| Representative BankFailed_share == 0 | PASS | max=0.000000 | abm | ok |
| Representative BalanceOK_share == 1 | PASS | min=1.000000 | abm | ok |
| SA components EV/OU/MD/CU present | PASS | components=['CU', 'EV', 'MD', 'OU'] | confirmatory | ok |
| SA has >=16 rows | PASS | rows=16 | confirmatory | ok |
| SA ranks present | PASS | rank_cols=True | confirmatory | ok |
| Structural: PriceDispersion median in [0.03,0.30] | PASS | median=0.0932 | interpretability | ok |
| Structural: |InventoryGap| p95 <= 1.5 | PASS | p95_abs=0.1499 | interpretability | ok |
| Structural: CreditRejectionRate mean <= 0.80 | PASS | mean=0.6676; p90=0.8400 | interpretability | ok |
| Confirmatory stability (|NROY delta| <= 5.0pp and SA top2 stable) | FAIL | nroy_delta_pp=64.00; sa_top2_stable=False | confirmatory | confirmatory_unstable |

## Wave-by-wave Convergence
```
 wave                                                                                                     wave_dir   n  steps  nroy_pct  Imax_median  Imax_p95  gpr_r2_cv_median  gpr_r2_cv_share_ge_threshold  shrink_mean  shrink_delta  wave_blocking_pass  stable_candidate
    1      C:\Users\Alexander\Desktop\Education\Aspa\code\output_evidence_campaign_v3_20260215\stage_b\R3\01_wave1 350    220 23.581213     3.368218  5.196998          0.711035                           1.0     3.776914           NaN               False             False
    2 C:\Users\Alexander\Desktop\Education\Aspa\code\output_evidence_campaign_v3_20260215\stage_b\R3\02_final_wave 350    220 25.929457     2.673258  3.013719          0.597600                           1.0    15.536846     11.759932               False             False
    3 C:\Users\Alexander\Desktop\Education\Aspa\code\output_evidence_campaign_v3_20260215\stage_b\R3\waves\wave_03 350    220 35.714286     2.617178  2.819606          0.606962                           1.0    24.585581      9.048736                True             False
```

- stopping_diagnostics: `{'stop_reason': 'max_waves_reached', 'final_wave': 3, 'waves_run': 3, 'wave_min': 2, 'wave_max': 3, 'adaptive_waves': True, 'stability_shrink_delta_pp_max': 5.0}`

## Economic Interpretability
- Representative points: `10`
- Employment_mean min: `4.0167`
- Output_mean min: `5.0267`
- Consumption_mean min: `9.5488`
- CreditRejections p90: `85.7150`

## Legacy Contour (reference)
| Check | Pass | Value |
|---|---|---|
| NROY wave2 in [25,60]% | PASS | 35.71% |

