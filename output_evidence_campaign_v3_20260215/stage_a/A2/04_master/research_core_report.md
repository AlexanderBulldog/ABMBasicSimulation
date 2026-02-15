# Research Core Report

## Scientific Verdict v3 (blocking)
- overall_gate_pass_v3: `PASS`
- legacy_gate_pass(reference): `PASS`

| Check | Pass | Value | Group | Reason |
|---|---|---|---|---|
| NROY wave2 in [25,65]% | PASS | 29.38% | emulator_hm | ok |
| I_max median wave2 < 3.0 | PASS | 2.6052 | emulator_hm | ok |
| I_max p95 wave2 < 4.5 | PASS | 2.9643 | emulator_hm | ok |
| Emulator CV-R2 median >= 0.60 | PASS | 0.6325 | emulator_hm | ok |
| Emulator CV-R2 share>=0.30 >= 0.65 | PASS | 1.0000 | emulator_hm | ok |
| bad_run_pct_w2 <= 5% | PASS | 0.00% | abm | ok |
| Representative Employment_mean >= 45 | PASS | min=91.6750 | interpretability | ok |
| Representative Output_mean >= 50 | PASS | min=109.3655 | interpretability | ok |
| Representative Consumption_mean >= 50 | PASS | min=102.2253 | interpretability | ok |
| Representative BankFailed_share == 0 | PASS | max=0.000000 | abm | ok |
| Representative BalanceOK_share == 1 | PASS | min=1.000000 | abm | ok |
| SA components EV/OU/MD/CU present | PASS | components=['CU', 'EV', 'MD', 'OU'] | confirmatory | ok |
| SA has >=16 rows | PASS | rows=16 | confirmatory | ok |
| SA ranks present | PASS | rank_cols=True | confirmatory | ok |
| Structural: PriceDispersion median in [0.03,0.30] | PASS | median=0.0918 | interpretability | ok |
| Structural: |InventoryGap| p95 <= 1.5 | PASS | p95_abs=0.3875 | interpretability | ok |
| Structural: CreditRejectionRate mean <= 0.80 | PASS | mean=0.6753; p90=0.9500 | interpretability | ok |

## Wave-by-wave Convergence
```
 wave                                                                                                     wave_dir   n  steps  nroy_pct  Imax_median  Imax_p95  gpr_r2_cv_median  gpr_r2_cv_share_ge_threshold  shrink_mean  shrink_delta  wave_blocking_pass  stable_candidate
    1      C:\Users\Alexander\Desktop\Education\Aspa\code\output_evidence_campaign_v3_20260215\stage_a\A2\01_wave1 160    120 26.526316     3.193056  4.761610          0.634431                           1.0     4.546163           NaN               False             False
    2 C:\Users\Alexander\Desktop\Education\Aspa\code\output_evidence_campaign_v3_20260215\stage_a\A2\02_final_wave 160    120 50.625000     2.441957  3.999279          0.575764                           1.0    16.344885     11.798722               False             False
    3 C:\Users\Alexander\Desktop\Education\Aspa\code\output_evidence_campaign_v3_20260215\stage_a\A2\waves\wave_03 160    120 29.375000     2.605215  2.964286          0.632497                           1.0    28.049293     11.704408                True             False
```

- stopping_diagnostics: `{'stop_reason': 'max_waves_reached', 'final_wave': 3, 'waves_run': 3, 'wave_min': 2, 'wave_max': 3, 'adaptive_waves': True, 'stability_shrink_delta_pp_max': 5.0}`

## Economic Interpretability
- Representative points: `4`
- Employment_mean min: `91.6750`
- Output_mean min: `109.3655`
- Consumption_mean min: `102.2253`
- CreditRejections p90: `72.6650`

## Legacy Contour (reference)
| Check | Pass | Value |
|---|---|---|
| NROY wave2 in [25,60]% | PASS | 29.38% |

