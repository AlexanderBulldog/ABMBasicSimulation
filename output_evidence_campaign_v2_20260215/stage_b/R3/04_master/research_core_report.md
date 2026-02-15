# Research Core Report

## Scientific Verdict v3 (blocking)
- overall_gate_pass_v3: `FAIL`
- legacy_gate_pass(reference): `PASS`

| Check | Pass | Value | Group | Reason |
|---|---|---|---|---|
| NROY wave2 in [25,65]% | PASS | 30.86% | emulator_hm | ok |
| I_max median wave2 < 3.0 | PASS | 2.7077 | emulator_hm | ok |
| I_max p95 wave2 < 4.5 | PASS | 2.9020 | emulator_hm | ok |
| Emulator CV-R2 median >= 0.60 | PASS | 0.6622 | emulator_hm | ok |
| Emulator CV-R2 share>=0.30 >= 0.65 | PASS | 1.0000 | emulator_hm | ok |
| bad_run_pct_w2 <= 5% | PASS | 0.00% | abm | ok |
| Representative Employment_mean >= 45 | PASS | min=78.1167 | interpretability | ok |
| Representative Output_mean >= 50 | PASS | min=91.5532 | interpretability | ok |
| Representative Consumption_mean >= 50 | PASS | min=81.1176 | interpretability | ok |
| Representative BankFailed_share == 0 | PASS | max=0.000000 | abm | ok |
| Representative BalanceOK_share == 1 | PASS | min=1.000000 | abm | ok |
| SA components EV/OU/MD/CU present | PASS | components=['CU', 'EV', 'MD', 'OU'] | confirmatory | ok |
| SA has >=16 rows | PASS | rows=16 | confirmatory | ok |
| SA ranks present | PASS | rank_cols=True | confirmatory | ok |
| Structural: PriceDispersion median in [0.03,0.30] | PASS | median=0.0936 | interpretability | ok |
| Structural: |InventoryGap| p95 <= 1.5 | PASS | p95_abs=0.1575 | interpretability | ok |
| Structural: CreditRejectionRate mean <= 0.80 | PASS | mean=0.6582; p90=0.8299 | interpretability | ok |
| Confirmatory stability (|NROY delta| <= 5.0pp and SA top2 stable) | FAIL | nroy_delta_pp=68.57; sa_top2_stable=False | confirmatory | confirmatory_unstable |

## Wave-by-wave Convergence
```
 wave                                                                                                     wave_dir   n  steps  nroy_pct  Imax_median  Imax_p95  gpr_r2_cv_median  gpr_r2_cv_share_ge_threshold  shrink_mean  shrink_delta  wave_blocking_pass  stable_candidate
    1      C:\Users\Alexander\Desktop\Education\Aspa\code\output_evidence_campaign_v2_20260215\stage_b\R3\01_wave1 350    220 24.461840     3.368218  5.196998          0.711035                           1.0     3.765507           NaN               False             False
    2 C:\Users\Alexander\Desktop\Education\Aspa\code\output_evidence_campaign_v2_20260215\stage_b\R3\02_final_wave 350    220 28.000000     2.713029  2.915132          0.657727                           1.0    14.937999     11.172492                True             False
    3 C:\Users\Alexander\Desktop\Education\Aspa\code\output_evidence_campaign_v2_20260215\stage_b\R3\waves\wave_03 350    220 30.857143     2.707695  2.901996          0.662223                           1.0    24.875199      9.937201                True             False
```

- stopping_diagnostics: `{'stop_reason': 'max_waves_reached', 'final_wave': 3, 'waves_run': 3, 'wave_min': 2, 'wave_max': 3, 'adaptive_waves': True, 'stability_shrink_delta_pp_max': 5.0}`

## Economic Interpretability
- Representative points: `10`
- Employment_mean min: `78.1167`
- Output_mean min: `91.5532`
- Consumption_mean min: `81.1176`
- CreditRejections p90: `76.9383`

## Legacy Contour (reference)
| Check | Pass | Value |
|---|---|---|
| NROY wave2 in [25,60]% | PASS | 30.86% |

