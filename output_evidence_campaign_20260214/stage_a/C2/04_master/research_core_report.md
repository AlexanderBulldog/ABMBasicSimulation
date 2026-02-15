# Research Core Report

## Scientific Verdict v3 (blocking)
- overall_gate_pass_v3: `FAIL`
- legacy_gate_pass(reference): `FAIL`

| Check | Pass | Value | Group | Reason |
|---|---|---|---|---|
| NROY wave2 in [25,65]% | FAIL | 74.17% | emulator_hm | nroy_out_of_range |
| I_max median wave2 < 3.0 | PASS | 2.4720 | emulator_hm | ok |
| I_max p95 wave2 < 4.5 | PASS | 3.3387 | emulator_hm | ok |
| Emulator CV-R2 median >= 0.60 | PASS | 0.6468 | emulator_hm | ok |
| Emulator CV-R2 share>=0.30 >= 0.65 | PASS | 1.0000 | emulator_hm | ok |
| bad_run_pct_w2 <= 5% | PASS | 0.00% | abm | ok |
| Representative Employment_mean >= 45 | PASS | min=92.9750 | interpretability | ok |
| Representative Output_mean >= 50 | PASS | min=104.1237 | interpretability | ok |
| Representative Consumption_mean >= 50 | PASS | min=102.0384 | interpretability | ok |
| Representative BankFailed_share == 0 | PASS | max=0.000000 | abm | ok |
| Representative BalanceOK_share == 1 | PASS | min=1.000000 | abm | ok |
| SA components EV/OU/MD/CU present | PASS | components=['CU', 'EV', 'MD', 'OU'] | confirmatory | ok |
| SA has >=16 rows | PASS | rows=16 | confirmatory | ok |
| SA ranks present | PASS | rank_cols=True | confirmatory | ok |
| Structural: PriceDispersion median in [0.03,0.30] | PASS | median=0.0903 | interpretability | ok |
| Structural: |InventoryGap| p95 <= 1.5 | PASS | p95_abs=0.3115 | interpretability | ok |
| Structural: CreditRejectionRate mean <= 0.80 | PASS | mean=0.6711; p90=0.9264 | interpretability | ok |

## Wave-by-wave Convergence
```
 wave                                                                                                  wave_dir   n  steps  nroy_pct  Imax_median  Imax_p95  gpr_r2_cv_median  gpr_r2_cv_share_ge_threshold  shrink_mean  shrink_delta  wave_blocking_pass  stable_candidate
    1      C:\Users\Alexander\Desktop\Education\Aspa\code\output_evidence_campaign_20260214\stage_a\C2\01_wave1 120    120 31.321839     3.288143  4.906567          0.690669                           1.0     5.728032           NaN               False             False
    2 C:\Users\Alexander\Desktop\Education\Aspa\code\output_evidence_campaign_20260214\stage_a\C2\02_final_wave 120    120 32.500000     2.843907  3.553854          0.680179                           1.0    18.857170     13.129137                True             False
    3 C:\Users\Alexander\Desktop\Education\Aspa\code\output_evidence_campaign_20260214\stage_a\C2\waves\wave_03 120    120 74.166667     2.471993  3.338714          0.646835                           1.0    28.039394      9.182224               False             False
```

- stopping_diagnostics: `{'stop_reason': 'max_waves_reached', 'final_wave': 3, 'waves_run': 3, 'wave_min': 2, 'wave_max': 3, 'adaptive_waves': True, 'stability_shrink_delta_pp_max': 5.0}`

## Economic Interpretability
- Representative points: `4`
- Employment_mean min: `92.9750`
- Output_mean min: `104.1237`
- Consumption_mean min: `102.0384`
- CreditRejections p90: `77.7775`

## Legacy Contour (reference)
| Check | Pass | Value |
|---|---|---|
| NROY wave2 in [25,60]% | FAIL | 74.17% |

