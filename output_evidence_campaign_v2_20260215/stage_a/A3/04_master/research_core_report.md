# Research Core Report

## Scientific Verdict v3 (blocking)
- overall_gate_pass_v3: `PASS`
- legacy_gate_pass(reference): `PASS`

| Check | Pass | Value | Group | Reason |
|---|---|---|---|---|
| NROY wave2 in [25,65]% | PASS | 55.00% | emulator_hm | ok |
| I_max median wave2 < 3.0 | PASS | 2.5747 | emulator_hm | ok |
| I_max p95 wave2 < 4.5 | PASS | 3.0311 | emulator_hm | ok |
| Emulator CV-R2 median >= 0.60 | PASS | 0.6374 | emulator_hm | ok |
| Emulator CV-R2 share>=0.30 >= 0.65 | PASS | 1.0000 | emulator_hm | ok |
| bad_run_pct_w2 <= 5% | PASS | 0.00% | abm | ok |
| Representative Employment_mean >= 45 | PASS | min=72.5500 | interpretability | ok |
| Representative Output_mean >= 50 | PASS | min=77.5112 | interpretability | ok |
| Representative Consumption_mean >= 50 | PASS | min=92.2157 | interpretability | ok |
| Representative BankFailed_share == 0 | PASS | max=0.000000 | abm | ok |
| Representative BalanceOK_share == 1 | PASS | min=1.000000 | abm | ok |
| SA components EV/OU/MD/CU present | PASS | components=['CU', 'EV', 'MD', 'OU'] | confirmatory | ok |
| SA has >=16 rows | PASS | rows=16 | confirmatory | ok |
| SA ranks present | PASS | rank_cols=True | confirmatory | ok |
| Structural: PriceDispersion median in [0.03,0.30] | PASS | median=0.0952 | interpretability | ok |
| Structural: |InventoryGap| p95 <= 1.5 | PASS | p95_abs=0.3954 | interpretability | ok |
| Structural: CreditRejectionRate mean <= 0.80 | PASS | mean=0.6915; p90=1.0000 | interpretability | ok |

## Wave-by-wave Convergence
```
 wave                                                                                                     wave_dir   n  steps  nroy_pct  Imax_median  Imax_p95  gpr_r2_cv_median  gpr_r2_cv_share_ge_threshold  shrink_mean  shrink_delta  wave_blocking_pass  stable_candidate
    1      C:\Users\Alexander\Desktop\Education\Aspa\code\output_evidence_campaign_v2_20260215\stage_a\A3\01_wave1 160    120 31.578947     3.193056  4.761610          0.634431                           1.0     4.097365           NaN               False             False
    2 C:\Users\Alexander\Desktop\Education\Aspa\code\output_evidence_campaign_v2_20260215\stage_a\A3\02_final_wave 160    120 49.375000     2.601068  3.674644          0.668088                           1.0    14.956449     10.859084                True             False
    3 C:\Users\Alexander\Desktop\Education\Aspa\code\output_evidence_campaign_v2_20260215\stage_a\A3\waves\wave_03 160    120 55.000000     2.574730  3.031066          0.637437                           1.0    24.972726     10.016277                True             False
```

- stopping_diagnostics: `{'stop_reason': 'max_waves_reached', 'final_wave': 3, 'waves_run': 3, 'wave_min': 2, 'wave_max': 3, 'adaptive_waves': True, 'stability_shrink_delta_pp_max': 5.0}`

## Economic Interpretability
- Representative points: `4`
- Employment_mean min: `72.5500`
- Output_mean min: `77.5112`
- Consumption_mean min: `92.2157`
- CreditRejections p90: `72.9200`

## Legacy Contour (reference)
| Check | Pass | Value |
|---|---|---|
| NROY wave2 in [25,60]% | PASS | 55.00% |

