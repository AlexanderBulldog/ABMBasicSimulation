# Research Core Report

## Scientific Verdict v3 (blocking)
- overall_gate_pass_v3: `PASS`
- legacy_gate_pass(reference): `FAIL`

| Check | Pass | Value | Group | Reason |
|---|---|---|---|---|
| NROY wave2 in [25,65]% | PASS | 64.17% | emulator_hm | ok |
| I_max median wave2 < 3.0 | PASS | 2.3923 | emulator_hm | ok |
| I_max p95 wave2 < 4.5 | PASS | 3.3477 | emulator_hm | ok |
| Emulator CV-R2 median >= 0.60 | PASS | 0.6586 | emulator_hm | ok |
| Emulator CV-R2 share>=0.30 >= 0.65 | PASS | 1.0000 | emulator_hm | ok |
| bad_run_pct_w2 <= 5% | PASS | 0.00% | abm | ok |
| Representative Employment_mean >= 45 | PASS | min=88.1000 | interpretability | ok |
| Representative Output_mean >= 50 | PASS | min=98.6049 | interpretability | ok |
| Representative Consumption_mean >= 50 | PASS | min=99.3505 | interpretability | ok |
| Representative BankFailed_share == 0 | PASS | max=0.000000 | abm | ok |
| Representative BalanceOK_share == 1 | PASS | min=1.000000 | abm | ok |
| SA components EV/OU/MD/CU present | PASS | components=['CU', 'EV', 'MD', 'OU'] | confirmatory | ok |
| SA has >=16 rows | PASS | rows=16 | confirmatory | ok |
| SA ranks present | PASS | rank_cols=True | confirmatory | ok |
| Structural: PriceDispersion median in [0.03,0.30] | PASS | median=0.0934 | interpretability | ok |
| Structural: |InventoryGap| p95 <= 1.5 | PASS | p95_abs=0.2789 | interpretability | ok |
| Structural: CreditRejectionRate mean <= 0.80 | PASS | mean=0.6664; p90=0.8342 | interpretability | ok |
| Confirmatory stability (|NROY delta| <= 5.0pp and SA top2 stable) | PASS | nroy_delta_pp=-1.67; sa_top2_stable=True | confirmatory | ok |

## Wave-by-wave Convergence
```
 wave                                                                                                                   wave_dir   n  steps  nroy_pct  Imax_median  Imax_p95  gpr_r2_cv_median  gpr_r2_cv_share_ge_threshold  shrink_mean  shrink_delta  wave_blocking_pass  stable_candidate
    1      C:\Users\Alexander\Desktop\Education\Aspa\code\output_evidence_campaign_v3_20260215\stage_a\A4\quick_confirm\01_wave1 120    120 19.252874     3.372382  5.147915          0.629603                           1.0     8.834371           NaN               False             False
    2 C:\Users\Alexander\Desktop\Education\Aspa\code\output_evidence_campaign_v3_20260215\stage_a\A4\quick_confirm\02_final_wave 120    120 29.166667     2.837670  3.650775          0.656429                           1.0    20.973350     12.138980                True             False
    3 C:\Users\Alexander\Desktop\Education\Aspa\code\output_evidence_campaign_v3_20260215\stage_a\A4\quick_confirm\waves\wave_03 120    120 64.166667     2.392306  3.347724          0.658628                           1.0    28.748736      7.775385                True             False
```

- stopping_diagnostics: `{'stop_reason': 'max_waves_reached', 'final_wave': 3, 'waves_run': 3, 'wave_min': 2, 'wave_max': 3, 'adaptive_waves': True, 'stability_shrink_delta_pp_max': 5.0}`

## Economic Interpretability
- Representative points: `2`
- Employment_mean min: `88.1000`
- Output_mean min: `98.6049`
- Consumption_mean min: `99.3505`
- CreditRejections p90: `79.1083`

## Legacy Contour (reference)
| Check | Pass | Value |
|---|---|---|
| NROY wave2 in [25,60]% | FAIL | 64.17% |

