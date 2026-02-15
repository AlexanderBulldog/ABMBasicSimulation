# Research Core Report

## Scientific Verdict v3 (blocking)
- overall_gate_pass_v3: `PASS`
- legacy_gate_pass(reference): `PASS`

| Check | Pass | Value | Group | Reason |
|---|---|---|---|---|
| NROY wave2 in [25,65]% | PASS | 57.50% | emulator_hm | ok |
| I_max median wave2 < 3.0 | PASS | 2.4903 | emulator_hm | ok |
| I_max p95 wave2 < 4.5 | PASS | 3.2611 | emulator_hm | ok |
| Emulator CV-R2 median >= 0.60 | PASS | 0.7115 | emulator_hm | ok |
| Emulator CV-R2 share>=0.30 >= 0.65 | PASS | 1.0000 | emulator_hm | ok |
| bad_run_pct_w2 <= 5% | PASS | 0.00% | abm | ok |
| Representative Employment_mean >= 45 | PASS | min=93.7500 | interpretability | ok |
| Representative Output_mean >= 50 | PASS | min=105.5236 | interpretability | ok |
| Representative Consumption_mean >= 50 | PASS | min=99.9338 | interpretability | ok |
| Representative BankFailed_share == 0 | PASS | max=0.000000 | abm | ok |
| Representative BalanceOK_share == 1 | PASS | min=1.000000 | abm | ok |
| SA components EV/OU/MD/CU present | PASS | components=['CU', 'EV', 'MD', 'OU'] | confirmatory | ok |
| SA has >=16 rows | PASS | rows=16 | confirmatory | ok |
| SA ranks present | PASS | rank_cols=True | confirmatory | ok |
| Structural: PriceDispersion median in [0.03,0.30] | PASS | median=0.0846 | interpretability | ok |
| Structural: |InventoryGap| p95 <= 1.5 | PASS | p95_abs=0.3586 | interpretability | ok |
| Structural: CreditRejectionRate mean <= 0.80 | PASS | mean=0.6801; p90=0.9289 | interpretability | ok |

## Wave-by-wave Convergence
```
 wave                                                                                           wave_dir  n  steps  nroy_pct  Imax_median  Imax_p95  gpr_r2_cv_median  gpr_r2_cv_share_ge_threshold  shrink_mean  shrink_delta  wave_blocking_pass  stable_candidate
    1      C:\Users\Alexander\Desktop\Education\Aspa\code\output_evidence_campaign_v2_sanity_A1\01_wave1 80    120 30.252101     3.333952  5.391692          0.695460                           1.0     6.366416           NaN               False             False
    2 C:\Users\Alexander\Desktop\Education\Aspa\code\output_evidence_campaign_v2_sanity_A1\02_final_wave 80    120 57.500000     2.489406  3.710108          0.649262                           1.0    19.900070     13.533655                True             False
    3 C:\Users\Alexander\Desktop\Education\Aspa\code\output_evidence_campaign_v2_sanity_A1\waves\wave_03 80    120 57.500000     2.490288  3.261096          0.711480                           1.0    29.419908      9.519838                True             False
```

- stopping_diagnostics: `{'stop_reason': 'max_waves_reached', 'final_wave': 3, 'waves_run': 3, 'wave_min': 2, 'wave_max': 3, 'adaptive_waves': True, 'stability_shrink_delta_pp_max': 5.0}`

## Economic Interpretability
- Representative points: `4`
- Employment_mean min: `93.7500`
- Output_mean min: `105.5236`
- Consumption_mean min: `99.9338`
- CreditRejections p90: `78.0850`

## Legacy Contour (reference)
| Check | Pass | Value |
|---|---|---|
| NROY wave2 in [25,60]% | PASS | 57.50% |

