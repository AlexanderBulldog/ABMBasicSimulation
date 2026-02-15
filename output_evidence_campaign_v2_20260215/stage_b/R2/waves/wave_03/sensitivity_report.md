# Uncertainty Sensitivity Report (2026-02-15)

## 1) Setup
- Dataset: `C:\Users\Alexander\Desktop\Education\Aspa\code\output_evidence_campaign_v2_20260215\stage_b\R2\waves\wave_03\lhs_runs_wave3.csv`
- Implausibility threshold: `2.6`
- SA domain: `nroy`
- Baseline points used: `483`
- Reductions: `10%, 20%, 30%, 40%`

## 2) Baseline Uncertainty Decomposition
| Metric | OU var | EV var | MD var | CU var (median) | Total var (median) |
|---|---:|---:|---:|---:|---:|
| AvgPrice_mean | 0.0025 | 0.000413203 | 0.00315091 | 0.000235071 | 0.00629919 |
| BankBailedOut_share | 0.0004 | 4.82094e-05 | 0.000478297 | 2.86239e-05 | 0.00095513 |
| BankBailoutAmount_mean | 0.04 | 0.0286431 | 0.0286043 | 0.0165758 | 0.113823 |
| BankResolutionHaircut_mean | 0.0001 | 6.93363e-05 | 0.000212576 | 5.09939e-05 | 0.000432907 |
| BankResolved_share | 0.0009 | 0.00108127 | 0.0013286 | 0.000655198 | 0.00396507 |
| Consumption_mean | 100 | 48.0494 | 92.5983 | 47.6878 | 288.336 |
| DefaultsHH_rate | 4e-06 | 1.43079e-06 | 4.78297e-06 | 1.08487e-06 | 1.12986e-05 |
| HH_Deposit_mean | 1 | 0.233716 | 0.715107 | 0.0235976 | 1.97242 |
| Output_mean | 100 | 35.0397 | 92.5983 | 37.4458 | 265.084 |
| PriceDispersion_mean | 0.0036 | 0.000375478 | 0.00765275 | 0.000290509 | 0.0119187 |
| Transfers_mean | 4 | 0.315017 | 2.86043 | 0.309933 | 7.48538 |

## 3) SA Results (newly implausible share)
| Component | Reduction | Points | Newly implausible | Share |
|---|---:|---:|---:|---:|
| CU | 10% | 483 | 24 | 0.0497 |
| CU | 20% | 483 | 39 | 0.0807 |
| CU | 30% | 483 | 51 | 0.1056 |
| CU | 40% | 483 | 54 | 0.1118 |
| EV | 10% | 483 | 24 | 0.0497 |
| EV | 20% | 483 | 42 | 0.0870 |
| EV | 30% | 483 | 51 | 0.1056 |
| EV | 40% | 483 | 57 | 0.1180 |
| MD | 10% | 483 | 39 | 0.0807 |
| MD | 20% | 483 | 66 | 0.1366 |
| MD | 30% | 483 | 96 | 0.1988 |
| MD | 40% | 483 | 117 | 0.2422 |
| OU | 10% | 483 | 45 | 0.0932 |
| OU | 20% | 483 | 66 | 0.1366 |
| OU | 30% | 483 | 105 | 0.2174 |
| OU | 40% | 483 | 117 | 0.2422 |

## 4) Interpretation
- Dominant component (rank at max reduction): `MD, OU`
- Investment guidance:
  - `CU` dominant: increase design density / improve GP specification.
  - `EV` dominant: increase replicate seeds `K` or refine stochastic structure.
  - `OU` dominant: improve measurement quality / uncertainty model of observations.
  - `MD` dominant: revisit model structure or discrepancy assumptions.

## 5) Limitations
- EV is treated as metric-level scalar (not x-dependent).
- SA is local to selected baseline domain.
- MD remains expert-specified and is not inferred from data.
