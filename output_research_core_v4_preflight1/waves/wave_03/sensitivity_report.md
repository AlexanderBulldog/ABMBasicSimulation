# Uncertainty Sensitivity Report (2026-02-14)

## 1) Setup
- Dataset: `C:\Users\Alexander\Desktop\Education\Aspa\code\output_research_core_v4_preflight1\waves\wave_03\lhs_runs_wave3.csv`
- Implausibility threshold: `2.75`
- SA domain: `nroy`
- Baseline points used: `483`
- Reductions: `10%, 20%, 30%, 40%`

## 2) Baseline Uncertainty Decomposition
| Metric | OU var | EV var | MD var | CU var (median) | Total var (median) |
|---|---:|---:|---:|---:|---:|
| AvgPrice_mean | 0.0025 | 0.000521861 | 0.00304287 | 0.000404645 | 0.00646938 |
| BankBailedOut_share | 0.0004 | 5.92593e-05 | 0.000461897 | 5.11336e-05 | 0.00097229 |
| BankBailoutAmount_mean | 0.04 | 0.0116577 | 0.0276235 | 0.00465462 | 0.0839358 |
| BankResolutionHaircut_mean | 0.0001 | 7.00379e-05 | 0.000205288 | 8.32857e-05 | 0.000458611 |
| BankResolved_share | 0.0009 | 0.000714074 | 0.00128305 | 0.000438823 | 0.00333594 |
| Consumption_mean | 100 | 74.0178 | 89.4233 | 102.048 | 365.489 |
| CreditRejections_mean | 100 | 16.5972 | 205.288 | 8.83431 | 330.719 |
| DefaultsHH_rate | 4e-06 | 4.80889e-07 | 4.61897e-06 | 5.96539e-07 | 9.6964e-06 |
| HH_Deposit_mean | 1 | 0.591714 | 0.690587 | 0.046813 | 2.32911 |
| Output_mean | 100 | 61.3776 | 89.4233 | 80.2982 | 331.099 |
| PriceDispersion_mean | 0.0036 | 0.000446167 | 0.00739035 | 0.000639409 | 0.0120759 |
| Transfers_mean | 4 | 0.632215 | 2.76235 | 0.875793 | 8.27036 |

## 3) SA Results (newly implausible share)
| Component | Reduction | Points | Newly implausible | Share |
|---|---:|---:|---:|---:|
| CU | 10% | 483 | 18 | 0.0373 |
| CU | 20% | 483 | 45 | 0.0932 |
| CU | 30% | 483 | 57 | 0.1180 |
| CU | 40% | 483 | 66 | 0.1366 |
| EV | 10% | 483 | 15 | 0.0311 |
| EV | 20% | 483 | 33 | 0.0683 |
| EV | 30% | 483 | 45 | 0.0932 |
| EV | 40% | 483 | 60 | 0.1242 |
| MD | 10% | 483 | 27 | 0.0559 |
| MD | 20% | 483 | 57 | 0.1180 |
| MD | 30% | 483 | 78 | 0.1615 |
| MD | 40% | 483 | 96 | 0.1988 |
| OU | 10% | 483 | 24 | 0.0497 |
| OU | 20% | 483 | 57 | 0.1180 |
| OU | 30% | 483 | 78 | 0.1615 |
| OU | 40% | 483 | 96 | 0.1988 |

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
