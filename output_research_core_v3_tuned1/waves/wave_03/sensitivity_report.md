# Uncertainty Sensitivity Report (2026-02-13)

## 1) Setup
- Dataset: `C:\Users\Alexander\Desktop\Education\Aspa\code\output_research_core_v3_tuned1\waves\wave_03\lhs_runs_wave3.csv`
- Implausibility threshold: `2.75`
- SA domain: `nroy`
- Baseline points used: `219`
- Reductions: `10%, 20%, 30%, 40%`

## 2) Baseline Uncertainty Decomposition
| Metric | OU var | EV var | MD var | CU var (median) | Total var (median) |
|---|---:|---:|---:|---:|---:|
| AvgPrice_mean | 0.0025 | 0.000592302 | 0.0093294 | 0.000525509 | 0.0129472 |
| BankBailedOut_share | 0.0004 | 0.000208333 | 0.00141617 | 0.000112532 | 0.00213703 |
| BankBailoutAmount_mean | 0.04 | 0.0531202 | 0.0846931 | 0.0349636 | 0.212777 |
| BankResolutionHaircut_mean | 0.0001 | 0.000307822 | 0.000629408 | 0.000206108 | 0.00124334 |
| BankResolved_share | 0.0009 | 0.00216204 | 0.0039338 | 0.00126511 | 0.00826094 |
| Consumption_mean | 100 | 60.1699 | 274.17 | 121.504 | 555.843 |
| CreditRejections_mean | 25 | 14.4893 | 629.408 | 7.49171 | 676.389 |
| DefaultsHH_rate | 4e-06 | 1.32361e-06 | 1.41617e-05 | 1.4544e-06 | 2.09397e-05 |
| HH_Deposit_mean | 1 | 1.61069 | 2.11733 | 0.408814 | 5.13683 |
| Output_mean | 100 | 55.2971 | 274.17 | 93.4209 | 522.888 |
| PriceDispersion_mean | 0.0036 | 0.000516142 | 0.0226587 | 0.000543168 | 0.027318 |
| Transfers_mean | 4 | 0.564973 | 8.46931 | 0.902036 | 13.9363 |

## 3) SA Results (newly implausible share)
| Component | Reduction | Points | Newly implausible | Share |
|---|---:|---:|---:|---:|
| CU | 10% | 219 | 9 | 0.0411 |
| CU | 20% | 219 | 12 | 0.0548 |
| CU | 30% | 219 | 15 | 0.0685 |
| CU | 40% | 219 | 18 | 0.0822 |
| EV | 10% | 219 | 0 | 0.0000 |
| EV | 20% | 219 | 0 | 0.0000 |
| EV | 30% | 219 | 0 | 0.0000 |
| EV | 40% | 219 | 9 | 0.0411 |
| MD | 10% | 219 | 3 | 0.0137 |
| MD | 20% | 219 | 9 | 0.0411 |
| MD | 30% | 219 | 30 | 0.1370 |
| MD | 40% | 219 | 45 | 0.2055 |
| OU | 10% | 219 | 0 | 0.0000 |
| OU | 20% | 219 | 0 | 0.0000 |
| OU | 30% | 219 | 3 | 0.0137 |
| OU | 40% | 219 | 3 | 0.0137 |

## 4) Interpretation
- Dominant component (rank at max reduction): `MD`
- Investment guidance:
  - `CU` dominant: increase design density / improve GP specification.
  - `EV` dominant: increase replicate seeds `K` or refine stochastic structure.
  - `OU` dominant: improve measurement quality / uncertainty model of observations.
  - `MD` dominant: revisit model structure or discrepancy assumptions.

## 5) Limitations
- EV is treated as metric-level scalar (not x-dependent).
- SA is local to selected baseline domain.
- MD remains expert-specified and is not inferred from data.
