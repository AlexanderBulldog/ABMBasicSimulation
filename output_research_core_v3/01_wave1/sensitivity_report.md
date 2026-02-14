# Uncertainty Sensitivity Report (2026-02-13)

## 1) Setup
- Dataset: `C:\Users\Alexander\Desktop\Education\Aspa\code\output_research_core_v3\01_wave1\lhs_runs.csv`
- Implausibility threshold: `2.6`
- SA domain: `nroy`
- Baseline points used: `314`
- Reductions: `10%, 20%, 30%, 40%`

## 2) Baseline Uncertainty Decomposition
| Metric | OU var | EV var | MD var | CU var (median) | Total var (median) |
|---|---:|---:|---:|---:|---:|
| AvgPrice_mean | 0.0025 | 0.000381005 | 0.0495135 | 0.00025287 | 0.0526473 |
| BankBailedOut_share | 0.0004 | 2.31481e-05 | 0.0004 | 1.5716e-05 | 0.000838864 |
| BankBailoutAmount_mean | 0.04 | 0.00997113 | 0.04 | 0.0041573 | 0.0941284 |
| BankResolutionHaircut_mean | 0.0001 | 5.61509e-05 | 0.00026962 | 1.70906e-05 | 0.000442861 |
| BankResolved_share | 0.0009 | 0.000648148 | 0.0025 | 0.000295279 | 0.00434343 |
| Bank_Equity_mean | 56.25 | 22.9858 | 19.5597 | 12.4743 | 111.27 |
| Consumption_mean | 100 | 44.3828 | 1794.45 | 69.0273 | 2007.86 |
| CreditRejections_mean | 1 | 16.4145 | 9980.01 | 7.19292 | 10004.6 |
| DefaultsHH_rate | 4e-06 | 7.39815e-07 | 8.66975e-06 | 2.35979e-07 | 1.36455e-05 |
| HH_Deposit_mean | 1 | 0.403916 | 2.25 | 0.0320891 | 3.68601 |
| InventoryGap_mean | 0.0625 | 0.0366432 | 0.00734057 | 0.0150666 | 0.12155 |
| Output_mean | 100 | 72.2976 | 3751.14 | 74.8499 | 3998.29 |
| PriceDispersion_mean | 0.0036 | 0.000399553 | 0.00190377 | 0.000246272 | 0.0061496 |
| Transfers_mean | 4 | 0.400593 | 22.5429 | 0.559919 | 27.5034 |

## 3) SA Results (newly implausible share)
| Component | Reduction | Points | Newly implausible | Share |
|---|---:|---:|---:|---:|
| CU | 10% | 314 | 0 | 0.0000 |
| CU | 20% | 314 | 0 | 0.0000 |
| CU | 30% | 314 | 0 | 0.0000 |
| CU | 40% | 314 | 0 | 0.0000 |
| EV | 10% | 314 | 0 | 0.0000 |
| EV | 20% | 314 | 0 | 0.0000 |
| EV | 30% | 314 | 0 | 0.0000 |
| EV | 40% | 314 | 0 | 0.0000 |
| MD | 10% | 314 | 0 | 0.0000 |
| MD | 20% | 314 | 6 | 0.0191 |
| MD | 30% | 314 | 15 | 0.0478 |
| MD | 40% | 314 | 21 | 0.0669 |
| OU | 10% | 314 | 0 | 0.0000 |
| OU | 20% | 314 | 0 | 0.0000 |
| OU | 30% | 314 | 0 | 0.0000 |
| OU | 40% | 314 | 3 | 0.0096 |

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
