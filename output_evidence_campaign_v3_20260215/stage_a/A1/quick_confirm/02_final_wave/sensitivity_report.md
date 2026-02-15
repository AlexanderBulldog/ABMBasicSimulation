# Uncertainty Sensitivity Report (2026-02-15)

## 1) Setup
- Dataset: `C:\Users\Alexander\Desktop\Education\Aspa\code\output_evidence_campaign_v3_20260215\stage_a\A1\quick_confirm\waves\wave_03\lhs_runs_wave3.csv`
- Implausibility threshold: `2.6`
- SA domain: `nroy`
- Baseline points used: `108`
- Reductions: `10%, 20%, 30%, 40%`

## 2) Baseline Uncertainty Decomposition
| Metric | OU var | EV var | MD var | CU var (median) | Total var (median) |
|---|---:|---:|---:|---:|---:|
| AvgPrice_mean | 0.0025 | 0.000622893 | 0.00806634 | 0.000422443 | 0.0116117 |
| BankBailedOut_share | 0.0004 | 0.000162037 | 0.00122444 | 8.69676e-05 | 0.00187344 |
| BankBailoutAmount_mean | 0.04 | 0.0881987 | 0.073227 | 0.056224 | 0.25765 |
| BankResolutionHaircut_mean | 0.0001 | 0.00024409 | 0.000544196 | 0.000119581 | 0.00100787 |
| BankResolved_share | 0.0009 | 0.00121528 | 0.00340122 | 0.000646147 | 0.00616265 |
| Consumption_mean | 100 | 46.1153 | 237.052 | 20.1097 | 403.277 |
| DefaultsHH_rate | 4e-06 | 3.36759e-06 | 1.22444e-05 | 3.11361e-06 | 2.27256e-05 |
| Employment_mean | 25 | 29.89 | 59.2629 | 11.4485 | 125.601 |
| HH_Deposit_mean | 1 | 0.225093 | 1.83067 | 0.020237 | 3.076 |
| Output_mean | 100 | 48.5244 | 237.052 | 20.6692 | 406.245 |
| PriceDispersion_mean | 0.0036 | 0.000413775 | 0.019591 | 0.000288019 | 0.0238928 |
| Transfers_mean | 4 | 0.336328 | 7.3227 | 0.152572 | 11.8116 |

## 3) SA Results (newly implausible share)
| Component | Reduction | Points | Newly implausible | Share |
|---|---:|---:|---:|---:|
| CU | 10% | 108 | 6 | 0.0556 |
| CU | 20% | 108 | 15 | 0.1389 |
| CU | 30% | 108 | 15 | 0.1389 |
| CU | 40% | 108 | 21 | 0.1944 |
| EV | 10% | 108 | 12 | 0.1111 |
| EV | 20% | 108 | 24 | 0.2222 |
| EV | 30% | 108 | 33 | 0.3056 |
| EV | 40% | 108 | 42 | 0.3889 |
| MD | 10% | 108 | 21 | 0.1944 |
| MD | 20% | 108 | 42 | 0.3889 |
| MD | 30% | 108 | 57 | 0.5278 |
| MD | 40% | 108 | 66 | 0.6111 |
| OU | 10% | 108 | 15 | 0.1389 |
| OU | 20% | 108 | 18 | 0.1667 |
| OU | 30% | 108 | 27 | 0.2500 |
| OU | 40% | 108 | 33 | 0.3056 |

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
