# Uncertainty Sensitivity Report (2026-02-14)

## 1) Setup
- Dataset: `C:\Users\Alexander\Desktop\Education\Aspa\code\output_quality_campaign_night_20260214_021850\exp03_adapt_emu\02_wave2\lhs_runs_wave2.csv`
- Implausibility threshold: `2.6`
- SA domain: `nroy`
- Baseline points used: `308`
- Reductions: `10%, 20%, 30%, 40%`

## 2) Baseline Uncertainty Decomposition
| Metric | OU var | EV var | MD var | CU var (median) | Total var (median) |
|---|---:|---:|---:|---:|---:|
| AvgPrice_mean | 0.0025 | 0.000595491 | 0.00944993 | 0.000669186 | 0.0132146 |
| BankBailedOut_share | 0.0004 | 1.30208e-05 | 0.00143446 | 4.30769e-05 | 0.00189056 |
| BankBailoutAmount_mean | 0.04 | 0.00203581 | 0.0857873 | 0.00357647 | 0.1314 |
| BankResolutionHaircut_mean | 0.0001 | 6.69687e-05 | 0.000637539 | 5.95307e-05 | 0.000864039 |
| BankResolved_share | 0.0009 | 0.00102865 | 0.00398462 | 0.000590567 | 0.00650383 |
| Consumption_mean | 100 | 52.6631 | 277.712 | 100.202 | 530.577 |
| CreditRejections_mean | 25 | 23.0471 | 637.539 | 27.1093 | 712.696 |
| DefaultsHH_rate | 4e-06 | 1.61406e-06 | 1.43446e-05 | 1.12181e-06 | 2.10805e-05 |
| Employment_mean | 25 | 30.6063 | 69.428 | 47.3409 | 172.375 |
| HH_Deposit_mean | 1 | 2.46214 | 2.14468 | 1.27352 | 6.88035 |
| Output_mean | 100 | 40.7547 | 277.712 | 67.5354 | 486.002 |
| PriceDispersion_mean | 0.0036 | 0.000616775 | 0.0229514 | 0.000470279 | 0.0276385 |
| Transfers_mean | 4 | 0.514087 | 8.57873 | 0.79991 | 13.8927 |

## 3) SA Results (newly implausible share)
| Component | Reduction | Points | Newly implausible | Share |
|---|---:|---:|---:|---:|
| CU | 10% | 308 | 60 | 0.1948 |
| CU | 20% | 308 | 96 | 0.3117 |
| CU | 30% | 308 | 147 | 0.4773 |
| CU | 40% | 308 | 162 | 0.5260 |
| EV | 10% | 308 | 33 | 0.1071 |
| EV | 20% | 308 | 69 | 0.2240 |
| EV | 30% | 308 | 90 | 0.2922 |
| EV | 40% | 308 | 108 | 0.3506 |
| MD | 10% | 308 | 75 | 0.2435 |
| MD | 20% | 308 | 123 | 0.3994 |
| MD | 30% | 308 | 174 | 0.5649 |
| MD | 40% | 308 | 186 | 0.6039 |
| OU | 10% | 308 | 24 | 0.0779 |
| OU | 20% | 308 | 60 | 0.1948 |
| OU | 30% | 308 | 75 | 0.2435 |
| OU | 40% | 308 | 87 | 0.2825 |

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
