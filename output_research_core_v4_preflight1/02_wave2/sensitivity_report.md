# Uncertainty Sensitivity Report (2026-02-14)

## 1) Setup
- Dataset: `C:\Users\Alexander\Desktop\Education\Aspa\code\output_research_core_v4_preflight1\02_wave2\lhs_runs_wave2.csv`
- Implausibility threshold: `2.75`
- SA domain: `nroy`
- Baseline points used: `420`
- Reductions: `10%, 20%, 30%, 40%`

## 2) Baseline Uncertainty Decomposition
| Metric | OU var | EV var | MD var | CU var (median) | Total var (median) |
|---|---:|---:|---:|---:|---:|
| AvgPrice_mean | 0.0025 | 0.000787577 | 0.00501831 | 0.000463289 | 0.00876917 |
| BankBailedOut_share | 0.0004 | 5.92593e-05 | 0.00076176 | 6.43319e-05 | 0.00128535 |
| BankBailoutAmount_mean | 0.04 | 0.010939 | 0.0455566 | 0.00687777 | 0.103373 |
| BankResolutionHaircut_mean | 0.0001 | 8.78469e-05 | 0.00033856 | 7.88197e-05 | 0.000605227 |
| BankResolved_share | 0.0009 | 0.000651852 | 0.002116 | 0.000376209 | 0.00404406 |
| Consumption_mean | 100 | 71.045 | 147.477 | 141.488 | 460.01 |
| CreditRejections_mean | 100 | 14.2445 | 338.56 | 9.60625 | 462.411 |
| DefaultsHH_rate | 4e-06 | 3.6237e-07 | 7.6176e-06 | 5.99604e-07 | 1.25796e-05 |
| Employment_mean | 25 | 37.6572 | 36.8692 | 80.7703 | 180.297 |
| HH_Deposit_mean | 1 | 0.96049 | 1.13892 | 0.0432363 | 3.14264 |
| Output_mean | 100 | 67.1979 | 147.477 | 123.406 | 438.081 |
| PriceDispersion_mean | 0.0036 | 0.000479691 | 0.0121882 | 0.000581525 | 0.0168494 |
| Transfers_mean | 4 | 0.541496 | 4.55566 | 1.23696 | 10.3341 |

## 3) SA Results (newly implausible share)
| Component | Reduction | Points | Newly implausible | Share |
|---|---:|---:|---:|---:|
| CU | 10% | 420 | 12 | 0.0286 |
| CU | 20% | 420 | 18 | 0.0429 |
| CU | 30% | 420 | 45 | 0.1071 |
| CU | 40% | 420 | 75 | 0.1786 |
| EV | 10% | 420 | 6 | 0.0143 |
| EV | 20% | 420 | 9 | 0.0214 |
| EV | 30% | 420 | 12 | 0.0286 |
| EV | 40% | 420 | 15 | 0.0357 |
| MD | 10% | 420 | 9 | 0.0214 |
| MD | 20% | 420 | 18 | 0.0429 |
| MD | 30% | 420 | 36 | 0.0857 |
| MD | 40% | 420 | 51 | 0.1214 |
| OU | 10% | 420 | 9 | 0.0214 |
| OU | 20% | 420 | 12 | 0.0286 |
| OU | 30% | 420 | 18 | 0.0429 |
| OU | 40% | 420 | 30 | 0.0714 |

## 4) Interpretation
- Dominant component (rank at max reduction): `CU`
- Investment guidance:
  - `CU` dominant: increase design density / improve GP specification.
  - `EV` dominant: increase replicate seeds `K` or refine stochastic structure.
  - `OU` dominant: improve measurement quality / uncertainty model of observations.
  - `MD` dominant: revisit model structure or discrepancy assumptions.

## 5) Limitations
- EV is treated as metric-level scalar (not x-dependent).
- SA is local to selected baseline domain.
- MD remains expert-specified and is not inferred from data.
