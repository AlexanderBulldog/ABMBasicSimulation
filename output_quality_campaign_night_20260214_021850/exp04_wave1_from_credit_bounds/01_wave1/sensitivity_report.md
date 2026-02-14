# Uncertainty Sensitivity Report (2026-02-14)

## 1) Setup
- Dataset: `C:\Users\Alexander\Desktop\Education\Aspa\code\output_quality_campaign_night_20260214_021850\exp04_wave1_from_credit_bounds\01_wave1\lhs_runs.csv`
- Implausibility threshold: `2.75`
- SA domain: `nroy`
- Baseline points used: `18`
- Reductions: `10%, 20%, 30%, 40%`

## 2) Baseline Uncertainty Decomposition
| Metric | OU var | EV var | MD var | CU var (median) | Total var (median) |
|---|---:|---:|---:|---:|---:|
| AvgPrice_mean | 0.0025 | 0.00044099 | 0.005929 | 0.000287057 | 0.00915705 |
| BankResolutionHaircut_mean | 0.0001 | 3.84925e-05 | 0.0004 | 3.288e-05 | 0.000571373 |
| BankResolved_share | 0.0009 | 0.00144048 | 0.0025 | 0.000657393 | 0.00549787 |
| Consumption_mean | 100 | 28.8052 | 174.24 | 14.82 | 317.865 |
| CreditRejections_mean | 25 | 16.5413 | 400 | 9.76431 | 451.306 |
| DefaultsHH_rate | 4e-06 | 4.28571e-07 | 9e-06 | 3.70504e-07 | 1.37991e-05 |
| Employment_mean | 25 | 16.5536 | 43.56 | 7.53847 | 92.6521 |
| HH_Deposit_mean | 1 | 1.85081 | 1.3456 | 1.6227 | 5.81911 |
| Output_mean | 100 | 27.2464 | 174.24 | 16.8507 | 318.337 |
| PriceDispersion_mean | 0.0036 | 0.000351032 | 0.0144 | 0.000220275 | 0.0185713 |
| Transfers_mean | 4 | 0.239245 | 5.3824 | 0.112914 | 9.73456 |

## 3) SA Results (newly implausible share)
| Component | Reduction | Points | Newly implausible | Share |
|---|---:|---:|---:|---:|
| CU | 10% | 18 | 0 | 0.0000 |
| CU | 20% | 18 | 3 | 0.1667 |
| CU | 30% | 18 | 3 | 0.1667 |
| CU | 40% | 18 | 3 | 0.1667 |
| EV | 10% | 18 | 3 | 0.1667 |
| EV | 20% | 18 | 3 | 0.1667 |
| EV | 30% | 18 | 6 | 0.3333 |
| EV | 40% | 18 | 6 | 0.3333 |
| MD | 10% | 18 | 6 | 0.3333 |
| MD | 20% | 18 | 12 | 0.6667 |
| MD | 30% | 18 | 15 | 0.8333 |
| MD | 40% | 18 | 15 | 0.8333 |
| OU | 10% | 18 | 3 | 0.1667 |
| OU | 20% | 18 | 6 | 0.3333 |
| OU | 30% | 18 | 6 | 0.3333 |
| OU | 40% | 18 | 12 | 0.6667 |

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
