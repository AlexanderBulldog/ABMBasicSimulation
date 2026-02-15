# Uncertainty Sensitivity Report (2026-02-15)

## 1) Setup
- Dataset: `C:\Users\Alexander\Desktop\Education\Aspa\code\output_confirm_fix_check_20260215\01_wave1\lhs_runs.csv`
- Implausibility threshold: `2.6`
- SA domain: `nroy`
- Baseline points used: `89`
- Reductions: `10%, 20%, 30%, 40%`

## 2) Baseline Uncertainty Decomposition
| Metric | OU var | EV var | MD var | CU var (median) | Total var (median) |
|---|---:|---:|---:|---:|---:|
| AvgPrice_mean | 0.0025 | 0.00048057 | 0.005929 | 0.000730615 | 0.00964019 |
| BankBailedOut_share | 0.0004 | 9.98623e-05 | 0.0009 | 8.48911e-05 | 0.00148475 |
| BankBailoutAmount_mean | 0.04 | 0.0226332 | 0.053824 | 0.0267607 | 0.143218 |
| BankResolutionHaircut_mean | 0.0001 | 0.00011792 | 0.0004 | 0.000120712 | 0.000738632 |
| BankResolved_share | 0.0009 | 0.000761019 | 0.0025 | 0.000533416 | 0.00469444 |
| Consumption_mean | 100 | 46.663 | 174.24 | 99.3951 | 420.298 |
| DefaultsHH_rate | 4e-06 | 1.98623e-06 | 9e-06 | 1.15326e-06 | 1.61395e-05 |
| Employment_mean | 25 | 24.9348 | 43.56 | 55.4905 | 148.985 |
| HH_Deposit_mean | 1 | 0.819501 | 1.3456 | 0.0544676 | 3.21957 |
| Output_mean | 100 | 59.752 | 174.24 | 82.2645 | 416.256 |
| PriceDispersion_mean | 0.0036 | 0.000396343 | 0.0144 | 0.000789307 | 0.0191856 |
| Transfers_mean | 4 | 0.335011 | 5.3824 | 0.8691 | 10.5865 |

## 3) SA Results (newly implausible share)
| Component | Reduction | Points | Newly implausible | Share |
|---|---:|---:|---:|---:|
| CU | 10% | 89 | 6 | 0.0674 |
| CU | 20% | 89 | 9 | 0.1011 |
| CU | 30% | 89 | 21 | 0.2360 |
| CU | 40% | 89 | 36 | 0.4045 |
| EV | 10% | 89 | 3 | 0.0337 |
| EV | 20% | 89 | 6 | 0.0674 |
| EV | 30% | 89 | 6 | 0.0674 |
| EV | 40% | 89 | 6 | 0.0674 |
| MD | 10% | 89 | 6 | 0.0674 |
| MD | 20% | 89 | 9 | 0.1011 |
| MD | 30% | 89 | 18 | 0.2022 |
| MD | 40% | 89 | 27 | 0.3034 |
| OU | 10% | 89 | 3 | 0.0337 |
| OU | 20% | 89 | 6 | 0.0674 |
| OU | 30% | 89 | 9 | 0.1011 |
| OU | 40% | 89 | 12 | 0.1348 |

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
