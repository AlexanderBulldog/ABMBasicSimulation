# Uncertainty Sensitivity Report (2026-02-14)

## 1) Setup
- Dataset: `C:\Users\Alexander\Desktop\Education\Aspa\code\output_research_core_v4_preflight1\01_wave1\lhs_runs.csv`
- Implausibility threshold: `2.75`
- SA domain: `nroy`
- Baseline points used: `41`
- Reductions: `10%, 20%, 30%, 40%`

## 2) Baseline Uncertainty Decomposition
| Metric | OU var | EV var | MD var | CU var (median) | Total var (median) |
|---|---:|---:|---:|---:|---:|
| AvgPrice_mean | 0.0025 | 0.000357694 | 0.005929 | 0.000210593 | 0.00899729 |
| BankBailedOut_share | 0.0004 | 1.48148e-05 | 0.0009 | 2.17044e-05 | 0.00133652 |
| BankBailoutAmount_mean | 0.04 | 0.00692484 | 0.053824 | 0.00319163 | 0.10394 |
| BankResolutionHaircut_mean | 0.0001 | 4.19309e-05 | 0.0004 | 2.95179e-05 | 0.000571449 |
| BankResolved_share | 0.0009 | 0.000568889 | 0.0025 | 0.000273747 | 0.00424264 |
| Bank_Equity_mean | 56.25 | 26.5401 | 59.9076 | 23.9089 | 166.607 |
| Consumption_mean | 100 | 47.4325 | 174.24 | 26.6673 | 348.34 |
| CreditRejections_mean | 100 | 23.8765 | 400 | 11.9409 | 535.817 |
| DefaultsHH_rate | 4e-06 | 4.28593e-07 | 9e-06 | 5.06854e-07 | 1.39354e-05 |
| Employment_mean | 25 | 26.7973 | 43.56 | 13.4994 | 108.857 |
| HH_Deposit_mean | 1 | 0.787826 | 1.3456 | 0.0364772 | 3.1699 |
| Output_mean | 100 | 81.4618 | 174.24 | 33.2843 | 388.986 |
| PriceDispersion_mean | 0.0036 | 0.000320086 | 0.0144 | 0.000234727 | 0.0185548 |
| Transfers_mean | 4 | 0.40714 | 5.3824 | 0.205924 | 9.99546 |

## 3) SA Results (newly implausible share)
| Component | Reduction | Points | Newly implausible | Share |
|---|---:|---:|---:|---:|
| CU | 10% | 41 | 0 | 0.0000 |
| CU | 20% | 41 | 6 | 0.1463 |
| CU | 30% | 41 | 6 | 0.1463 |
| CU | 40% | 41 | 9 | 0.2195 |
| EV | 10% | 41 | 6 | 0.1463 |
| EV | 20% | 41 | 9 | 0.2195 |
| EV | 30% | 41 | 9 | 0.2195 |
| EV | 40% | 41 | 12 | 0.2927 |
| MD | 10% | 41 | 12 | 0.2927 |
| MD | 20% | 41 | 15 | 0.3659 |
| MD | 30% | 41 | 21 | 0.5122 |
| MD | 40% | 41 | 21 | 0.5122 |
| OU | 10% | 41 | 9 | 0.2195 |
| OU | 20% | 41 | 12 | 0.2927 |
| OU | 30% | 41 | 12 | 0.2927 |
| OU | 40% | 41 | 15 | 0.3659 |

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
