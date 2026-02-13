# Uncertainty Sensitivity Report (2026-02-10)

## 1) Setup
- Dataset: `output_research_core_v2_full\02_wave2\lhs_runs_wave2.csv`
- Implausibility threshold: `3.0`
- SA domain: `nroy`
- Baseline points used: `498`
- Reductions: `10%, 20%, 30%, 40%`

## 2) Baseline Uncertainty Decomposition
| Metric | OU var | EV var | MD var | CU var (median) | Total var (median) |
|---|---:|---:|---:|---:|---:|
| AvgPrice_mean | 0.0025 | 0.000227364 | 0.00501063 | 0.000166067 | 0.00790407 |
| BankBailedOut_share | 0.0004 | 4.11523e-05 | 1.61303e-05 | 1.3994e-05 | 0.000471277 |
| BankBailoutAmount_mean | 0.04 | 0.00339765 | 0.00406885 | 0.00136378 | 0.0488303 |
| BankResolutionHaircut_mean | 0.0001 | 5.08498e-05 | 1.07355e-05 | 3.03566e-05 | 0.000191942 |
| BankResolved_share | 0.0009 | 0.00110905 | 0.000254303 | 0.000919066 | 0.00318242 |
| Bank_Equity_mean | 56.25 | 6.75938 | 5.77878 | 3.35 | 72.1382 |
| Consumption_mean | 100 | 452.717 | 162.607 | 228.08 | 943.404 |
| DefaultsHH_rate | 4e-06 | 3.59053e-07 | 3.35024e-07 | 2.71566e-07 | 4.96564e-06 |
| Employment_mean | 25 | 259.98 | 87.8572 | 142.49 | 515.327 |
| HH_Deposit_mean | 1 | 0.328402 | 0.190191 | 0.0643253 | 1.58292 |
| Output_mean | 100 | 628.467 | 120.71 | 316.842 | 1166.02 |
| Transfers_mean | 4 | 3.65438 | 0.982325 | 1.8833 | 10.52 |

## 3) SA Results (newly implausible share)
| Component | Reduction | Points | Newly implausible | Share |
|---|---:|---:|---:|---:|
| CU | 10% | 498 | 0 | 0.0000 |
| CU | 20% | 498 | 0 | 0.0000 |
| CU | 30% | 498 | 3 | 0.0060 |
| CU | 40% | 498 | 3 | 0.0060 |
| EV | 10% | 498 | 0 | 0.0000 |
| EV | 20% | 498 | 0 | 0.0000 |
| EV | 30% | 498 | 3 | 0.0060 |
| EV | 40% | 498 | 6 | 0.0120 |
| MD | 10% | 498 | 27 | 0.0542 |
| MD | 20% | 498 | 49 | 0.0984 |
| MD | 30% | 498 | 72 | 0.1446 |
| MD | 40% | 498 | 111 | 0.2229 |
| OU | 10% | 498 | 21 | 0.0422 |
| OU | 20% | 498 | 36 | 0.0723 |
| OU | 30% | 498 | 53 | 0.1064 |
| OU | 40% | 498 | 63 | 0.1265 |

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
