# Uncertainty Sensitivity Report (2026-02-13)

## 1) Setup
- Dataset: `C:\Users\Alexander\Desktop\Education\Aspa\code\output_research_core_v3_tuned2\waves\wave_03\lhs_runs_wave3.csv`
- Implausibility threshold: `2.75`
- SA domain: `nroy`
- Baseline points used: `96`
- Reductions: `10%, 20%, 30%, 40%`

## 2) Baseline Uncertainty Decomposition
| Metric | OU var | EV var | MD var | CU var (median) | Total var (median) |
|---|---:|---:|---:|---:|---:|
| AvgPrice_mean | 0.0025 | 0.00086006 | 0.0230992 | 0.000721112 | 0.0271804 |
| BankBailedOut_share | 0.0004 | 0.000166667 | 0.00350638 | 9.65206e-05 | 0.00416957 |
| BankBailoutAmount_mean | 0.04 | 0.0391159 | 0.209697 | 0.0267537 | 0.315567 |
| BankResolutionHaircut_mean | 0.0001 | 0.000259256 | 0.00155839 | 0.000157388 | 0.00207503 |
| BankResolved_share | 0.0009 | 0.00227315 | 0.00973994 | 0.00139146 | 0.0143045 |
| Consumption_mean | 100 | 89.5834 | 678.835 | 78.7484 | 947.167 |
| CreditRejections_mean | 25 | 12.5171 | 400 | 13.3546 | 450.872 |
| DefaultsHH_rate | 4e-06 | 6.09259e-07 | 3.50638e-05 | 4.98045e-07 | 4.01711e-05 |
| Employment_mean | 25 | 55.6105 | 169.709 | 32.9457 | 283.265 |
| HH_Deposit_mean | 1 | 1.45168 | 5.24243 | 3.67077 | 11.3649 |
| Output_mean | 100 | 82.1408 | 678.835 | 63.2767 | 924.252 |
| PriceDispersion_mean | 0.0036 | 0.00036352 | 0.0144 | 0.000289009 | 0.0186525 |
| Transfers_mean | 4 | 0.60496 | 20.9697 | 0.592275 | 26.1669 |

## 3) SA Results (newly implausible share)
| Component | Reduction | Points | Newly implausible | Share |
|---|---:|---:|---:|---:|
| CU | 10% | 96 | 3 | 0.0312 |
| CU | 20% | 96 | 9 | 0.0938 |
| CU | 30% | 96 | 12 | 0.1250 |
| CU | 40% | 96 | 12 | 0.1250 |
| EV | 10% | 96 | 0 | 0.0000 |
| EV | 20% | 96 | 0 | 0.0000 |
| EV | 30% | 96 | 0 | 0.0000 |
| EV | 40% | 96 | 0 | 0.0000 |
| MD | 10% | 96 | 0 | 0.0000 |
| MD | 20% | 96 | 9 | 0.0938 |
| MD | 30% | 96 | 9 | 0.0938 |
| MD | 40% | 96 | 21 | 0.2188 |
| OU | 10% | 96 | 0 | 0.0000 |
| OU | 20% | 96 | 0 | 0.0000 |
| OU | 30% | 96 | 0 | 0.0000 |
| OU | 40% | 96 | 0 | 0.0000 |

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
