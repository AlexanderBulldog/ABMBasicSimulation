# Uncertainty Sensitivity Report (2026-02-14)

## 1) Setup
- Dataset: `C:\Users\Alexander\Desktop\Education\Aspa\code\output_quality_campaign_night_20260214_021850\exp02_emu_focus_high_n\waves\wave_05\lhs_runs_wave5.csv`
- Implausibility threshold: `2.75`
- SA domain: `nroy`
- Baseline points used: `405`
- Reductions: `10%, 20%, 30%, 40%`

## 2) Baseline Uncertainty Decomposition
| Metric | OU var | EV var | MD var | CU var (median) | Total var (median) |
|---|---:|---:|---:|---:|---:|
| AvgPrice_mean | 0.0025 | 0.000500428 | 0.00682735 | 0.000348607 | 0.0101764 |
| BankBailedOut_share | 0.0004 | 5.92593e-05 | 0.00103637 | 2.62633e-05 | 0.00152189 |
| BankBailoutAmount_mean | 0.04 | 0.007592 | 0.0619793 | 0.003659 | 0.11323 |
| BankResolutionHaircut_mean | 0.0001 | 0.000112696 | 0.000460607 | 6.49906e-05 | 0.000738293 |
| BankResolved_share | 0.0009 | 0.00143704 | 0.00287879 | 0.000959312 | 0.00617514 |
| Consumption_mean | 100 | 40.067 | 200.64 | 33.1873 | 373.895 |
| CreditRejections_mean | 25 | 19.8166 | 460.607 | 13.3197 | 518.743 |
| DefaultsHH_rate | 4e-06 | 1.38059e-06 | 1.03637e-05 | 1.01283e-06 | 1.67571e-05 |
| HH_Deposit_mean | 1 | 1.88265 | 1.54948 | 1.00735 | 5.43948 |
| Output_mean | 100 | 37.0708 | 200.64 | 26.0402 | 363.751 |
| PriceDispersion_mean | 0.0036 | 0.00040161 | 0.0165819 | 0.000290608 | 0.0208741 |
| Transfers_mean | 4 | 0.342351 | 6.19793 | 0.274881 | 10.8152 |

## 3) SA Results (newly implausible share)
| Component | Reduction | Points | Newly implausible | Share |
|---|---:|---:|---:|---:|
| CU | 10% | 405 | 3 | 0.0074 |
| CU | 20% | 405 | 3 | 0.0074 |
| CU | 30% | 405 | 6 | 0.0148 |
| CU | 40% | 405 | 18 | 0.0444 |
| EV | 10% | 405 | 0 | 0.0000 |
| EV | 20% | 405 | 3 | 0.0074 |
| EV | 30% | 405 | 12 | 0.0296 |
| EV | 40% | 405 | 21 | 0.0519 |
| MD | 10% | 405 | 27 | 0.0667 |
| MD | 20% | 405 | 75 | 0.1852 |
| MD | 30% | 405 | 120 | 0.2963 |
| MD | 40% | 405 | 159 | 0.3926 |
| OU | 10% | 405 | 3 | 0.0074 |
| OU | 20% | 405 | 27 | 0.0667 |
| OU | 30% | 405 | 48 | 0.1185 |
| OU | 40% | 405 | 78 | 0.1926 |

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
