# Uncertainty Sensitivity Report (2026-02-14)

## 1) Setup
- Dataset: `C:\Users\Alexander\Desktop\Education\Aspa\code\output_quality_campaign_night_20260214_021850\exp01_baseline_dense\waves\wave_03\lhs_runs_wave3.csv`
- Implausibility threshold: `2.75`
- SA domain: `nroy`
- Baseline points used: `252`
- Reductions: `10%, 20%, 30%, 40%`

## 2) Baseline Uncertainty Decomposition
| Metric | OU var | EV var | MD var | CU var (median) | Total var (median) |
|---|---:|---:|---:|---:|---:|
| AvgPrice_mean | 0.0025 | 0.000649986 | 0.00717409 | 0.000399922 | 0.010724 |
| BankBailedOut_share | 0.0004 | 1.70068e-05 | 0.001089 | 2.95098e-05 | 0.00153552 |
| BankBailoutAmount_mean | 0.04 | 0.000745576 | 0.065127 | 0.00387173 | 0.109744 |
| BankResolutionHaircut_mean | 0.0001 | 6.24129e-05 | 0.000484 | 5.96765e-05 | 0.000706089 |
| BankResolved_share | 0.0009 | 0.00104762 | 0.003025 | 0.000522345 | 0.00549496 |
| Consumption_mean | 100 | 55.2733 | 210.83 | 46.3241 | 412.428 |
| CreditRejections_mean | 25 | 26.6865 | 484 | 16.6699 | 552.356 |
| DefaultsHH_rate | 4e-06 | 1.3818e-06 | 1.089e-05 | 1.09225e-06 | 1.73641e-05 |
| HH_Deposit_mean | 1 | 1.83853 | 1.62818 | 1.05454 | 5.52125 |
| Output_mean | 100 | 52.5548 | 210.83 | 40.0704 | 403.456 |
| PriceDispersion_mean | 0.0036 | 0.000619418 | 0.017424 | 0.000385629 | 0.022029 |
| Transfers_mean | 4 | 0.511596 | 6.5127 | 0.396368 | 11.4207 |

## 3) SA Results (newly implausible share)
| Component | Reduction | Points | Newly implausible | Share |
|---|---:|---:|---:|---:|
| CU | 10% | 252 | 3 | 0.0119 |
| CU | 20% | 252 | 3 | 0.0119 |
| CU | 30% | 252 | 9 | 0.0357 |
| CU | 40% | 252 | 9 | 0.0357 |
| EV | 10% | 252 | 3 | 0.0119 |
| EV | 20% | 252 | 3 | 0.0119 |
| EV | 30% | 252 | 6 | 0.0238 |
| EV | 40% | 252 | 6 | 0.0238 |
| MD | 10% | 252 | 3 | 0.0119 |
| MD | 20% | 252 | 30 | 0.1190 |
| MD | 30% | 252 | 42 | 0.1667 |
| MD | 40% | 252 | 54 | 0.2143 |
| OU | 10% | 252 | 0 | 0.0000 |
| OU | 20% | 252 | 3 | 0.0119 |
| OU | 30% | 252 | 6 | 0.0238 |
| OU | 40% | 252 | 24 | 0.0952 |

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
