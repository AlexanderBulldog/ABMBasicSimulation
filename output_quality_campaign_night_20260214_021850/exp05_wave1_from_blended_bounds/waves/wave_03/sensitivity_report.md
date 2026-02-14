# Uncertainty Sensitivity Report (2026-02-14)

## 1) Setup
- Dataset: `C:\Users\Alexander\Desktop\Education\Aspa\code\output_quality_campaign_night_20260214_021850\exp05_wave1_from_blended_bounds\waves\wave_03\lhs_runs_wave3.csv`
- Implausibility threshold: `2.75`
- SA domain: `nroy`
- Baseline points used: `414`
- Reductions: `10%, 20%, 30%, 40%`

## 2) Baseline Uncertainty Decomposition
| Metric | OU var | EV var | MD var | CU var (median) | Total var (median) |
|---|---:|---:|---:|---:|---:|
| AvgPrice_mean | 0.0025 | 0.000325937 | 0.00501831 | 0.000226123 | 0.00807037 |
| Consumption_mean | 100 | 11.1314 | 147.477 | 5.32442 | 263.933 |
| DefaultsHH_rate | 4e-06 | 1.88444e-07 | 7.6176e-06 | 9.12969e-08 | 1.18973e-05 |
| HH_Deposit_mean | 1 | 1.10503 | 1.13892 | 0.364293 | 3.60824 |
| Output_mean | 100 | 14.0357 | 147.477 | 5.60634 | 267.119 |
| Transfers_mean | 4 | 0.0883244 | 4.55566 | 0.0376634 | 8.68165 |

## 3) SA Results (newly implausible share)
| Component | Reduction | Points | Newly implausible | Share |
|---|---:|---:|---:|---:|
| CU | 10% | 414 | 3 | 0.0072 |
| CU | 20% | 414 | 3 | 0.0072 |
| CU | 30% | 414 | 12 | 0.0290 |
| CU | 40% | 414 | 15 | 0.0362 |
| EV | 10% | 414 | 3 | 0.0072 |
| EV | 20% | 414 | 15 | 0.0362 |
| EV | 30% | 414 | 27 | 0.0652 |
| EV | 40% | 414 | 33 | 0.0797 |
| MD | 10% | 414 | 114 | 0.2754 |
| MD | 20% | 414 | 210 | 0.5072 |
| MD | 30% | 414 | 276 | 0.6667 |
| MD | 40% | 414 | 333 | 0.8043 |
| OU | 10% | 414 | 78 | 0.1884 |
| OU | 20% | 414 | 150 | 0.3623 |
| OU | 30% | 414 | 210 | 0.5072 |
| OU | 40% | 414 | 243 | 0.5870 |

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
