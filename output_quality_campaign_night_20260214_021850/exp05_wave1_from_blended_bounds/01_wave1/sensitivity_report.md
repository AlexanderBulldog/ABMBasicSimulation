# Uncertainty Sensitivity Report (2026-02-14)

## 1) Setup
- Dataset: `C:\Users\Alexander\Desktop\Education\Aspa\code\output_quality_campaign_night_20260214_021850\exp05_wave1_from_blended_bounds\01_wave1\lhs_runs.csv`
- Implausibility threshold: `2.75`
- SA domain: `nroy`
- Baseline points used: `165`
- Reductions: `10%, 20%, 30%, 40%`

## 2) Baseline Uncertainty Decomposition
| Metric | OU var | EV var | MD var | CU var (median) | Total var (median) |
|---|---:|---:|---:|---:|---:|
| AvgPrice_mean | 0.0025 | 0.00026281 | 0.005929 | 0.000185244 | 0.00887705 |
| BankResolutionHaircut_mean | 0.0001 | 6.46785e-06 | 0.0004 | 3.86958e-06 | 0.000510337 |
| BankResolved_share | 0.0009 | 0.000551111 | 0.0025 | 0.00032347 | 0.00427458 |
| Consumption_mean | 100 | 22.1819 | 174.24 | 11.7957 | 308.218 |
| DefaultsHH_rate | 4e-06 | 1.97037e-07 | 9e-06 | 1.26369e-07 | 1.33234e-05 |
| HH_Deposit_mean | 1 | 1.42817 | 1.3456 | 0.766846 | 4.54062 |
| Output_mean | 100 | 21.9082 | 174.24 | 13.8584 | 310.007 |
| PriceDispersion_mean | 0.0036 | 0.000316539 | 0.0144 | 0.000170713 | 0.0184873 |
| Transfers_mean | 4 | 0.159777 | 5.3824 | 0.0945787 | 9.63676 |

## 3) SA Results (newly implausible share)
| Component | Reduction | Points | Newly implausible | Share |
|---|---:|---:|---:|---:|
| CU | 10% | 165 | 3 | 0.0182 |
| CU | 20% | 165 | 9 | 0.0545 |
| CU | 30% | 165 | 9 | 0.0545 |
| CU | 40% | 165 | 12 | 0.0727 |
| EV | 10% | 165 | 9 | 0.0545 |
| EV | 20% | 165 | 9 | 0.0545 |
| EV | 30% | 165 | 15 | 0.0909 |
| EV | 40% | 165 | 18 | 0.1091 |
| MD | 10% | 165 | 27 | 0.1636 |
| MD | 20% | 165 | 36 | 0.2182 |
| MD | 30% | 165 | 75 | 0.4545 |
| MD | 40% | 165 | 108 | 0.6545 |
| OU | 10% | 165 | 18 | 0.1091 |
| OU | 20% | 165 | 30 | 0.1818 |
| OU | 30% | 165 | 36 | 0.2182 |
| OU | 40% | 165 | 57 | 0.3455 |

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
