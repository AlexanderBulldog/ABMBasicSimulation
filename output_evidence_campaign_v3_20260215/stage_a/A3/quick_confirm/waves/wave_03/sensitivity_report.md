# Uncertainty Sensitivity Report (2026-02-15)

## 1) Setup
- Dataset: `C:\Users\Alexander\Desktop\Education\Aspa\code\output_evidence_campaign_v3_20260215\stage_a\A3\quick_confirm\waves\wave_03\lhs_runs_wave3.csv`
- Implausibility threshold: `2.6`
- SA domain: `nroy`
- Baseline points used: `123`
- Reductions: `10%, 20%, 30%, 40%`

## 2) Baseline Uncertainty Decomposition
| Metric | OU var | EV var | MD var | CU var (median) | Total var (median) |
|---|---:|---:|---:|---:|---:|
| AvgPrice_mean | 0.0025 | 0.000622893 | 0.00868065 | 0.000422443 | 0.012226 |
| BankBailedOut_share | 0.0004 | 0.000162037 | 0.00131769 | 8.69676e-05 | 0.00196669 |
| BankBailoutAmount_mean | 0.04 | 0.0881987 | 0.0788037 | 0.056224 | 0.263226 |
| BankResolutionHaircut_mean | 0.0001 | 0.00024409 | 0.00058564 | 0.000119581 | 0.00104931 |
| BankResolved_share | 0.0009 | 0.00121528 | 0.00366025 | 0.000646147 | 0.00642168 |
| Consumption_mean | 100 | 46.1153 | 255.105 | 20.1097 | 421.33 |
| DefaultsHH_rate | 4e-06 | 3.36759e-06 | 1.31769e-05 | 3.11361e-06 | 2.36581e-05 |
| Employment_mean | 25 | 29.89 | 63.7762 | 11.4485 | 130.115 |
| HH_Deposit_mean | 1 | 0.225093 | 1.97009 | 0.0169907 | 3.21218 |
| Output_mean | 100 | 48.5244 | 255.105 | 20.6692 | 424.298 |
| PriceDispersion_mean | 0.0036 | 0.000413775 | 0.021083 | 0.000288019 | 0.0253848 |
| Transfers_mean | 4 | 0.336328 | 7.88037 | 0.152572 | 12.3693 |

## 3) SA Results (newly implausible share)
| Component | Reduction | Points | Newly implausible | Share |
|---|---:|---:|---:|---:|
| CU | 10% | 123 | 0 | 0.0000 |
| CU | 20% | 123 | 6 | 0.0488 |
| CU | 30% | 123 | 15 | 0.1220 |
| CU | 40% | 123 | 18 | 0.1463 |
| EV | 10% | 123 | 9 | 0.0732 |
| EV | 20% | 123 | 24 | 0.1951 |
| EV | 30% | 123 | 30 | 0.2439 |
| EV | 40% | 123 | 39 | 0.3171 |
| MD | 10% | 123 | 27 | 0.2195 |
| MD | 20% | 123 | 42 | 0.3415 |
| MD | 30% | 123 | 60 | 0.4878 |
| MD | 40% | 123 | 78 | 0.6341 |
| OU | 10% | 123 | 6 | 0.0488 |
| OU | 20% | 123 | 15 | 0.1220 |
| OU | 30% | 123 | 30 | 0.2439 |
| OU | 40% | 123 | 33 | 0.2683 |

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
