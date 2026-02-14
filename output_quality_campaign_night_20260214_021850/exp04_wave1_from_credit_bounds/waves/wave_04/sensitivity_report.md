# Uncertainty Sensitivity Report (2026-02-14)

## 1) Setup
- Dataset: `C:\Users\Alexander\Desktop\Education\Aspa\code\output_quality_campaign_night_20260214_021850\exp04_wave1_from_credit_bounds\waves\wave_04\lhs_runs_wave4.csv`
- Implausibility threshold: `2.75`
- SA domain: `nroy`
- Baseline points used: `204`
- Reductions: `10%, 20%, 30%, 40%`

## 2) Baseline Uncertainty Decomposition
| Metric | OU var | EV var | MD var | CU var (median) | Total var (median) |
|---|---:|---:|---:|---:|---:|
| AvgPrice_mean | 0.0025 | 0.000435376 | 0.00373673 | 0.000212976 | 0.00688509 |
| BankResolutionHaircut_mean | 0.0001 | 2.41698e-05 | 0.000252099 | 1.32766e-05 | 0.000389545 |
| BankResolved_share | 0.0009 | 0.00155102 | 0.00157562 | 0.000892509 | 0.00491915 |
| CreditRejections_mean | 25 | 12.6421 | 252.099 | 8.84167 | 298.583 |
| DefaultsHH_rate | 4e-06 | 4.40476e-07 | 5.67222e-06 | 2.04025e-07 | 1.03167e-05 |
| HH_Deposit_mean | 1 | 2.44977 | 0.84806 | 2.34372 | 6.64155 |
| Output_mean | 100 | 35.7461 | 109.814 | 19.4205 | 264.981 |

## 3) SA Results (newly implausible share)
| Component | Reduction | Points | Newly implausible | Share |
|---|---:|---:|---:|---:|
| CU | 10% | 204 | 21 | 0.1029 |
| CU | 20% | 204 | 39 | 0.1912 |
| CU | 30% | 204 | 48 | 0.2353 |
| CU | 40% | 204 | 63 | 0.3088 |
| EV | 10% | 204 | 18 | 0.0882 |
| EV | 20% | 204 | 42 | 0.2059 |
| EV | 30% | 204 | 51 | 0.2500 |
| EV | 40% | 204 | 69 | 0.3382 |
| MD | 10% | 204 | 24 | 0.1176 |
| MD | 20% | 204 | 36 | 0.1765 |
| MD | 30% | 204 | 45 | 0.2206 |
| MD | 40% | 204 | 72 | 0.3529 |
| OU | 10% | 204 | 24 | 0.1176 |
| OU | 20% | 204 | 36 | 0.1765 |
| OU | 30% | 204 | 48 | 0.2353 |
| OU | 40% | 204 | 72 | 0.3529 |

## 4) Interpretation
- Dominant component (rank at max reduction): `MD, OU`
- Investment guidance:
  - `CU` dominant: increase design density / improve GP specification.
  - `EV` dominant: increase replicate seeds `K` or refine stochastic structure.
  - `OU` dominant: improve measurement quality / uncertainty model of observations.
  - `MD` dominant: revisit model structure or discrepancy assumptions.

## 5) Limitations
- EV is treated as metric-level scalar (not x-dependent).
- SA is local to selected baseline domain.
- MD remains expert-specified and is not inferred from data.
