# Uncertainty Sensitivity Report (2026-02-14)

## 1) Setup
- Dataset: `C:\Users\Alexander\Desktop\Education\Aspa\code\output_quality_campaign_night_20260214_021850\exp04_wave1_from_credit_bounds\waves\wave_05\lhs_runs_wave5.csv`
- Implausibility threshold: `2.75`
- SA domain: `nroy`
- Baseline points used: `414`
- Reductions: `10%, 20%, 30%, 40%`

## 2) Baseline Uncertainty Decomposition
| Metric | OU var | EV var | MD var | CU var (median) | Total var (median) |
|---|---:|---:|---:|---:|---:|
| AvgPrice_mean | 0.0025 | 0.000419345 | 0.00373673 | 0.000206413 | 0.00686249 |
| BankResolutionHaircut_mean | 0.0001 | 2.03051e-05 | 0.000252099 | 1.24495e-05 | 0.000384853 |
| CreditRejections_mean | 25 | 12.3861 | 252.099 | 7.69407 | 297.179 |
| DefaultsHH_rate | 4e-06 | 5.22109e-07 | 5.67222e-06 | 2.88305e-07 | 1.04826e-05 |
| Output_mean | 100 | 40.7252 | 109.814 | 26.6783 | 277.218 |

## 3) SA Results (newly implausible share)
| Component | Reduction | Points | Newly implausible | Share |
|---|---:|---:|---:|---:|
| CU | 10% | 414 | 3 | 0.0072 |
| CU | 20% | 414 | 3 | 0.0072 |
| CU | 30% | 414 | 6 | 0.0145 |
| CU | 40% | 414 | 12 | 0.0290 |
| EV | 10% | 414 | 3 | 0.0072 |
| EV | 20% | 414 | 9 | 0.0217 |
| EV | 30% | 414 | 12 | 0.0290 |
| EV | 40% | 414 | 15 | 0.0362 |
| MD | 10% | 414 | 12 | 0.0290 |
| MD | 20% | 414 | 30 | 0.0725 |
| MD | 30% | 414 | 45 | 0.1087 |
| MD | 40% | 414 | 57 | 0.1377 |
| OU | 10% | 414 | 12 | 0.0290 |
| OU | 20% | 414 | 27 | 0.0652 |
| OU | 30% | 414 | 39 | 0.0942 |
| OU | 40% | 414 | 51 | 0.1232 |

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
