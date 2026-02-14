# Uncertainty Sensitivity Report (2026-02-14)

## 1) Setup
- Dataset: `C:\Users\Alexander\Desktop\Education\Aspa\code\output_research_core_v5_preflight1\waves\wave_03\lhs_runs_wave3.csv`
- Implausibility threshold: `2.75`
- SA domain: `nroy`
- Baseline points used: `225`
- Reductions: `10%, 20%, 30%, 40%`

## 2) Baseline Uncertainty Decomposition
| Metric | OU var | EV var | MD var | CU var (median) | Total var (median) |
|---|---:|---:|---:|---:|---:|
| AvgPrice_mean | 0.0025 | 0.000606595 | 0.005929 | 0.000609799 | 0.00964539 |
| BankBailedOut_share | 0.0004 | 0.000162037 | 0.0009 | 0.000106053 | 0.00156809 |
| BankBailoutAmount_mean | 0.04 | 0.064296 | 0.053824 | 0.0998347 | 0.257955 |
| BankResolutionHaircut_mean | 0.0001 | 0.00023533 | 0.0004 | 0.000165542 | 0.000900873 |
| BankResolved_share | 0.0009 | 0.00141204 | 0.0025 | 0.000701209 | 0.00551325 |
| Consumption_mean | 100 | 61.8962 | 174.24 | 43.8721 | 380.008 |
| DefaultsHH_rate | 4e-06 | 1.19236e-06 | 9e-06 | 3.14128e-06 | 1.73336e-05 |
| HH_Deposit_mean | 1 | 0.32651 | 1.3456 | 0.029212 | 2.70132 |
| Output_mean | 100 | 60.4572 | 174.24 | 41.3984 | 376.096 |
| PriceDispersion_mean | 0.0036 | 0.000424749 | 0.0144 | 0.000390515 | 0.0188153 |
| Transfers_mean | 4 | 0.450507 | 5.3824 | 0.362076 | 10.195 |

## 3) SA Results (newly implausible share)
| Component | Reduction | Points | Newly implausible | Share |
|---|---:|---:|---:|---:|
| CU | 10% | 225 | 9 | 0.0400 |
| CU | 20% | 225 | 9 | 0.0400 |
| CU | 30% | 225 | 12 | 0.0533 |
| CU | 40% | 225 | 12 | 0.0533 |
| EV | 10% | 225 | 6 | 0.0267 |
| EV | 20% | 225 | 12 | 0.0533 |
| EV | 30% | 225 | 12 | 0.0533 |
| EV | 40% | 225 | 15 | 0.0667 |
| MD | 10% | 225 | 6 | 0.0267 |
| MD | 20% | 225 | 33 | 0.1467 |
| MD | 30% | 225 | 42 | 0.1867 |
| MD | 40% | 225 | 57 | 0.2533 |
| OU | 10% | 225 | 6 | 0.0267 |
| OU | 20% | 225 | 6 | 0.0267 |
| OU | 30% | 225 | 24 | 0.1067 |
| OU | 40% | 225 | 30 | 0.1333 |

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
