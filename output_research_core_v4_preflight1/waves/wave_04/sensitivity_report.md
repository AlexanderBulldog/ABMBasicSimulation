# Uncertainty Sensitivity Report (2026-02-14)

## 1) Setup
- Dataset: `C:\Users\Alexander\Desktop\Education\Aspa\code\output_research_core_v4_preflight1\waves\wave_04\lhs_runs_wave4.csv`
- Implausibility threshold: `2.75`
- SA domain: `nroy`
- Baseline points used: `504`
- Reductions: `10%, 20%, 30%, 40%`

## 2) Baseline Uncertainty Decomposition
| Metric | OU var | EV var | MD var | CU var (median) | Total var (median) |
|---|---:|---:|---:|---:|---:|
| AvgPrice_mean | 0.0025 | 0.000515431 | 0.00304287 | 0.000412688 | 0.00647099 |
| BankBailedOut_share | 0.0004 | 5.92593e-05 | 0.000461897 | 2.85301e-05 | 0.000949686 |
| BankBailoutAmount_mean | 0.04 | 0.00866567 | 0.0276235 | 0.00392703 | 0.0802162 |
| BankResolutionHaircut_mean | 0.0001 | 9.01278e-05 | 0.000205288 | 4.60722e-05 | 0.000441488 |
| BankResolved_share | 0.0009 | 0.000725926 | 0.00128305 | 0.000378711 | 0.00328768 |
| Consumption_mean | 100 | 66.3931 | 89.4233 | 59.398 | 315.214 |
| CreditRejections_mean | 100 | 15.0983 | 205.288 | 9.36132 | 329.747 |
| DefaultsHH_rate | 4e-06 | 3.33333e-07 | 4.61897e-06 | 5.52585e-07 | 9.50489e-06 |
| HH_Deposit_mean | 1 | 0.264107 | 0.690587 | 0.0415256 | 1.99622 |
| Output_mean | 100 | 52.82 | 89.4233 | 43.1258 | 285.369 |
| PriceDispersion_mean | 0.0036 | 0.000324457 | 0.00739035 | 0.000260593 | 0.0115754 |
| Transfers_mean | 4 | 0.511436 | 2.76235 | 0.474176 | 7.74796 |

## 3) SA Results (newly implausible share)
| Component | Reduction | Points | Newly implausible | Share |
|---|---:|---:|---:|---:|
| CU | 10% | 504 | 6 | 0.0119 |
| CU | 20% | 504 | 18 | 0.0357 |
| CU | 30% | 504 | 21 | 0.0417 |
| CU | 40% | 504 | 36 | 0.0714 |
| EV | 10% | 504 | 6 | 0.0119 |
| EV | 20% | 504 | 18 | 0.0357 |
| EV | 30% | 504 | 24 | 0.0476 |
| EV | 40% | 504 | 45 | 0.0893 |
| MD | 10% | 504 | 18 | 0.0357 |
| MD | 20% | 504 | 24 | 0.0476 |
| MD | 30% | 504 | 54 | 0.1071 |
| MD | 40% | 504 | 66 | 0.1310 |
| OU | 10% | 504 | 18 | 0.0357 |
| OU | 20% | 504 | 36 | 0.0714 |
| OU | 30% | 504 | 57 | 0.1131 |
| OU | 40% | 504 | 75 | 0.1488 |

## 4) Interpretation
- Dominant component (rank at max reduction): `OU`
- Investment guidance:
  - `CU` dominant: increase design density / improve GP specification.
  - `EV` dominant: increase replicate seeds `K` or refine stochastic structure.
  - `OU` dominant: improve measurement quality / uncertainty model of observations.
  - `MD` dominant: revisit model structure or discrepancy assumptions.

## 5) Limitations
- EV is treated as metric-level scalar (not x-dependent).
- SA is local to selected baseline domain.
- MD remains expert-specified and is not inferred from data.
