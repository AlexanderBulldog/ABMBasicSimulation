# Uncertainty Sensitivity Report (2026-02-14)

## 1) Setup
- Dataset: `C:\Users\Alexander\Desktop\Education\Aspa\code\output_research_core_v6_full_20260214_182812\waves\wave_04\lhs_runs_wave4.csv`
- Implausibility threshold: `2.6`
- SA domain: `nroy`
- Baseline points used: `597`
- Reductions: `10%, 20%, 30%, 40%`

## 2) Baseline Uncertainty Decomposition
| Metric | OU var | EV var | MD var | CU var (median) | Total var (median) |
|---|---:|---:|---:|---:|---:|
| AvgPrice_mean | 0.0025 | 0.000423679 | 0.00257549 | 0.000202664 | 0.00570183 |
| BankBailedOut_share | 0.0004 | 4.82094e-05 | 0.00039095 | 2.64279e-05 | 0.000865587 |
| BankBailoutAmount_mean | 0.04 | 0.013398 | 0.0233805 | 0.00804266 | 0.0848212 |
| BankResolutionHaircut_mean | 0.0001 | 7.78518e-05 | 0.000173755 | 4.09003e-05 | 0.000392508 |
| BankResolved_share | 0.0009 | 0.00119904 | 0.00108597 | 0.000558673 | 0.00374368 |
| Consumption_mean | 100 | 45.4449 | 75.6878 | 30.2796 | 251.412 |
| DefaultsHH_rate | 4e-06 | 2.25255e-06 | 3.9095e-06 | 8.07314e-07 | 1.09694e-05 |
| HH_Deposit_mean | 1 | 0.158149 | 0.584513 | 0.0235085 | 1.76617 |
| Output_mean | 100 | 37.1842 | 75.6878 | 24.7853 | 237.657 |
| PriceDispersion_mean | 0.0036 | 0.00038468 | 0.00625519 | 0.000361903 | 0.0106018 |
| Transfers_mean | 4 | 0.312773 | 2.33805 | 0.223133 | 6.87396 |

## 3) SA Results (newly implausible share)
| Component | Reduction | Points | Newly implausible | Share |
|---|---:|---:|---:|---:|
| CU | 10% | 597 | 12 | 0.0201 |
| CU | 20% | 597 | 18 | 0.0302 |
| CU | 30% | 597 | 33 | 0.0553 |
| CU | 40% | 597 | 45 | 0.0754 |
| EV | 10% | 597 | 18 | 0.0302 |
| EV | 20% | 597 | 33 | 0.0553 |
| EV | 30% | 597 | 51 | 0.0854 |
| EV | 40% | 597 | 66 | 0.1106 |
| MD | 10% | 597 | 33 | 0.0553 |
| MD | 20% | 597 | 66 | 0.1106 |
| MD | 30% | 597 | 111 | 0.1859 |
| MD | 40% | 597 | 138 | 0.2312 |
| OU | 10% | 597 | 45 | 0.0754 |
| OU | 20% | 597 | 90 | 0.1508 |
| OU | 30% | 597 | 135 | 0.2261 |
| OU | 40% | 597 | 168 | 0.2814 |

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
