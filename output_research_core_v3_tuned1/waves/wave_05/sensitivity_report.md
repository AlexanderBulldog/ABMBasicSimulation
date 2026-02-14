# Uncertainty Sensitivity Report (2026-02-13)

## 1) Setup
- Dataset: `C:\Users\Alexander\Desktop\Education\Aspa\code\output_research_core_v3_tuned1\waves\wave_05\lhs_runs_wave5.csv`
- Implausibility threshold: `2.75`
- SA domain: `nroy`
- Baseline points used: `303`
- Reductions: `10%, 20%, 30%, 40%`

## 2) Baseline Uncertainty Decomposition
| Metric | OU var | EV var | MD var | CU var (median) | Total var (median) |
|---|---:|---:|---:|---:|---:|
| AvgPrice_mean | 0.0025 | 0.000481836 | 0.00722468 | 0.000314843 | 0.0105214 |
| BankBailedOut_share | 0.0004 | 9.25926e-05 | 0.00109668 | 4.09528e-05 | 0.00163023 |
| BankBailoutAmount_mean | 0.04 | 0.0252472 | 0.0655863 | 0.0102907 | 0.141124 |
| BankResolutionHaircut_mean | 0.0001 | 0.000211937 | 0.000487413 | 8.95139e-05 | 0.000888865 |
| BankResolved_share | 0.0009 | 0.00156481 | 0.00304633 | 0.000977323 | 0.00648847 |
| Consumption_mean | 100 | 54.9259 | 212.317 | 42.1126 | 409.356 |
| CreditRejections_mean | 25 | 18.8018 | 487.413 | 13.5186 | 544.734 |
| DefaultsHH_rate | 4e-06 | 5.93287e-07 | 1.09668e-05 | 5.24668e-07 | 1.60848e-05 |
| Employment_mean | 25 | 30.3873 | 53.0793 | 21.0593 | 129.526 |
| HH_Deposit_mean | 1 | 1.5522 | 1.63966 | 1.26412 | 5.45598 |
| InventoryGap_mean | 0.0625 | 0.0221285 | 0.14927 | 0.0291529 | 0.263052 |
| Output_mean | 100 | 39.7967 | 212.317 | 29.5867 | 381.701 |
| PriceDispersion_mean | 0.0036 | 0.000543912 | 0.0175469 | 0.000245602 | 0.0219364 |
| Transfers_mean | 4 | 0.458051 | 6.55863 | 0.346425 | 11.3631 |

## 3) SA Results (newly implausible share)
| Component | Reduction | Points | Newly implausible | Share |
|---|---:|---:|---:|---:|
| CU | 10% | 303 | 42 | 0.1386 |
| CU | 20% | 303 | 54 | 0.1782 |
| CU | 30% | 303 | 66 | 0.2178 |
| CU | 40% | 303 | 72 | 0.2376 |
| EV | 10% | 303 | 48 | 0.1584 |
| EV | 20% | 303 | 63 | 0.2079 |
| EV | 30% | 303 | 81 | 0.2673 |
| EV | 40% | 303 | 102 | 0.3366 |
| MD | 10% | 303 | 57 | 0.1881 |
| MD | 20% | 303 | 90 | 0.2970 |
| MD | 30% | 303 | 135 | 0.4455 |
| MD | 40% | 303 | 195 | 0.6436 |
| OU | 10% | 303 | 45 | 0.1485 |
| OU | 20% | 303 | 57 | 0.1881 |
| OU | 30% | 303 | 69 | 0.2277 |
| OU | 40% | 303 | 87 | 0.2871 |

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
