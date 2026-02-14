# Uncertainty Sensitivity Report (2026-02-13)

## 1) Setup
- Dataset: `C:\Users\Alexander\Desktop\Education\Aspa\code\output_research_core_v3_tuned2\02_wave2\lhs_runs_wave2.csv`
- Implausibility threshold: `2.75`
- SA domain: `nroy`
- Baseline points used: `102`
- Reductions: `10%, 20%, 30%, 40%`

## 2) Baseline Uncertainty Decomposition
| Metric | OU var | EV var | MD var | CU var (median) | Total var (median) |
|---|---:|---:|---:|---:|---:|
| AvgPrice_mean | 0.0025 | 0.000697854 | 0.0230992 | 0.000372317 | 0.0266694 |
| BankBailedOut_share | 0.0004 | 0.000162037 | 0.00350638 | 8.59266e-05 | 0.00415434 |
| BankBailoutAmount_mean | 0.04 | 0.0574222 | 0.209697 | 0.0286428 | 0.335762 |
| BankResolutionHaircut_mean | 0.0001 | 0.000221548 | 0.00155839 | 0.000151807 | 0.00203175 |
| BankResolved_share | 0.0009 | 0.00169444 | 0.00973994 | 0.00109314 | 0.0134275 |
| Consumption_mean | 100 | 81.7554 | 678.835 | 74.2719 | 934.862 |
| CreditRejections_mean | 25 | 20.0658 | 400 | 13.2634 | 458.329 |
| DefaultsHH_rate | 4e-06 | 1.43426e-06 | 3.50638e-05 | 1.33743e-06 | 4.18355e-05 |
| Employment_mean | 25 | 35.9928 | 169.709 | 32.6504 | 263.352 |
| HH_Deposit_mean | 1 | 1.22163 | 5.24243 | 2.5662 | 10.0303 |
| InventoryGap_mean | 0.0625 | 0.0445204 | 0.1225 | 0.101221 | 0.330741 |
| Output_mean | 100 | 83.6104 | 678.835 | 65.0675 | 927.513 |
| PriceDispersion_mean | 0.0036 | 0.000433805 | 0.0144 | 0.000226904 | 0.0186607 |
| Transfers_mean | 4 | 0.538264 | 20.9697 | 0.521148 | 26.0291 |

## 3) SA Results (newly implausible share)
| Component | Reduction | Points | Newly implausible | Share |
|---|---:|---:|---:|---:|
| CU | 10% | 102 | 3 | 0.0294 |
| CU | 20% | 102 | 18 | 0.1765 |
| CU | 30% | 102 | 18 | 0.1765 |
| CU | 40% | 102 | 21 | 0.2059 |
| EV | 10% | 102 | 6 | 0.0588 |
| EV | 20% | 102 | 12 | 0.1176 |
| EV | 30% | 102 | 12 | 0.1176 |
| EV | 40% | 102 | 12 | 0.1176 |
| MD | 10% | 102 | 12 | 0.1176 |
| MD | 20% | 102 | 15 | 0.1471 |
| MD | 30% | 102 | 21 | 0.2059 |
| MD | 40% | 102 | 30 | 0.2941 |
| OU | 10% | 102 | 6 | 0.0588 |
| OU | 20% | 102 | 9 | 0.0882 |
| OU | 30% | 102 | 12 | 0.1176 |
| OU | 40% | 102 | 12 | 0.1176 |

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
