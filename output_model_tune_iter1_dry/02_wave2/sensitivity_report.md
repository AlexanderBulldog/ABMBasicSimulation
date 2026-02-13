# Uncertainty Sensitivity Report (2026-02-12)

## 1) Setup
- Dataset: `C:\Users\smart\OneDrive\Рабочий стол\Аспирантура\code\ABMBasicSimulation\output_model_tune_iter1_dry\02_wave2\lhs_runs_wave2.csv`
- Implausibility threshold: `3.0`
- SA domain: `nroy`
- Baseline points used: `45`
- Reductions: `10%, 20%, 30%, 40%`

## 2) Baseline Uncertainty Decomposition
| Metric | OU var | EV var | MD var | CU var (median) | Total var (median) |
|---|---:|---:|---:|---:|---:|
| AvgPrice_mean | 0.0025 | 0.000242628 | 0.0636853 | 0.000126677 | 0.0665546 |
| BankBailedOut_share | 0.0004 | 0.00012037 | 0.000324 | 0.000134545 | 0.000978915 |
| BankBailoutAmount_mean | 0.04 | 0.0186697 | 0.0324 | 0.00228256 | 0.0933523 |
| BankResolutionHaircut_mean | 0.0001 | 0.000346541 | 0.000324 | 0.000163838 | 0.000934379 |
| BankResolved_share | 0.0009 | 0.000666667 | 0.0064 | 0.000473967 | 0.00844063 |
| Consumption_mean | 100 | 48.9699 | 554.084 | 20.0827 | 723.137 |
| CreditRejections_mean | 1 | 27.4218 | 6301.18 | 21.588 | 6351.19 |
| DefaultsHH_rate | 4e-06 | 3.7037e-07 | 1.6e-05 | 9.87453e-07 | 2.13578e-05 |
| Employment_mean | 25 | 33.1996 | 548.965 | 14.4524 | 621.617 |
| HH_Deposit_mean | 1 | 0.468071 | 1.65042 | 0.0197528 | 3.13824 |
| InventoryGap_mean | 0.0625 | 0.927646 | 0.0624059 | 4.09277 | 5.14532 |
| Output_mean | 100 | 35.1138 | 813.563 | 22.2256 | 970.902 |
| PriceDispersion_mean | 0.0036 | 8.19558e-05 | 0.00192048 | 6.95226e-05 | 0.00567196 |
| Transfers_mean | 4 | 0.464641 | 7.76739 | 0.213282 | 12.4453 |

## 3) SA Results (newly implausible share)
| Component | Reduction | Points | Newly implausible | Share |
|---|---:|---:|---:|---:|
| CU | 10% | 45 | 0 | 0.0000 |
| CU | 20% | 45 | 0 | 0.0000 |
| CU | 30% | 45 | 0 | 0.0000 |
| CU | 40% | 45 | 0 | 0.0000 |
| EV | 10% | 45 | 0 | 0.0000 |
| EV | 20% | 45 | 0 | 0.0000 |
| EV | 30% | 45 | 0 | 0.0000 |
| EV | 40% | 45 | 0 | 0.0000 |
| MD | 10% | 45 | 0 | 0.0000 |
| MD | 20% | 45 | 0 | 0.0000 |
| MD | 30% | 45 | 6 | 0.1333 |
| MD | 40% | 45 | 15 | 0.3333 |
| OU | 10% | 45 | 0 | 0.0000 |
| OU | 20% | 45 | 0 | 0.0000 |
| OU | 30% | 45 | 0 | 0.0000 |
| OU | 40% | 45 | 0 | 0.0000 |

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
