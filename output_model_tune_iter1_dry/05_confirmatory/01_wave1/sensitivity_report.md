# Uncertainty Sensitivity Report (2026-02-12)

## 1) Setup
- Dataset: `C:\Users\smart\OneDrive\Рабочий стол\Аспирантура\code\ABMBasicSimulation\output_model_tune_iter1_dry\05_confirmatory\01_wave1\lhs_runs.csv`
- Implausibility threshold: `3.0`
- SA domain: `nroy`
- Baseline points used: `48`
- Reductions: `10%, 20%, 30%, 40%`

## 2) Baseline Uncertainty Decomposition
| Metric | OU var | EV var | MD var | CU var (median) | Total var (median) |
|---|---:|---:|---:|---:|---:|
| AvgPrice_mean | 0.0025 | 0.000153727 | 0.0317836 | 0.000147943 | 0.0345853 |
| BankBailedOut_share | 0.0004 | 0.000296296 | 0.0004 | 9.91344e-06 | 0.00110621 |
| BankBailoutAmount_mean | 0.04 | 0.0499573 | 0.04 | 0.00780101 | 0.137758 |
| BankResolutionHaircut_mean | 0.0001 | 0.000221762 | 0.000374209 | 3.97211e-05 | 0.000735692 |
| BankResolved_share | 0.0009 | 0.000833333 | 0.011142 | 0.000932157 | 0.0138075 |
| Bank_Equity_mean | 56.25 | 113.899 | 41.3736 | 10.8826 | 222.405 |
| Consumption_mean | 100 | 59.0481 | 849.784 | 189.383 | 1198.21 |
| CreditRejections_mean | 1 | 11.2565 | 8451.74 | 6.6798 | 8470.67 |
| DefaultsHH_rate | 4e-06 | 1.00185e-06 | 4.37068e-05 | 1.2005e-06 | 4.99091e-05 |
| Employment_mean | 25 | 35.2168 | 559.322 | 93.8263 | 713.366 |
| HH_Deposit_mean | 1 | 0.529982 | 2.16632 | 0.0969365 | 3.79324 |
| InventoryGap_mean | 0.0625 | 1.23215 | 0.18655 | 3.94353 | 5.42473 |
| Output_mean | 100 | 81.4608 | 1166.82 | 116.569 | 1464.85 |
| PriceDispersion_mean | 0.0036 | 0.00023514 | 0.00188162 | 0.000196942 | 0.0059137 |
| Transfers_mean | 4 | 0.547423 | 8.25988 | 1.54793 | 14.3552 |

## 3) SA Results (newly implausible share)
| Component | Reduction | Points | Newly implausible | Share |
|---|---:|---:|---:|---:|
| CU | 10% | 48 | 3 | 0.0625 |
| CU | 20% | 48 | 3 | 0.0625 |
| CU | 30% | 48 | 3 | 0.0625 |
| CU | 40% | 48 | 3 | 0.0625 |
| EV | 10% | 48 | 0 | 0.0000 |
| EV | 20% | 48 | 0 | 0.0000 |
| EV | 30% | 48 | 0 | 0.0000 |
| EV | 40% | 48 | 0 | 0.0000 |
| MD | 10% | 48 | 0 | 0.0000 |
| MD | 20% | 48 | 3 | 0.0625 |
| MD | 30% | 48 | 6 | 0.1250 |
| MD | 40% | 48 | 12 | 0.2500 |
| OU | 10% | 48 | 0 | 0.0000 |
| OU | 20% | 48 | 0 | 0.0000 |
| OU | 30% | 48 | 3 | 0.0625 |
| OU | 40% | 48 | 3 | 0.0625 |

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
