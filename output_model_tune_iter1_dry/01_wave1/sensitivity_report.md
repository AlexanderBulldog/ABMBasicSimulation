# Uncertainty Sensitivity Report (2026-02-12)

## 1) Setup
- Dataset: `C:\Users\smart\OneDrive\Рабочий стол\Аспирантура\code\ABMBasicSimulation\output_model_tune_iter1_dry\01_wave1\lhs_runs.csv`
- Implausibility threshold: `3.0`
- SA domain: `nroy`
- Baseline points used: `39`
- Reductions: `10%, 20%, 30%, 40%`

## 2) Baseline Uncertainty Decomposition
| Metric | OU var | EV var | MD var | CU var (median) | Total var (median) |
|---|---:|---:|---:|---:|---:|
| AvgPrice_mean | 0.0025 | 0.000255598 | 0.0786238 | 0.000122338 | 0.0815017 |
| BankBailedOut_share | 0.0004 | 0.000388889 | 0.0004 | 0.000117541 | 0.00130643 |
| BankBailoutAmount_mean | 0.04 | 0.0513555 | 0.04 | 0.00825738 | 0.139613 |
| BankResolutionHaircut_mean | 0.0001 | 0.000170771 | 0.0004 | 3.49559e-05 | 0.000705726 |
| BankResolved_share | 0.0009 | 0.00101852 | 0.00790123 | 0.000414528 | 0.0102343 |
| Bank_Equity_mean | 56.25 | 25.3612 | 36.1116 | 8.33756 | 126.06 |
| Consumption_mean | 100 | 55.7047 | 684.054 | 19.5539 | 859.313 |
| CreditRejections_mean | 1 | 24.5872 | 7779.24 | 9.80834 | 7814.64 |
| DefaultsHH_rate | 4e-06 | 9.18519e-07 | 1.97531e-05 | 3.7652e-07 | 2.50481e-05 |
| Employment_mean | 25 | 30.6612 | 677.734 | 13.612 | 747.008 |
| HH_Deposit_mean | 1 | 0.396942 | 2.03755 | 0.0273901 | 3.46188 |
| InventoryGap_mean | 0.0625 | 0.423143 | 0.0770443 | 0.0503596 | 0.613047 |
| Output_mean | 100 | 57.309 | 1004.4 | 21.5699 | 1183.28 |
| PriceDispersion_mean | 0.0036 | 0.000286056 | 0.00237096 | 0.000150474 | 0.00640749 |
| Transfers_mean | 4 | 0.503545 | 9.58937 | 0.197896 | 14.2908 |

## 3) SA Results (newly implausible share)
| Component | Reduction | Points | Newly implausible | Share |
|---|---:|---:|---:|---:|
| CU | 10% | 39 | 0 | 0.0000 |
| CU | 20% | 39 | 0 | 0.0000 |
| CU | 30% | 39 | 0 | 0.0000 |
| CU | 40% | 39 | 0 | 0.0000 |
| EV | 10% | 39 | 0 | 0.0000 |
| EV | 20% | 39 | 0 | 0.0000 |
| EV | 30% | 39 | 0 | 0.0000 |
| EV | 40% | 39 | 0 | 0.0000 |
| MD | 10% | 39 | 0 | 0.0000 |
| MD | 20% | 39 | 0 | 0.0000 |
| MD | 30% | 39 | 0 | 0.0000 |
| MD | 40% | 39 | 3 | 0.0769 |
| OU | 10% | 39 | 0 | 0.0000 |
| OU | 20% | 39 | 0 | 0.0000 |
| OU | 30% | 39 | 0 | 0.0000 |
| OU | 40% | 39 | 0 | 0.0000 |

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
