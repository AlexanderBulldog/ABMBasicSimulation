# Uncertainty Sensitivity Report (2026-02-12)

## 1) Setup
- Dataset: `C:\Users\smart\OneDrive\Рабочий стол\Аспирантура\code\ABMBasicSimulation\output_model_tune_iter1_dry\05_confirmatory\02_wave2\lhs_runs_wave2.csv`
- Implausibility threshold: `3.0`
- SA domain: `nroy`
- Baseline points used: `45`
- Reductions: `10%, 20%, 30%, 40%`

## 2) Baseline Uncertainty Decomposition
| Metric | OU var | EV var | MD var | CU var (median) | Total var (median) |
|---|---:|---:|---:|---:|---:|
| AvgPrice_mean | 0.0025 | 0.000298799 | 0.0257448 | 0.000168445 | 0.028712 |
| BankBailoutAmount_mean | 0.04 | 0.00186871 | 0.0324 | 0.0621427 | 0.136411 |
| BankResolutionHaircut_mean | 0.0001 | 0.000249406 | 0.000303109 | 0.000120174 | 0.000772689 |
| BankResolved_share | 0.0009 | 0.00134259 | 0.009025 | 0.000479799 | 0.0117474 |
| Bank_Equity_mean | 56.25 | 20.7243 | 33.5126 | 2.80682 | 113.294 |
| Consumption_mean | 100 | 28.5098 | 688.325 | 9.80696 | 826.641 |
| CreditRejections_mean | 1 | 15.4125 | 6845.91 | 9.65811 | 6871.98 |
| DefaultsHH_rate | 4e-06 | 8.4537e-07 | 3.54025e-05 | 3.27797e-07 | 4.05757e-05 |
| Employment_mean | 25 | 17.3361 | 453.051 | 5.79033 | 501.178 |
| HH_Deposit_mean | 1 | 0.65359 | 1.75472 | 0.0525474 | 3.46086 |
| InventoryGap_mean | 0.0625 | 0.16278 | 0.151106 | 0.0713328 | 0.447718 |
| Output_mean | 100 | 51.3187 | 945.128 | 13.686 | 1110.13 |
| PriceDispersion_mean | 0.0036 | 0.000147769 | 0.00152411 | 0.000277839 | 0.00554972 |
| Transfers_mean | 4 | 0.291327 | 6.6905 | 0.0907449 | 11.0726 |

## 3) SA Results (newly implausible share)
| Component | Reduction | Points | Newly implausible | Share |
|---|---:|---:|---:|---:|
| CU | 10% | 45 | 0 | 0.0000 |
| CU | 20% | 45 | 3 | 0.0667 |
| CU | 30% | 45 | 3 | 0.0667 |
| CU | 40% | 45 | 3 | 0.0667 |
| EV | 10% | 45 | 0 | 0.0000 |
| EV | 20% | 45 | 0 | 0.0000 |
| EV | 30% | 45 | 0 | 0.0000 |
| EV | 40% | 45 | 3 | 0.0667 |
| MD | 10% | 45 | 3 | 0.0667 |
| MD | 20% | 45 | 3 | 0.0667 |
| MD | 30% | 45 | 6 | 0.1333 |
| MD | 40% | 45 | 6 | 0.1333 |
| OU | 10% | 45 | 0 | 0.0000 |
| OU | 20% | 45 | 3 | 0.0667 |
| OU | 30% | 45 | 6 | 0.1333 |
| OU | 40% | 45 | 6 | 0.1333 |

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
