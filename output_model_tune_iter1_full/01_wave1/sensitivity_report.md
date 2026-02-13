# Uncertainty Sensitivity Report (2026-02-12)

## 1) Setup
- Dataset: `C:\Users\smart\OneDrive\Рабочий стол\Аспирантура\code\ABMBasicSimulation\output_model_tune_iter1_full\01_wave1\lhs_runs.csv`
- Implausibility threshold: `3.0`
- SA domain: `nroy`
- Baseline points used: `1141`
- Reductions: `10%, 20%, 30%, 40%`

## 2) Baseline Uncertainty Decomposition
| Metric | OU var | EV var | MD var | CU var (median) | Total var (median) |
|---|---:|---:|---:|---:|---:|
| AvgPrice_mean | 0.0025 | 0.000384268 | 0.0511731 | 0.00033596 | 0.0543933 |
| BankBailedOut_share | 0.0004 | 3.08642e-05 | 0.0004 | 1.45266e-05 | 0.000845391 |
| BankBailoutAmount_mean | 0.04 | 0.00500068 | 0.04 | 0.00274843 | 0.0877491 |
| BankResolutionHaircut_mean | 0.0001 | 3.67792e-05 | 0.000250142 | 1.91967e-05 | 0.000406118 |
| BankResolved_share | 0.0009 | 0.00037037 | 0.00197531 | 0.000227866 | 0.00347354 |
| Bank_Equity_mean | 56.25 | 33.8359 | 14.1141 | 20.5265 | 124.727 |
| Consumption_mean | 100 | 31.4689 | 1924.02 | 67.6091 | 2123.1 |
| CreditRejections_mean | 1 | 14.0233 | 9931.23 | 6.67605 | 9952.93 |
| DefaultsHH_rate | 4e-06 | 4.33128e-07 | 3.85322e-06 | 3.17364e-07 | 8.60372e-06 |
| Employment_mean | 25 | 19.4715 | 1100.03 | 30.4888 | 1174.99 |
| HH_Deposit_mean | 1 | 0.650195 | 2.25 | 0.0264083 | 3.9266 |
| Output_mean | 100 | 40.8381 | 3484.92 | 73.7157 | 3699.47 |
| PriceDispersion_mean | 0.0036 | 0.000282841 | 0.00172444 | 0.000253553 | 0.00586084 |
| Transfers_mean | 4 | 0.262929 | 23.1107 | 0.543209 | 27.9168 |

## 3) SA Results (newly implausible share)
| Component | Reduction | Points | Newly implausible | Share |
|---|---:|---:|---:|---:|
| CU | 10% | 1141 | 0 | 0.0000 |
| CU | 20% | 1141 | 0 | 0.0000 |
| CU | 30% | 1141 | 0 | 0.0000 |
| CU | 40% | 1141 | 0 | 0.0000 |
| EV | 10% | 1141 | 0 | 0.0000 |
| EV | 20% | 1141 | 0 | 0.0000 |
| EV | 30% | 1141 | 0 | 0.0000 |
| EV | 40% | 1141 | 0 | 0.0000 |
| MD | 10% | 1141 | 0 | 0.0000 |
| MD | 20% | 1141 | 0 | 0.0000 |
| MD | 30% | 1141 | 6 | 0.0053 |
| MD | 40% | 1141 | 15 | 0.0131 |
| OU | 10% | 1141 | 0 | 0.0000 |
| OU | 20% | 1141 | 0 | 0.0000 |
| OU | 30% | 1141 | 0 | 0.0000 |
| OU | 40% | 1141 | 3 | 0.0026 |

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
