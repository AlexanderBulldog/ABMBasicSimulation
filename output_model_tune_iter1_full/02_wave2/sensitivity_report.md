# Uncertainty Sensitivity Report (2026-02-12)

## 1) Setup
- Dataset: `C:\Users\smart\OneDrive\Рабочий стол\Аспирантура\code\ABMBasicSimulation\output_model_tune_iter1_full\02_wave2\lhs_runs_wave2.csv`
- Implausibility threshold: `3.0`
- SA domain: `nroy`
- Baseline points used: `1181`
- Reductions: `10%, 20%, 30%, 40%`

## 2) Baseline Uncertainty Decomposition
| Metric | OU var | EV var | MD var | CU var (median) | Total var (median) |
|---|---:|---:|---:|---:|---:|
| AvgPrice_mean | 0.0025 | 0.000267755 | 0.0271955 | 0.000354602 | 0.0303179 |
| BankBailedOut_share | 0.0004 | 1.02881e-05 | 0.000212576 | 1.71778e-05 | 0.000640042 |
| BankBailoutAmount_mean | 0.04 | 0.00373673 | 0.0212576 | 0.00232872 | 0.0673231 |
| BankResolutionHaircut_mean | 0.0001 | 2.67529e-05 | 0.000132936 | 2.86684e-05 | 0.000288357 |
| BankResolved_share | 0.0009 | 0.00037037 | 0.00104976 | 0.000202991 | 0.00252312 |
| Bank_Equity_mean | 56.25 | 28.9311 | 7.50079 | 17.6165 | 110.298 |
| Consumption_mean | 100 | 22.2679 | 1022.5 | 67.4803 | 1212.25 |
| CreditRejections_mean | 1 | 12.8557 | 5277.86 | 8.7356 | 5300.45 |
| DefaultsHH_rate | 4e-06 | 2.18107e-07 | 2.04776e-06 | 1.80001e-07 | 6.44587e-06 |
| Employment_mean | 25 | 12.4717 | 584.6 | 29.9567 | 652.028 |
| HH_Deposit_mean | 1 | 0.513393 | 1.19574 | 0.0298692 | 2.739 |
| Output_mean | 100 | 31.0628 | 1852.03 | 63.2466 | 2046.34 |
| PriceDispersion_mean | 0.0036 | 0.000279252 | 0.00091644 | 0.000302416 | 0.00509811 |
| Transfers_mean | 4 | 0.168717 | 12.282 | 0.555097 | 17.0058 |

## 3) SA Results (newly implausible share)
| Component | Reduction | Points | Newly implausible | Share |
|---|---:|---:|---:|---:|
| CU | 10% | 1181 | 0 | 0.0000 |
| CU | 20% | 1181 | 3 | 0.0025 |
| CU | 30% | 1181 | 6 | 0.0051 |
| CU | 40% | 1181 | 6 | 0.0051 |
| EV | 10% | 1181 | 0 | 0.0000 |
| EV | 20% | 1181 | 0 | 0.0000 |
| EV | 30% | 1181 | 0 | 0.0000 |
| EV | 40% | 1181 | 0 | 0.0000 |
| MD | 10% | 1181 | 6 | 0.0051 |
| MD | 20% | 1181 | 18 | 0.0152 |
| MD | 30% | 1181 | 58 | 0.0491 |
| MD | 40% | 1181 | 91 | 0.0771 |
| OU | 10% | 1181 | 0 | 0.0000 |
| OU | 20% | 1181 | 3 | 0.0025 |
| OU | 30% | 1181 | 7 | 0.0059 |
| OU | 40% | 1181 | 16 | 0.0135 |

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
