# Uncertainty Sensitivity Report (2026-02-12)

## 1) Setup
- Dataset: `C:\Users\smart\OneDrive\Рабочий стол\Аспирантура\code\ABMBasicSimulation\output_model_tune_iter1_full\05_confirmatory\02_wave2\lhs_runs_wave2.csv`
- Implausibility threshold: `3.0`
- SA domain: `nroy`
- Baseline points used: `1161`
- Reductions: `10%, 20%, 30%, 40%`

## 2) Baseline Uncertainty Decomposition
| Metric | OU var | EV var | MD var | CU var (median) | Total var (median) |
|---|---:|---:|---:|---:|---:|
| AvgPrice_mean | 0.0025 | 0.000301443 | 0.0277195 | 0.000223398 | 0.0307443 |
| BankBailedOut_share | 0.0004 | 3.08642e-05 | 0.000212576 | 1.08454e-05 | 0.000654286 |
| BankBailoutAmount_mean | 0.04 | 0.00470284 | 0.0212576 | 0.00209673 | 0.0680572 |
| BankResolutionHaircut_mean | 0.0001 | 3.28222e-05 | 0.000132157 | 2.29848e-05 | 0.000287964 |
| BankResolved_share | 0.0009 | 0.000288066 | 0.00123201 | 0.000174591 | 0.00259467 |
| Bank_Equity_mean | 56.25 | 28.5543 | 7.56196 | 19.3347 | 111.701 |
| Consumption_mean | 100 | 33.8289 | 977.229 | 32.715 | 1143.77 |
| CreditRejections_mean | 1 | 13.4013 | 5308.51 | 7.22554 | 5330.13 |
| DefaultsHH_rate | 4e-06 | 2.08436e-07 | 2.16531e-06 | 1.63892e-07 | 6.53764e-06 |
| Employment_mean | 25 | 20.6482 | 583.034 | 19.3088 | 647.991 |
| HH_Deposit_mean | 1 | 0.612108 | 1.19574 | 0.0359412 | 2.84379 |
| Output_mean | 100 | 48.5084 | 2064.71 | 45.2507 | 2258.47 |
| PriceDispersion_mean | 0.0036 | 0.000258679 | 0.000954321 | 0.000300576 | 0.00511358 |
| Transfers_mean | 4 | 0.281643 | 11.9839 | 0.266951 | 16.5325 |

## 3) SA Results (newly implausible share)
| Component | Reduction | Points | Newly implausible | Share |
|---|---:|---:|---:|---:|
| CU | 10% | 1161 | 0 | 0.0000 |
| CU | 20% | 1161 | 3 | 0.0026 |
| CU | 30% | 1161 | 6 | 0.0052 |
| CU | 40% | 1161 | 6 | 0.0052 |
| EV | 10% | 1161 | 0 | 0.0000 |
| EV | 20% | 1161 | 3 | 0.0026 |
| EV | 30% | 1161 | 3 | 0.0026 |
| EV | 40% | 1161 | 3 | 0.0026 |
| MD | 10% | 1161 | 9 | 0.0078 |
| MD | 20% | 1161 | 24 | 0.0207 |
| MD | 30% | 1161 | 54 | 0.0465 |
| MD | 40% | 1161 | 84 | 0.0724 |
| OU | 10% | 1161 | 3 | 0.0026 |
| OU | 20% | 1161 | 6 | 0.0052 |
| OU | 30% | 1161 | 6 | 0.0052 |
| OU | 40% | 1161 | 9 | 0.0078 |

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
