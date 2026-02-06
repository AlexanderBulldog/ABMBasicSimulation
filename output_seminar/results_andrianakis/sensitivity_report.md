# Uncertainty Sensitivity Report (2026-02-06)

## 1) Setup
- Dataset: `output_seminar\datasets\lhs_runs.csv`
- Implausibility threshold: `3.0`
- SA domain: `nroy`
- Baseline points used: `406`
- Reductions: `10%, 20%, 30%, 40%`

## 2) Baseline Uncertainty Decomposition
| Metric | OU var | EV var | MD var | CU var (median) | Total var (median) |
|---|---:|---:|---:|---:|---:|
| AvgPrice_mean | 0.0025 | 0.000173339 | 0.01 | 0.000134318 | 0.0128077 |
| BankBailedOut_share | 0.0004 | 4.44444e-05 | 0.0009 | 3.57197e-05 | 0.00138016 |
| BankBailoutAmount_mean | 0.04 | 0.00471906 | 0.09 | 0.00254199 | 0.137261 |
| BankResolutionHaircut_mean | 0.0001 | 6.67298e-05 | 0.0004 | 4.15758e-05 | 0.000608306 |
| BankResolved_share | 0.0009 | 0.00138963 | 0.0025 | 0.00087822 | 0.00566785 |
| Bank_Equity_mean | 56.25 | 30.4098 | 100 | 3.42523 | 190.085 |
| Consumption_mean | 100 | 434.152 | 225 | 178.455 | 937.607 |
| DefaultsHH_rate | 4e-06 | 3.94963e-07 | 9e-06 | 3.39317e-07 | 1.37343e-05 |
| Employment_mean | 25 | 206.225 | 56.25 | 104.516 | 391.991 |
| HH_Deposit_mean | 1 | 0.64037 | 2.25 | 0.068195 | 3.95857 |
| Output_mean | 100 | 446.355 | 225 | 203.226 | 974.581 |
| Transfers_mean | 4 | 3.46003 | 9 | 1.47326 | 17.9333 |

## 3) SA Results (newly implausible share)
| Component | Reduction | Points | Newly implausible | Share |
|---|---:|---:|---:|---:|
| CU | 10% | 406 | 0 | 0.0000 |
| CU | 20% | 406 | 0 | 0.0000 |
| CU | 30% | 406 | 0 | 0.0000 |
| CU | 40% | 406 | 3 | 0.0074 |
| EV | 10% | 406 | 0 | 0.0000 |
| EV | 20% | 406 | 3 | 0.0074 |
| EV | 30% | 406 | 5 | 0.0123 |
| EV | 40% | 406 | 24 | 0.0591 |
| MD | 10% | 406 | 9 | 0.0222 |
| MD | 20% | 406 | 45 | 0.1108 |
| MD | 30% | 406 | 62 | 0.1527 |
| MD | 40% | 406 | 78 | 0.1921 |
| OU | 10% | 406 | 0 | 0.0000 |
| OU | 20% | 406 | 6 | 0.0148 |
| OU | 30% | 406 | 12 | 0.0296 |
| OU | 40% | 406 | 17 | 0.0419 |

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
