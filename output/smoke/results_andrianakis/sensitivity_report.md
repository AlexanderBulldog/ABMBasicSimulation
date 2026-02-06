# Uncertainty Sensitivity Report (2026-02-06)

## 1) Setup
- Dataset: `output\smoke\datasets\lhs_smoke.csv`
- Implausibility threshold: `3.0`
- SA domain: `nroy`
- Baseline points used: `11`
- Reductions: `10%, 20%, 30%, 40%`

## 2) Baseline Uncertainty Decomposition
| Metric | OU var | EV var | MD var | CU var (median) | Total var (median) |
|---|---:|---:|---:|---:|---:|
| AvgPrice_mean | 0.0025 | 1.14269e-05 | 0.01 | 1.45049e-06 | 0.0125129 |
| BankBailedOut_share | 0.0004 | 0.000555556 | 0.0009 | 0.000333436 | 0.00218899 |
| BankBailoutAmount_mean | 0.04 | 0.0277009 | 0.09 | 0.0284437 | 0.186145 |
| BankResolutionHaircut_mean | 0.0001 | 0.000115768 | 0.0004 | 0.000176065 | 0.000791833 |
| BankResolved_share | 0.0009 | 0.000555556 | 0.0025 | 0.000616666 | 0.00457222 |
| Bank_Equity_mean | 56.25 | 17.7296 | 100 | 2.23392 | 176.214 |
| Consumption_mean | 100 | 26.5682 | 225 | 19.8992 | 371.467 |
| DefaultsHH_rate | 4e-06 | 1.125e-06 | 9e-06 | 1.38559e-08 | 1.41389e-05 |
| Employment_mean | 25 | 12.9611 | 56.25 | 2.20958 | 96.4207 |
| HH_Deposit_mean | 1 | 0.215443 | 2.25 | 0.00551378 | 3.47096 |
| Output_mean | 100 | 46.5597 | 225 | 24.9951 | 396.555 |
| Transfers_mean | 4 | 0.195774 | 9 | 0.0384299 | 13.2342 |

## 3) SA Results (newly implausible share)
| Component | Reduction | Points | Newly implausible | Share |
|---|---:|---:|---:|---:|
| CU | 10% | 11 | 0 | 0.0000 |
| CU | 20% | 11 | 0 | 0.0000 |
| CU | 30% | 11 | 0 | 0.0000 |
| CU | 40% | 11 | 0 | 0.0000 |
| EV | 10% | 11 | 0 | 0.0000 |
| EV | 20% | 11 | 0 | 0.0000 |
| EV | 30% | 11 | 0 | 0.0000 |
| EV | 40% | 11 | 0 | 0.0000 |
| MD | 10% | 11 | 0 | 0.0000 |
| MD | 20% | 11 | 4 | 0.3636 |
| MD | 30% | 11 | 4 | 0.3636 |
| MD | 40% | 11 | 4 | 0.3636 |
| OU | 10% | 11 | 0 | 0.0000 |
| OU | 20% | 11 | 0 | 0.0000 |
| OU | 30% | 11 | 0 | 0.0000 |
| OU | 40% | 11 | 0 | 0.0000 |

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
