# Uncertainty Sensitivity Report (2026-02-06)

## 1) Setup
- Dataset: `C:\Users\Alexander\Desktop\Education\Aspa\code\output_research_core_smoke\02_wave2\lhs_runs_wave2.csv`
- Implausibility threshold: `3.0`
- SA domain: `nroy`
- Baseline points used: `60`
- Reductions: `10%, 20%, 30%, 40%`

## 2) Baseline Uncertainty Decomposition
| Metric | OU var | EV var | MD var | CU var (median) | Total var (median) |
|---|---:|---:|---:|---:|---:|
| AvgPrice_mean | 0.0025 | 0.000140323 | 0.01 | 0.000158283 | 0.0127986 |
| Bank_Equity_mean | 56.25 | 35.9494 | 100 | 105.567 | 297.767 |
| Consumption_mean | 100 | 14.2418 | 225 | 14.243 | 353.485 |
| DefaultsHH_rate | 4e-06 | 1.84259e-07 | 9e-06 | 8.11742e-08 | 1.32654e-05 |
| Employment_mean | 25 | 7.74933 | 56.25 | 8.85793 | 97.8573 |
| HH_Deposit_mean | 1 | 0.472847 | 2.25 | 0.15482 | 3.87767 |
| Output_mean | 100 | 12.8635 | 225 | 12.1676 | 350.031 |
| Transfers_mean | 4 | 0.100636 | 9 | 0.114369 | 13.215 |

## 3) SA Results (newly implausible share)
| Component | Reduction | Points | Newly implausible | Share |
|---|---:|---:|---:|---:|
| CU | 10% | 60 | 0 | 0.0000 |
| CU | 20% | 60 | 0 | 0.0000 |
| CU | 30% | 60 | 0 | 0.0000 |
| CU | 40% | 60 | 0 | 0.0000 |
| EV | 10% | 60 | 0 | 0.0000 |
| EV | 20% | 60 | 0 | 0.0000 |
| EV | 30% | 60 | 0 | 0.0000 |
| EV | 40% | 60 | 0 | 0.0000 |
| MD | 10% | 60 | 0 | 0.0000 |
| MD | 20% | 60 | 0 | 0.0000 |
| MD | 30% | 60 | 0 | 0.0000 |
| MD | 40% | 60 | 0 | 0.0000 |
| OU | 10% | 60 | 0 | 0.0000 |
| OU | 20% | 60 | 0 | 0.0000 |
| OU | 30% | 60 | 0 | 0.0000 |
| OU | 40% | 60 | 0 | 0.0000 |

## 4) Interpretation
- Dominant component (rank at max reduction): `CU, EV, MD, OU`
- Investment guidance:
  - `CU` dominant: increase design density / improve GP specification.
  - `EV` dominant: increase replicate seeds `K` or refine stochastic structure.
  - `OU` dominant: improve measurement quality / uncertainty model of observations.
  - `MD` dominant: revisit model structure or discrepancy assumptions.

## 5) Limitations
- EV is treated as metric-level scalar (not x-dependent).
- SA is local to selected baseline domain.
- MD remains expert-specified and is not inferred from data.
