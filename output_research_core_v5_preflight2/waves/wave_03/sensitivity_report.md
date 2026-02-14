# Uncertainty Sensitivity Report (2026-02-14)

## 1) Setup
- Dataset: `C:\Users\Alexander\Desktop\Education\Aspa\code\output_research_core_v5_preflight2\waves\wave_03\lhs_runs_wave3.csv`
- Implausibility threshold: `2.6`
- SA domain: `nroy`
- Baseline points used: `246`
- Reductions: `10%, 20%, 30%, 40%`

## 2) Baseline Uncertainty Decomposition
| Metric | OU var | EV var | MD var | CU var (median) | Total var (median) |
|---|---:|---:|---:|---:|---:|
| AvgPrice_mean | 0.0025 | 0.000569588 | 0.00796342 | 0.000408014 | 0.011441 |
| BankBailedOut_share | 0.0004 | 0.000162037 | 0.00120882 | 7.79664e-05 | 0.00184882 |
| BankBailoutAmount_mean | 0.04 | 0.0654494 | 0.0722927 | 0.0458667 | 0.223609 |
| BankResolutionHaircut_mean | 0.0001 | 0.000184068 | 0.000537252 | 9.36932e-05 | 0.000915014 |
| BankResolved_share | 0.0009 | 0.00156481 | 0.00335783 | 0.000870944 | 0.00669359 |
| Consumption_mean | 100 | 57.4039 | 234.027 | 38.7594 | 430.19 |
| DefaultsHH_rate | 4e-06 | 2.15833e-06 | 1.20882e-05 | 1.08081e-06 | 1.93273e-05 |
| HH_Deposit_mean | 1 | 0.164866 | 1.80732 | 0.0301733 | 3.00236 |
| Output_mean | 100 | 53.6344 | 234.027 | 33.9228 | 421.584 |
| PriceDispersion_mean | 0.0036 | 0.000437492 | 0.0193411 | 0.000343265 | 0.0237218 |
| Transfers_mean | 4 | 0.460035 | 7.22927 | 0.280986 | 11.9703 |

## 3) SA Results (newly implausible share)
| Component | Reduction | Points | Newly implausible | Share |
|---|---:|---:|---:|---:|
| CU | 10% | 246 | 0 | 0.0000 |
| CU | 20% | 246 | 3 | 0.0122 |
| CU | 30% | 246 | 9 | 0.0366 |
| CU | 40% | 246 | 9 | 0.0366 |
| EV | 10% | 246 | 3 | 0.0122 |
| EV | 20% | 246 | 6 | 0.0244 |
| EV | 30% | 246 | 9 | 0.0366 |
| EV | 40% | 246 | 12 | 0.0488 |
| MD | 10% | 246 | 9 | 0.0366 |
| MD | 20% | 246 | 27 | 0.1098 |
| MD | 30% | 246 | 36 | 0.1463 |
| MD | 40% | 246 | 63 | 0.2561 |
| OU | 10% | 246 | 9 | 0.0366 |
| OU | 20% | 246 | 9 | 0.0366 |
| OU | 30% | 246 | 12 | 0.0488 |
| OU | 40% | 246 | 24 | 0.0976 |

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
