# Uncertainty Sensitivity Report (2026-02-10)

## 1) Setup
- Dataset: `C:\Users\Alexander\Desktop\Education\Aspa\code\output_research_core_v2_dry\05_confirmatory\01_wave1\lhs_runs.csv`
- Implausibility threshold: `3.0`
- SA domain: `nroy`
- Baseline points used: `35`
- Reductions: `10%, 20%, 30%, 40%`

## 2) Baseline Uncertainty Decomposition
| Metric | OU var | EV var | MD var | CU var (median) | Total var (median) |
|---|---:|---:|---:|---:|---:|
| AvgPrice_mean | 0.0025 | 0.000174406 | 0.0353809 | 0.000121968 | 0.0381772 |
| BankBailedOut_share | 0.0004 | 0.000425926 | 0.0004 | 0.000323979 | 0.00154991 |
| BankBailoutAmount_mean | 0.04 | 0.0288967 | 0.0988648 | 0.00663491 | 0.174396 |
| BankResolutionHaircut_mean | 0.0001 | 0.000483882 | 0.00216951 | 0.00019234 | 0.00294573 |
| BankResolved_share | 0.0009 | 0.000888889 | 0.00444444 | 0.000628723 | 0.00686206 |
| Bank_Equity_mean | 56.25 | 11.5057 | 77.4929 | 0.936694 | 146.185 |
| Consumption_mean | 100 | 54.3066 | 1666.31 | 16.6661 | 1837.28 |
| DefaultsHH_rate | 4e-06 | 5.03704e-07 | 2.55586e-05 | 2.91725e-07 | 3.03541e-05 |
| Employment_mean | 25 | 30.854 | 1312.85 | 10.2246 | 1378.93 |
| HH_Deposit_mean | 1 | 0.150587 | 1.71972 | 0.0951727 | 2.96548 |
| Output_mean | 100 | 74.3412 | 620.417 | 27.4927 | 822.251 |
| Transfers_mean | 4 | 0.487365 | 8.2675 | 0.145388 | 12.9003 |

## 3) SA Results (newly implausible share)
| Component | Reduction | Points | Newly implausible | Share |
|---|---:|---:|---:|---:|
| CU | 10% | 35 | 0 | 0.0000 |
| CU | 20% | 35 | 0 | 0.0000 |
| CU | 30% | 35 | 0 | 0.0000 |
| CU | 40% | 35 | 0 | 0.0000 |
| EV | 10% | 35 | 0 | 0.0000 |
| EV | 20% | 35 | 0 | 0.0000 |
| EV | 30% | 35 | 0 | 0.0000 |
| EV | 40% | 35 | 0 | 0.0000 |
| MD | 10% | 35 | 0 | 0.0000 |
| MD | 20% | 35 | 2 | 0.0571 |
| MD | 30% | 35 | 2 | 0.0571 |
| MD | 40% | 35 | 2 | 0.0571 |
| OU | 10% | 35 | 0 | 0.0000 |
| OU | 20% | 35 | 0 | 0.0000 |
| OU | 30% | 35 | 0 | 0.0000 |
| OU | 40% | 35 | 2 | 0.0571 |

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
