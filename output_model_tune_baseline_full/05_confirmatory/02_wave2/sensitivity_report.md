# Uncertainty Sensitivity Report (2026-02-10)

## 1) Setup
- Dataset: `output_research_core_v2_full\05_confirmatory\02_wave2\lhs_runs_wave2.csv`
- Implausibility threshold: `3.0`
- SA domain: `nroy`
- Baseline points used: `555`
- Reductions: `10%, 20%, 30%, 40%`

## 2) Baseline Uncertainty Decomposition
| Metric | OU var | EV var | MD var | CU var (median) | Total var (median) |
|---|---:|---:|---:|---:|---:|
| AvgPrice_mean | 0.0025 | 0.000189525 | 0.00532983 | 0.000134897 | 0.00815425 |
| BankBailedOut_share | 0.0004 | 4.11523e-05 | 1.61303e-05 | 3.11504e-05 | 0.000488433 |
| BankBailoutAmount_mean | 0.04 | 0.00326922 | 0.00406885 | 0.00165367 | 0.0489917 |
| BankResolutionHaircut_mean | 0.0001 | 6.12843e-05 | 1.18201e-05 | 3.41521e-05 | 0.000207256 |
| BankResolved_share | 0.0009 | 0.00125 | 0.000254303 | 0.000753922 | 0.00315822 |
| Bank_Equity_mean | 56.25 | 10.2151 | 5.77926 | 4.44793 | 76.6923 |
| Consumption_mean | 100 | 425.341 | 173.411 | 228.371 | 927.124 |
| DefaultsHH_rate | 4e-06 | 3.71399e-07 | 3.21489e-07 | 2.61287e-07 | 4.95418e-06 |
| Employment_mean | 25 | 265.957 | 97.4389 | 140.507 | 528.903 |
| HH_Deposit_mean | 1 | 0.321786 | 0.186216 | 0.0657235 | 1.57373 |
| Output_mean | 100 | 664.508 | 122.883 | 282.69 | 1170.08 |
| Transfers_mean | 4 | 3.26688 | 1.00872 | 1.90177 | 10.1774 |

## 3) SA Results (newly implausible share)
| Component | Reduction | Points | Newly implausible | Share |
|---|---:|---:|---:|---:|
| CU | 10% | 555 | 0 | 0.0000 |
| CU | 20% | 555 | 3 | 0.0054 |
| CU | 30% | 555 | 3 | 0.0054 |
| CU | 40% | 555 | 3 | 0.0054 |
| EV | 10% | 555 | 3 | 0.0054 |
| EV | 20% | 555 | 3 | 0.0054 |
| EV | 30% | 555 | 3 | 0.0054 |
| EV | 40% | 555 | 3 | 0.0054 |
| MD | 10% | 555 | 19 | 0.0342 |
| MD | 20% | 555 | 44 | 0.0793 |
| MD | 30% | 555 | 62 | 0.1117 |
| MD | 40% | 555 | 83 | 0.1495 |
| OU | 10% | 555 | 10 | 0.0180 |
| OU | 20% | 555 | 29 | 0.0523 |
| OU | 30% | 555 | 51 | 0.0919 |
| OU | 40% | 555 | 74 | 0.1333 |

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
