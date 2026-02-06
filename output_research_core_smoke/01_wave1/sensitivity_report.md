# Uncertainty Sensitivity Report (2026-02-06)

## 1) Setup
- Dataset: `C:\Users\Alexander\Desktop\Education\Aspa\code\output_research_core_smoke\01_wave1\lhs_runs.csv`
- Implausibility threshold: `3.0`
- SA domain: `nroy`
- Baseline points used: `9`
- Reductions: `10%, 20%, 30%, 40%`

## 2) Baseline Uncertainty Decomposition
| Metric | OU var | EV var | MD var | CU var (median) | Total var (median) |
|---|---:|---:|---:|---:|---:|
| AvgPrice_mean | 0.0025 | 0.000134609 | 0.01 | 7.67292e-05 | 0.0127113 |
| BankBailedOut_share | 0.0004 | 0.00251852 | 0.0009 | 0.00110037 | 0.00491889 |
| BankBailoutAmount_mean | 0.04 | 0.0682189 | 0.09 | 0.0844048 | 0.282624 |
| BankResolutionHaircut_mean | 0.0001 | 0.000613984 | 0.0004 | 0.000283472 | 0.00139746 |
| BankResolved_share | 0.0009 | 0.00227778 | 0.0025 | 0.0019666 | 0.00764438 |
| Bank_Equity_mean | 56.25 | 5.22456 | 100 | 200.535 | 362.009 |
| Consumption_mean | 100 | 30.4669 | 225 | 14.3784 | 369.845 |
| DefaultsHH_rate | 4e-06 | 9.31481e-07 | 9e-06 | 1.23497e-06 | 1.51665e-05 |
| Employment_mean | 25 | 18.1853 | 56.25 | 12.5399 | 111.975 |
| HH_Deposit_mean | 1 | 0.112143 | 2.25 | 0.0323581 | 3.3945 |
| Output_mean | 100 | 46.4826 | 225 | 26.2658 | 397.748 |
| Transfers_mean | 4 | 0.275501 | 9 | 0.170447 | 13.4459 |

## 3) SA Results (newly implausible share)
| Component | Reduction | Points | Newly implausible | Share |
|---|---:|---:|---:|---:|
| CU | 10% | 9 | 0 | 0.0000 |
| CU | 20% | 9 | 0 | 0.0000 |
| CU | 30% | 9 | 0 | 0.0000 |
| CU | 40% | 9 | 0 | 0.0000 |
| EV | 10% | 9 | 0 | 0.0000 |
| EV | 20% | 9 | 0 | 0.0000 |
| EV | 30% | 9 | 0 | 0.0000 |
| EV | 40% | 9 | 0 | 0.0000 |
| MD | 10% | 9 | 0 | 0.0000 |
| MD | 20% | 9 | 0 | 0.0000 |
| MD | 30% | 9 | 0 | 0.0000 |
| MD | 40% | 9 | 3 | 0.3333 |
| OU | 10% | 9 | 0 | 0.0000 |
| OU | 20% | 9 | 0 | 0.0000 |
| OU | 30% | 9 | 0 | 0.0000 |
| OU | 40% | 9 | 0 | 0.0000 |

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
