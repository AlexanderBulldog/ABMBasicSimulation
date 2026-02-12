# Uncertainty Sensitivity Report (2026-02-10)

## 1) Setup
- Dataset: `C:\Users\Alexander\Desktop\Education\Aspa\code\output_research_core_v2_dry\01_wave1\lhs_runs.csv`
- Implausibility threshold: `3.0`
- SA domain: `nroy`
- Baseline points used: `45`
- Reductions: `10%, 20%, 30%, 40%`

## 2) Baseline Uncertainty Decomposition
| Metric | OU var | EV var | MD var | CU var (median) | Total var (median) |
|---|---:|---:|---:|---:|---:|
| AvgPrice_mean | 0.0025 | 0.000134609 | 0.074802 | 7.67006e-05 | 0.0775133 |
| BankBailedOut_share | 0.0004 | 0.00251852 | 0.0004 | 0.00109294 | 0.00441146 |
| BankBailoutAmount_mean | 0.04 | 0.0682189 | 1.07069 | 0.084404 | 1.26332 |
| BankResolutionHaircut_mean | 0.0001 | 0.000613984 | 0.00154734 | 0.000283426 | 0.00254475 |
| BankResolved_share | 0.0009 | 0.00227778 | 0.00444444 | 0.00196013 | 0.00958235 |
| Bank_Equity_mean | 56.25 | 5.22456 | 81.5434 | 0.882578 | 143.901 |
| Consumption_mean | 100 | 30.4669 | 2308.24 | 14.3784 | 2453.08 |
| DefaultsHH_rate | 4e-06 | 9.31481e-07 | 5.05679e-05 | 1.23473e-06 | 5.67341e-05 |
| Employment_mean | 25 | 18.1853 | 1280.45 | 12.5395 | 1336.17 |
| HH_Deposit_mean | 1 | 0.112143 | 2.16878 | 0.0105508 | 3.29147 |
| Output_mean | 100 | 46.4826 | 1221.64 | 26.2641 | 1394.39 |
| Transfers_mean | 4 | 0.275501 | 11.6855 | 0.170447 | 16.1315 |

## 3) SA Results (newly implausible share)
| Component | Reduction | Points | Newly implausible | Share |
|---|---:|---:|---:|---:|
| CU | 10% | 45 | 0 | 0.0000 |
| CU | 20% | 45 | 0 | 0.0000 |
| CU | 30% | 45 | 0 | 0.0000 |
| CU | 40% | 45 | 0 | 0.0000 |
| EV | 10% | 45 | 0 | 0.0000 |
| EV | 20% | 45 | 0 | 0.0000 |
| EV | 30% | 45 | 0 | 0.0000 |
| EV | 40% | 45 | 0 | 0.0000 |
| MD | 10% | 45 | 0 | 0.0000 |
| MD | 20% | 45 | 0 | 0.0000 |
| MD | 30% | 45 | 9 | 0.2000 |
| MD | 40% | 45 | 9 | 0.2000 |
| OU | 10% | 45 | 0 | 0.0000 |
| OU | 20% | 45 | 0 | 0.0000 |
| OU | 30% | 45 | 0 | 0.0000 |
| OU | 40% | 45 | 0 | 0.0000 |

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
