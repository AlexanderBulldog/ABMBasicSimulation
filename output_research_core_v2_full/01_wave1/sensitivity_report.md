# Uncertainty Sensitivity Report (2026-02-10)

## 1) Setup
- Dataset: `C:\Users\Alexander\Desktop\Education\Aspa\code\output_research_core_v2_full\01_wave1\lhs_runs.csv`
- Implausibility threshold: `3.0`
- SA domain: `nroy`
- Baseline points used: `809`
- Reductions: `10%, 20%, 30%, 40%`

## 2) Baseline Uncertainty Decomposition
| Metric | OU var | EV var | MD var | CU var (median) | Total var (median) |
|---|---:|---:|---:|---:|---:|
| AvgPrice_mean | 0.0025 | 0.000227841 | 0.0492585 | 0.000135914 | 0.0521223 |
| BankBailedOut_share | 0.0004 | 3.08642e-05 | 0.000158573 | 3.3682e-05 | 0.00062312 |
| BankBailoutAmount_mean | 0.04 | 0.00351304 | 0.04 | 0.00160847 | 0.0851215 |
| BankResolutionHaircut_mean | 0.0001 | 4.96992e-05 | 0.000105539 | 3.76153e-05 | 0.000292853 |
| BankResolved_share | 0.0009 | 0.00114198 | 0.0025 | 0.000811816 | 0.00535379 |
| Bank_Equity_mean | 56.25 | 27.1302 | 56.81 | 4.97285 | 145.163 |
| Consumption_mean | 100 | 386.066 | 1598.56 | 231.956 | 2316.58 |
| DefaultsHH_rate | 4e-06 | 4.30658e-07 | 3.29355e-06 | 3.44033e-07 | 8.06824e-06 |
| Employment_mean | 25 | 242.692 | 863.707 | 139.075 | 1270.47 |
| HH_Deposit_mean | 1 | 0.576257 | 1.86973 | 0.0703721 | 3.51635 |
| Output_mean | 100 | 551.901 | 1186.68 | 314.207 | 2152.79 |
| Transfers_mean | 4 | 3.19428 | 9.65704 | 1.93642 | 18.7877 |

## 3) SA Results (newly implausible share)
| Component | Reduction | Points | Newly implausible | Share |
|---|---:|---:|---:|---:|
| CU | 10% | 809 | 0 | 0.0000 |
| CU | 20% | 809 | 0 | 0.0000 |
| CU | 30% | 809 | 0 | 0.0000 |
| CU | 40% | 809 | 0 | 0.0000 |
| EV | 10% | 809 | 0 | 0.0000 |
| EV | 20% | 809 | 0 | 0.0000 |
| EV | 30% | 809 | 0 | 0.0000 |
| EV | 40% | 809 | 0 | 0.0000 |
| MD | 10% | 809 | 0 | 0.0000 |
| MD | 20% | 809 | 3 | 0.0037 |
| MD | 30% | 809 | 8 | 0.0099 |
| MD | 40% | 809 | 18 | 0.0222 |
| OU | 10% | 809 | 0 | 0.0000 |
| OU | 20% | 809 | 3 | 0.0037 |
| OU | 30% | 809 | 5 | 0.0062 |
| OU | 40% | 809 | 15 | 0.0185 |

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
