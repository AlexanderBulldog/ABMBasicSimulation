# Uncertainty Sensitivity Report (2026-02-06)

## 1) Setup
- Dataset: `C:\Users\Alexander\Desktop\Education\Aspa\code\output_research_core\01_wave1\lhs_runs.csv`
- Implausibility threshold: `3.0`
- SA domain: `nroy`
- Baseline points used: `641`
- Reductions: `10%, 20%, 30%, 40%`

## 2) Baseline Uncertainty Decomposition
| Metric | OU var | EV var | MD var | CU var (median) | Total var (median) |
|---|---:|---:|---:|---:|---:|
| AvgPrice_mean | 0.0025 | 0.000227841 | 0.01 | 0.000135914 | 0.0128638 |
| BankBailedOut_share | 0.0004 | 3.08642e-05 | 0.0009 | 3.36511e-05 | 0.00136452 |
| BankBailoutAmount_mean | 0.04 | 0.00351304 | 0.09 | 0.00160814 | 0.135121 |
| BankResolutionHaircut_mean | 0.0001 | 4.96992e-05 | 0.0004 | 3.76153e-05 | 0.000587315 |
| BankResolved_share | 0.0009 | 0.00114198 | 0.0025 | 0.000811411 | 0.00535339 |
| Bank_Equity_mean | 56.25 | 27.1302 | 100 | 5.13332 | 188.514 |
| Consumption_mean | 100 | 386.066 | 225 | 231.956 | 943.022 |
| DefaultsHH_rate | 4e-06 | 4.30658e-07 | 9e-06 | 3.43847e-07 | 1.37745e-05 |
| Employment_mean | 25 | 242.692 | 56.25 | 138.983 | 462.925 |
| HH_Deposit_mean | 1 | 0.576257 | 2.25 | 0.0661442 | 3.8924 |
| Output_mean | 100 | 551.901 | 225 | 314.026 | 1190.93 |
| Transfers_mean | 4 | 3.19428 | 9 | 1.93642 | 18.1307 |

## 3) SA Results (newly implausible share)
| Component | Reduction | Points | Newly implausible | Share |
|---|---:|---:|---:|---:|
| CU | 10% | 641 | 0 | 0.0000 |
| CU | 20% | 641 | 0 | 0.0000 |
| CU | 30% | 641 | 0 | 0.0000 |
| CU | 40% | 641 | 0 | 0.0000 |
| EV | 10% | 641 | 0 | 0.0000 |
| EV | 20% | 641 | 0 | 0.0000 |
| EV | 30% | 641 | 0 | 0.0000 |
| EV | 40% | 641 | 0 | 0.0000 |
| MD | 10% | 641 | 28 | 0.0437 |
| MD | 20% | 641 | 52 | 0.0811 |
| MD | 30% | 641 | 74 | 0.1154 |
| MD | 40% | 641 | 118 | 0.1841 |
| OU | 10% | 641 | 12 | 0.0187 |
| OU | 20% | 641 | 19 | 0.0296 |
| OU | 30% | 641 | 22 | 0.0343 |
| OU | 40% | 641 | 28 | 0.0437 |

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
