# Uncertainty Sensitivity Report (2026-02-14)

## 1) Setup
- Dataset: `C:\Users\Alexander\Desktop\Education\Aspa\code\output_quality_campaign_night_20260214_021850\exp01_baseline_dense\01_wave1\lhs_runs.csv`
- Implausibility threshold: `2.75`
- SA domain: `nroy`
- Baseline points used: `41`
- Reductions: `10%, 20%, 30%, 40%`

## 2) Baseline Uncertainty Decomposition
| Metric | OU var | EV var | MD var | CU var (median) | Total var (median) |
|---|---:|---:|---:|---:|---:|
| AvgPrice_mean | 0.0025 | 0.000915805 | 0.005929 | 0.000782217 | 0.010127 |
| BankBailedOut_share | 0.0004 | 0.000215986 | 0.0009 | 0.000475609 | 0.0019916 |
| BankBailoutAmount_mean | 0.04 | 0.0428682 | 0.053824 | 0.0878058 | 0.224498 |
| BankResolutionHaircut_mean | 0.0001 | 0.000154816 | 0.0004 | 0.000400936 | 0.00105575 |
| BankResolved_share | 0.0009 | 0.00132823 | 0.0025 | 0.000790844 | 0.00551907 |
| Bank_Equity_mean | 56.25 | 16.6923 | 59.9076 | 13.7295 | 146.579 |
| Consumption_mean | 100 | 85.7858 | 174.24 | 144.4 | 504.425 |
| CreditRejections_mean | 25 | 25.4141 | 400 | 18.5234 | 468.937 |
| DefaultsHH_rate | 4e-06 | 1.15986e-06 | 9e-06 | 1.72011e-06 | 1.588e-05 |
| Employment_mean | 25 | 48.7238 | 43.56 | 72.3085 | 189.592 |
| FirmDowntimeShare_mean | 0.0016 | 0 | 0.0049 | 1.57244e-05 | 0.00651572 |
| HH_Deposit_mean | 1 | 2.79067 | 1.3456 | 1.84759 | 6.98386 |
| Output_mean | 100 | 101.838 | 174.24 | 156.55 | 532.629 |
| PriceDispersion_mean | 0.0036 | 0.000500201 | 0.0144 | 0.000354028 | 0.0188542 |
| Transfers_mean | 4 | 0.775943 | 5.3824 | 1.29914 | 11.4575 |

## 3) SA Results (newly implausible share)
| Component | Reduction | Points | Newly implausible | Share |
|---|---:|---:|---:|---:|
| CU | 10% | 41 | 3 | 0.0732 |
| CU | 20% | 41 | 6 | 0.1463 |
| CU | 30% | 41 | 6 | 0.1463 |
| CU | 40% | 41 | 6 | 0.1463 |
| EV | 10% | 41 | 0 | 0.0000 |
| EV | 20% | 41 | 3 | 0.0732 |
| EV | 30% | 41 | 3 | 0.0732 |
| EV | 40% | 41 | 6 | 0.1463 |
| MD | 10% | 41 | 0 | 0.0000 |
| MD | 20% | 41 | 3 | 0.0732 |
| MD | 30% | 41 | 6 | 0.1463 |
| MD | 40% | 41 | 6 | 0.1463 |
| OU | 10% | 41 | 0 | 0.0000 |
| OU | 20% | 41 | 0 | 0.0000 |
| OU | 30% | 41 | 0 | 0.0000 |
| OU | 40% | 41 | 6 | 0.1463 |

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
