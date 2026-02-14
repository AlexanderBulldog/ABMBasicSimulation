# Uncertainty Sensitivity Report (2026-02-14)

## 1) Setup
- Dataset: `C:\Users\Alexander\Desktop\Education\Aspa\code\output_quality_campaign_night_20260214_021850\exp02_adapt_emu\waves\wave_03\lhs_runs_wave3.csv`
- Implausibility threshold: `2.75`
- SA domain: `nroy`
- Baseline points used: `216`
- Reductions: `10%, 20%, 30%, 40%`

## 2) Baseline Uncertainty Decomposition
| Metric | OU var | EV var | MD var | CU var (median) | Total var (median) |
|---|---:|---:|---:|---:|---:|
| AvgPrice_mean | 0.0025 | 0.000628008 | 0.00806634 | 0.000449529 | 0.0116439 |
| BankBailedOut_share | 0.0004 | 5.92593e-05 | 0.00122444 | 6.19592e-05 | 0.00174566 |
| BankBailoutAmount_mean | 0.04 | 0.0202938 | 0.073227 | 0.0093205 | 0.142841 |
| BankResolutionHaircut_mean | 0.0001 | 0.000209961 | 0.000544196 | 0.000114049 | 0.000968206 |
| BankResolved_share | 0.0009 | 0.00179704 | 0.00340122 | 0.000821542 | 0.0069198 |
| Consumption_mean | 100 | 62.7537 | 237.052 | 42.9892 | 442.795 |
| CreditRejections_mean | 25 | 22.1568 | 544.196 | 13.7413 | 605.094 |
| DefaultsHH_rate | 4e-06 | 3.084e-06 | 1.22444e-05 | 1.8475e-06 | 2.11759e-05 |
| Employment_mean | 25 | 39.5203 | 59.2629 | 24.0482 | 147.831 |
| HH_Deposit_mean | 1 | 1.94025 | 1.83067 | 1.2058 | 5.97673 |
| Output_mean | 100 | 50.6459 | 237.052 | 30.2735 | 417.971 |
| PriceDispersion_mean | 0.0036 | 0.000513152 | 0.019591 | 0.000393887 | 0.0240981 |
| Transfers_mean | 4 | 0.627333 | 7.3227 | 0.37338 | 12.3234 |

## 3) SA Results (newly implausible share)
| Component | Reduction | Points | Newly implausible | Share |
|---|---:|---:|---:|---:|
| CU | 10% | 216 | 39 | 0.1806 |
| CU | 20% | 216 | 60 | 0.2778 |
| CU | 30% | 216 | 75 | 0.3472 |
| CU | 40% | 216 | 90 | 0.4167 |
| EV | 10% | 216 | 45 | 0.2083 |
| EV | 20% | 216 | 66 | 0.3056 |
| EV | 30% | 216 | 87 | 0.4028 |
| EV | 40% | 216 | 108 | 0.5000 |
| MD | 10% | 216 | 60 | 0.2778 |
| MD | 20% | 216 | 93 | 0.4306 |
| MD | 30% | 216 | 117 | 0.5417 |
| MD | 40% | 216 | 135 | 0.6250 |
| OU | 10% | 216 | 36 | 0.1667 |
| OU | 20% | 216 | 51 | 0.2361 |
| OU | 30% | 216 | 69 | 0.3194 |
| OU | 40% | 216 | 87 | 0.4028 |

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
