# Uncertainty Sensitivity Report (2026-02-15)

## 1) Setup
- Dataset: `C:\Users\Alexander\Desktop\Education\Aspa\code\output_evidence_campaign_v2_20260215\stage_b\R2\01_wave1\lhs_runs.csv`
- Implausibility threshold: `2.6`
- SA domain: `nroy`
- Baseline points used: `304`
- Reductions: `10%, 20%, 30%, 40%`

## 2) Baseline Uncertainty Decomposition
| Metric | OU var | EV var | MD var | CU var (median) | Total var (median) |
|---|---:|---:|---:|---:|---:|
| AvgPrice_mean | 0.0025 | 0.000483313 | 0.005929 | 0.000403467 | 0.00931578 |
| BankBailedOut_share | 0.0004 | 6.19835e-05 | 0.0009 | 4.84419e-05 | 0.00141043 |
| BankBailoutAmount_mean | 0.04 | 0.0152268 | 0.053824 | 0.0344299 | 0.143481 |
| BankResolutionHaircut_mean | 0.0001 | 9.68348e-05 | 0.0004 | 8.20926e-05 | 0.000678927 |
| BankResolved_share | 0.0009 | 0.000764463 | 0.0025 | 0.000561164 | 0.00472563 |
| Consumption_mean | 100 | 78.0921 | 174.24 | 170.097 | 522.429 |
| DefaultsHH_rate | 4e-06 | 1.71584e-06 | 9e-06 | 1.4893e-06 | 1.62051e-05 |
| HH_Deposit_mean | 1 | 0.778226 | 1.3456 | 0.0356675 | 3.15949 |
| Output_mean | 100 | 62.5681 | 174.24 | 162.036 | 498.844 |
| PriceDispersion_mean | 0.0036 | 0.000337574 | 0.0144 | 0.000677159 | 0.0190147 |
| Transfers_mean | 4 | 0.494026 | 5.3824 | 1.09563 | 10.9721 |

## 3) SA Results (newly implausible share)
| Component | Reduction | Points | Newly implausible | Share |
|---|---:|---:|---:|---:|
| CU | 10% | 304 | 9 | 0.0296 |
| CU | 20% | 304 | 21 | 0.0691 |
| CU | 30% | 304 | 36 | 0.1184 |
| CU | 40% | 304 | 53 | 0.1743 |
| EV | 10% | 304 | 0 | 0.0000 |
| EV | 20% | 304 | 15 | 0.0493 |
| EV | 30% | 304 | 15 | 0.0493 |
| EV | 40% | 304 | 18 | 0.0592 |
| MD | 10% | 304 | 18 | 0.0592 |
| MD | 20% | 304 | 33 | 0.1086 |
| MD | 30% | 304 | 45 | 0.1480 |
| MD | 40% | 304 | 66 | 0.2171 |
| OU | 10% | 304 | 12 | 0.0395 |
| OU | 20% | 304 | 18 | 0.0592 |
| OU | 30% | 304 | 21 | 0.0691 |
| OU | 40% | 304 | 39 | 0.1283 |

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
