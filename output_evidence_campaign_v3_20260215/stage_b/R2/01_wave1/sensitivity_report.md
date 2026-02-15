# Uncertainty Sensitivity Report (2026-02-15)

## 1) Setup
- Dataset: `C:\Users\Alexander\Desktop\Education\Aspa\code\output_evidence_campaign_v3_20260215\stage_b\R2\01_wave1\lhs_runs.csv`
- Implausibility threshold: `2.55`
- SA domain: `nroy`
- Baseline points used: `286`
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
| Output_mean | 100 | 62.5681 | 174.24 | 162.042 | 498.85 |
| PriceDispersion_mean | 0.0036 | 0.000337574 | 0.0144 | 0.000677159 | 0.0190147 |
| Transfers_mean | 4 | 0.494026 | 5.3824 | 1.09651 | 10.9729 |

## 3) SA Results (newly implausible share)
| Component | Reduction | Points | Newly implausible | Share |
|---|---:|---:|---:|---:|
| CU | 10% | 286 | 15 | 0.0524 |
| CU | 20% | 286 | 24 | 0.0839 |
| CU | 30% | 286 | 38 | 0.1329 |
| CU | 40% | 286 | 41 | 0.1434 |
| EV | 10% | 286 | 3 | 0.0105 |
| EV | 20% | 286 | 3 | 0.0105 |
| EV | 30% | 286 | 18 | 0.0629 |
| EV | 40% | 286 | 21 | 0.0734 |
| MD | 10% | 286 | 12 | 0.0420 |
| MD | 20% | 286 | 30 | 0.1049 |
| MD | 30% | 286 | 45 | 0.1573 |
| MD | 40% | 286 | 59 | 0.2063 |
| OU | 10% | 286 | 3 | 0.0105 |
| OU | 20% | 286 | 18 | 0.0629 |
| OU | 30% | 286 | 27 | 0.0944 |
| OU | 40% | 286 | 36 | 0.1259 |

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
