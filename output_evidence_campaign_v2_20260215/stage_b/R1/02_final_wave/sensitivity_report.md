# Uncertainty Sensitivity Report (2026-02-15)

## 1) Setup
- Dataset: `C:\Users\Alexander\Desktop\Education\Aspa\code\output_evidence_campaign_v2_20260215\stage_b\R1\waves\wave_03\lhs_runs_wave3.csv`
- Implausibility threshold: `2.6`
- SA domain: `nroy`
- Baseline points used: `573`
- Reductions: `10%, 20%, 30%, 40%`

## 2) Baseline Uncertainty Decomposition
| Metric | OU var | EV var | MD var | CU var (median) | Total var (median) |
|---|---:|---:|---:|---:|---:|
| AvgPrice_mean | 0.0025 | 0.000391548 | 0.00389002 | 0.000243052 | 0.00702462 |
| BankBailedOut_share | 0.0004 | 8.95317e-05 | 0.00059049 | 5.66477e-05 | 0.00113667 |
| BankBailoutAmount_mean | 0.04 | 0.0338119 | 0.0353139 | 0.0287015 | 0.137827 |
| BankResolutionHaircut_mean | 0.0001 | 9.72368e-05 | 0.00026244 | 8.47777e-05 | 0.000544454 |
| BankResolved_share | 0.0009 | 0.00112259 | 0.00164025 | 0.000687476 | 0.00435032 |
| Consumption_mean | 100 | 40.0142 | 114.319 | 26.5127 | 280.846 |
| DefaultsHH_rate | 4e-06 | 1.73581e-06 | 5.9049e-06 | 1.53624e-06 | 1.3177e-05 |
| HH_Deposit_mean | 1 | 0.259167 | 0.882848 | 0.0225777 | 2.16459 |
| Output_mean | 100 | 36.7218 | 114.319 | 32.8498 | 283.89 |
| PriceDispersion_mean | 0.0036 | 0.000384374 | 0.00944784 | 0.000470963 | 0.0139032 |
| Transfers_mean | 4 | 0.250943 | 3.53139 | 0.200822 | 7.98316 |

## 3) SA Results (newly implausible share)
| Component | Reduction | Points | Newly implausible | Share |
|---|---:|---:|---:|---:|
| CU | 10% | 573 | 12 | 0.0209 |
| CU | 20% | 573 | 15 | 0.0262 |
| CU | 30% | 573 | 21 | 0.0366 |
| CU | 40% | 573 | 30 | 0.0524 |
| EV | 10% | 573 | 12 | 0.0209 |
| EV | 20% | 573 | 18 | 0.0314 |
| EV | 30% | 573 | 33 | 0.0576 |
| EV | 40% | 573 | 48 | 0.0838 |
| MD | 10% | 573 | 27 | 0.0471 |
| MD | 20% | 573 | 78 | 0.1361 |
| MD | 30% | 573 | 108 | 0.1885 |
| MD | 40% | 573 | 150 | 0.2618 |
| OU | 10% | 573 | 27 | 0.0471 |
| OU | 20% | 573 | 69 | 0.1204 |
| OU | 30% | 573 | 102 | 0.1780 |
| OU | 40% | 573 | 123 | 0.2147 |

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
