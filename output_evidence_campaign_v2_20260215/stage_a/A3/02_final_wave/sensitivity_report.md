# Uncertainty Sensitivity Report (2026-02-15)

## 1) Setup
- Dataset: `C:\Users\Alexander\Desktop\Education\Aspa\code\output_evidence_campaign_v2_20260215\stage_a\A3\waves\wave_03\lhs_runs_wave3.csv`
- Implausibility threshold: `2.6`
- SA domain: `nroy`
- Baseline points used: `264`
- Reductions: `10%, 20%, 30%, 40%`

## 2) Baseline Uncertainty Decomposition
| Metric | OU var | EV var | MD var | CU var (median) | Total var (median) |
|---|---:|---:|---:|---:|---:|
| AvgPrice_mean | 0.0025 | 0.000642487 | 0.00480249 | 0.00061053 | 0.00855551 |
| BankBailedOut_share | 0.0004 | 0.000162037 | 0.000729 | 0.000161696 | 0.00145273 |
| BankBailoutAmount_mean | 0.04 | 0.0789973 | 0.0435974 | 0.224347 | 0.386941 |
| BankResolutionHaircut_mean | 0.0001 | 0.000236152 | 0.000324 | 0.000245888 | 0.000906041 |
| BankResolved_share | 0.0009 | 0.00155093 | 0.002025 | 0.00106145 | 0.00553738 |
| Consumption_mean | 100 | 80.4476 | 141.134 | 79.7093 | 401.291 |
| DefaultsHH_rate | 4e-06 | 2.01736e-06 | 7.29e-06 | 2.49484e-06 | 1.58022e-05 |
| Employment_mean | 25 | 44.5861 | 35.2836 | 37.0354 | 141.905 |
| HH_Deposit_mean | 1 | 0.316949 | 1.08994 | 0.0309855 | 2.43787 |
| Output_mean | 100 | 74.4261 | 141.134 | 73.2582 | 388.819 |
| PriceDispersion_mean | 0.0036 | 0.000368118 | 0.011664 | 0.000414791 | 0.0160469 |
| Transfers_mean | 4 | 0.619938 | 4.35974 | 0.514249 | 9.49393 |

## 3) SA Results (newly implausible share)
| Component | Reduction | Points | Newly implausible | Share |
|---|---:|---:|---:|---:|
| CU | 10% | 264 | 39 | 0.1477 |
| CU | 20% | 264 | 57 | 0.2159 |
| CU | 30% | 264 | 75 | 0.2841 |
| CU | 40% | 264 | 78 | 0.2955 |
| EV | 10% | 264 | 42 | 0.1591 |
| EV | 20% | 264 | 60 | 0.2273 |
| EV | 30% | 264 | 72 | 0.2727 |
| EV | 40% | 264 | 93 | 0.3523 |
| MD | 10% | 264 | 33 | 0.1250 |
| MD | 20% | 264 | 66 | 0.2500 |
| MD | 30% | 264 | 93 | 0.3523 |
| MD | 40% | 264 | 108 | 0.4091 |
| OU | 10% | 264 | 24 | 0.0909 |
| OU | 20% | 264 | 60 | 0.2273 |
| OU | 30% | 264 | 69 | 0.2614 |
| OU | 40% | 264 | 84 | 0.3182 |

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
