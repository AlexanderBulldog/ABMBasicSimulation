# Uncertainty Sensitivity Report (2026-02-15)

## 1) Setup
- Dataset: `C:\Users\Alexander\Desktop\Education\Aspa\code\output_evidence_campaign_v2_20260215\stage_a\A1\waves\wave_03\lhs_runs_wave3.csv`
- Implausibility threshold: `2.6`
- SA domain: `nroy`
- Baseline points used: `279`
- Reductions: `10%, 20%, 30%, 40%`

## 2) Baseline Uncertainty Decomposition
| Metric | OU var | EV var | MD var | CU var (median) | Total var (median) |
|---|---:|---:|---:|---:|---:|
| AvgPrice_mean | 0.0025 | 0.000642487 | 0.00501831 | 0.00061053 | 0.00877132 |
| BankBailedOut_share | 0.0004 | 0.000162037 | 0.00076176 | 0.000161696 | 0.00148549 |
| BankBailoutAmount_mean | 0.04 | 0.0789973 | 0.0455566 | 0.224347 | 0.3889 |
| BankResolutionHaircut_mean | 0.0001 | 0.000236152 | 0.00033856 | 0.000245888 | 0.000920601 |
| BankResolved_share | 0.0009 | 0.00155093 | 0.002116 | 0.00106145 | 0.00562838 |
| Consumption_mean | 100 | 80.4476 | 147.477 | 79.7093 | 407.634 |
| DefaultsHH_rate | 4e-06 | 2.01736e-06 | 7.6176e-06 | 2.49484e-06 | 1.61298e-05 |
| Employment_mean | 25 | 44.5861 | 36.8692 | 37.0354 | 143.491 |
| HH_Deposit_mean | 1 | 0.316949 | 1.13892 | 0.0312691 | 2.48713 |
| Output_mean | 100 | 74.4261 | 147.477 | 73.2582 | 395.161 |
| PriceDispersion_mean | 0.0036 | 0.000368118 | 0.0121882 | 0.000414791 | 0.0165711 |
| Transfers_mean | 4 | 0.619938 | 4.55566 | 0.514249 | 9.68985 |

## 3) SA Results (newly implausible share)
| Component | Reduction | Points | Newly implausible | Share |
|---|---:|---:|---:|---:|
| CU | 10% | 279 | 21 | 0.0753 |
| CU | 20% | 279 | 69 | 0.2473 |
| CU | 30% | 279 | 81 | 0.2903 |
| CU | 40% | 279 | 90 | 0.3226 |
| EV | 10% | 279 | 30 | 0.1075 |
| EV | 20% | 279 | 66 | 0.2366 |
| EV | 30% | 279 | 81 | 0.2903 |
| EV | 40% | 279 | 99 | 0.3548 |
| MD | 10% | 279 | 36 | 0.1290 |
| MD | 20% | 279 | 81 | 0.2903 |
| MD | 30% | 279 | 102 | 0.3656 |
| MD | 40% | 279 | 117 | 0.4194 |
| OU | 10% | 279 | 21 | 0.0753 |
| OU | 20% | 279 | 48 | 0.1720 |
| OU | 30% | 279 | 78 | 0.2796 |
| OU | 40% | 279 | 96 | 0.3441 |

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
