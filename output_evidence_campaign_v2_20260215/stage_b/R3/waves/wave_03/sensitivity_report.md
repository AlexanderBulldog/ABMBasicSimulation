# Uncertainty Sensitivity Report (2026-02-15)

## 1) Setup
- Dataset: `C:\Users\Alexander\Desktop\Education\Aspa\code\output_evidence_campaign_v2_20260215\stage_b\R3\waves\wave_03\lhs_runs_wave3.csv`
- Implausibility threshold: `2.6`
- SA domain: `nroy`
- Baseline points used: `324`
- Reductions: `10%, 20%, 30%, 40%`

## 2) Baseline Uncertainty Decomposition
| Metric | OU var | EV var | MD var | CU var (median) | Total var (median) |
|---|---:|---:|---:|---:|---:|
| AvgPrice_mean | 0.0025 | 0.000367708 | 0.00868065 | 0.000516409 | 0.0120648 |
| BankBailedOut_share | 0.0004 | 4.82094e-05 | 0.00131769 | 5.15921e-05 | 0.00181749 |
| BankBailoutAmount_mean | 0.04 | 0.0180482 | 0.0788037 | 0.011109 | 0.147961 |
| BankResolutionHaircut_mean | 0.0001 | 7.89089e-05 | 0.00058564 | 7.49014e-05 | 0.00083945 |
| BankResolved_share | 0.0009 | 0.000993802 | 0.00366025 | 0.000583194 | 0.00613725 |
| Consumption_mean | 100 | 27.1613 | 255.105 | 62.8656 | 445.132 |
| DefaultsHH_rate | 4e-06 | 1.87114e-06 | 1.31769e-05 | 7.77106e-07 | 1.98251e-05 |
| Employment_mean | 25 | 15.7257 | 63.7762 | 37.4542 | 141.956 |
| HH_Deposit_mean | 1 | 0.407127 | 1.97009 | 0.0335212 | 3.41074 |
| Output_mean | 100 | 28.4988 | 255.105 | 55.0493 | 438.653 |
| PriceDispersion_mean | 0.0036 | 0.000384235 | 0.021083 | 0.000340787 | 0.0254081 |
| Transfers_mean | 4 | 0.178767 | 7.88037 | 0.536661 | 12.5958 |

## 3) SA Results (newly implausible share)
| Component | Reduction | Points | Newly implausible | Share |
|---|---:|---:|---:|---:|
| CU | 10% | 324 | 57 | 0.1759 |
| CU | 20% | 324 | 120 | 0.3704 |
| CU | 30% | 324 | 165 | 0.5093 |
| CU | 40% | 324 | 198 | 0.6111 |
| EV | 10% | 324 | 30 | 0.0926 |
| EV | 20% | 324 | 39 | 0.1204 |
| EV | 30% | 324 | 78 | 0.2407 |
| EV | 40% | 324 | 102 | 0.3148 |
| MD | 10% | 324 | 108 | 0.3333 |
| MD | 20% | 324 | 180 | 0.5556 |
| MD | 30% | 324 | 234 | 0.7222 |
| MD | 40% | 324 | 264 | 0.8148 |
| OU | 10% | 324 | 33 | 0.1019 |
| OU | 20% | 324 | 87 | 0.2685 |
| OU | 30% | 324 | 123 | 0.3796 |
| OU | 40% | 324 | 153 | 0.4722 |

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
