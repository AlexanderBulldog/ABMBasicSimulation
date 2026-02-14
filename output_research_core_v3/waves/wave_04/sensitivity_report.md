# Uncertainty Sensitivity Report (2026-02-13)

## 1) Setup
- Dataset: `C:\Users\Alexander\Desktop\Education\Aspa\code\output_research_core_v3\waves\wave_04\lhs_runs_wave4.csv`
- Implausibility threshold: `2.6`
- SA domain: `nroy`
- Baseline points used: `501`
- Reductions: `10%, 20%, 30%, 40%`

## 2) Baseline Uncertainty Decomposition
| Metric | OU var | EV var | MD var | CU var (median) | Total var (median) |
|---|---:|---:|---:|---:|---:|
| AvgPrice_mean | 0.0025 | 0.000323115 | 0.00758028 | 0.000214803 | 0.0106182 |
| BankBailedOut_share | 0.0004 | 2.77778e-05 | 6.12381e-05 | 1.28426e-05 | 0.000501858 |
| BankBailoutAmount_mean | 0.04 | 0.00889073 | 0.00612381 | 0.00257223 | 0.0575868 |
| BankResolutionHaircut_mean | 0.0001 | 4.72423e-05 | 4.12775e-05 | 2.21587e-05 | 0.000210678 |
| BankResolved_share | 0.0009 | 0.000486111 | 0.000382738 | 0.000278343 | 0.00204719 |
| Consumption_mean | 100 | 32.9985 | 274.722 | 16.1413 | 423.862 |
| CreditRejections_mean | 1 | 11.2328 | 1527.89 | 8.18633 | 1548.31 |
| DefaultsHH_rate | 4e-06 | 2.16204e-07 | 1.3273e-06 | 1.55373e-07 | 5.69887e-06 |
| Employment_mean | 25 | 20.2341 | 168.748 | 8.51265 | 222.494 |
| HH_Deposit_mean | 1 | 0.173401 | 0.344464 | 0.0262131 | 1.54408 |
| InventoryGap_mean | 0.0625 | 0.0158922 | 0.00112381 | 0.00793286 | 0.0874489 |
| Output_mean | 100 | 34.7548 | 574.282 | 17.3014 | 726.338 |
| PriceDispersion_mean | 0.0036 | 0.000272935 | 0.000291458 | 0.000138212 | 0.0043026 |
| Transfers_mean | 4 | 0.279184 | 3.45121 | 0.121447 | 7.85184 |

## 3) SA Results (newly implausible share)
| Component | Reduction | Points | Newly implausible | Share |
|---|---:|---:|---:|---:|
| CU | 10% | 501 | 0 | 0.0000 |
| CU | 20% | 501 | 3 | 0.0060 |
| CU | 30% | 501 | 9 | 0.0180 |
| CU | 40% | 501 | 15 | 0.0299 |
| EV | 10% | 501 | 3 | 0.0060 |
| EV | 20% | 501 | 9 | 0.0180 |
| EV | 30% | 501 | 21 | 0.0419 |
| EV | 40% | 501 | 27 | 0.0539 |
| MD | 10% | 501 | 48 | 0.0958 |
| MD | 20% | 501 | 99 | 0.1976 |
| MD | 30% | 501 | 192 | 0.3832 |
| MD | 40% | 501 | 333 | 0.6647 |
| OU | 10% | 501 | 27 | 0.0539 |
| OU | 20% | 501 | 45 | 0.0898 |
| OU | 30% | 501 | 69 | 0.1377 |
| OU | 40% | 501 | 105 | 0.2096 |

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
