# Uncertainty Sensitivity Report (2026-02-14)

## 1) Setup
- Dataset: `C:\Users\Alexander\Desktop\Education\Aspa\code\output_research_core_v4_drycheck\waves\wave_04\lhs_runs_wave4.csv`
- Implausibility threshold: `2.6`
- SA domain: `nroy`
- Baseline points used: `51`
- Reductions: `10%, 20%, 30%, 40%`

## 2) Baseline Uncertainty Decomposition
| Metric | OU var | EV var | MD var | CU var (median) | Total var (median) |
|---|---:|---:|---:|---:|---:|
| AvgPrice_mean | 0.0025 | 0.00080665 | 0.00355559 | 0.000526529 | 0.00738877 |
| BankBailedOut_share | 0.0004 | 0.000287037 | 0.000539726 | 6.83064e-05 | 0.00129507 |
| BankBailoutAmount_mean | 0.04 | 0.120196 | 0.032278 | 0.0627542 | 0.255228 |
| BankResolved_share | 0.0009 | 0.00198148 | 0.00149924 | 0.00127558 | 0.0056563 |
| Consumption_mean | 100 | 294.406 | 104.491 | 204.989 | 703.886 |
| CreditRejections_mean | 100 | 262.368 | 239.878 | 144.29 | 746.536 |
| DefaultsHH_rate | 4e-06 | 8.59815e-06 | 5.39726e-06 | 4.02573e-06 | 2.20211e-05 |
| Employment_mean | 25 | 121.56 | 26.1227 | 97.8249 | 270.508 |
| HH_Deposit_mean | 1 | 0.0403444 | 0.80695 | 0.0157751 | 1.86307 |
| InventoryGap_mean | 0.0625 | 0.216057 | 0.0734627 | 0.106647 | 0.458666 |
| Output_mean | 100 | 254.467 | 104.491 | 128.359 | 587.317 |
| Transfers_mean | 4 | 2.17449 | 3.2278 | 1.53757 | 10.9399 |

## 3) SA Results (newly implausible share)
| Component | Reduction | Points | Newly implausible | Share |
|---|---:|---:|---:|---:|
| CU | 10% | 51 | 0 | 0.0000 |
| CU | 20% | 51 | 0 | 0.0000 |
| CU | 30% | 51 | 9 | 0.1765 |
| CU | 40% | 51 | 15 | 0.2941 |
| EV | 10% | 51 | 0 | 0.0000 |
| EV | 20% | 51 | 9 | 0.1765 |
| EV | 30% | 51 | 15 | 0.2941 |
| EV | 40% | 51 | 15 | 0.2941 |
| MD | 10% | 51 | 0 | 0.0000 |
| MD | 20% | 51 | 0 | 0.0000 |
| MD | 30% | 51 | 9 | 0.1765 |
| MD | 40% | 51 | 15 | 0.2941 |
| OU | 10% | 51 | 0 | 0.0000 |
| OU | 20% | 51 | 0 | 0.0000 |
| OU | 30% | 51 | 3 | 0.0588 |
| OU | 40% | 51 | 6 | 0.1176 |

## 4) Interpretation
- Dominant component (rank at max reduction): `CU, EV, MD`
- Investment guidance:
  - `CU` dominant: increase design density / improve GP specification.
  - `EV` dominant: increase replicate seeds `K` or refine stochastic structure.
  - `OU` dominant: improve measurement quality / uncertainty model of observations.
  - `MD` dominant: revisit model structure or discrepancy assumptions.

## 5) Limitations
- EV is treated as metric-level scalar (not x-dependent).
- SA is local to selected baseline domain.
- MD remains expert-specified and is not inferred from data.
