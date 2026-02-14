# Uncertainty Sensitivity Report (2026-02-14)

## 1) Setup
- Dataset: `C:\Users\Alexander\Desktop\Education\Aspa\code\output_research_core_v4_preflight2\waves\wave_03\lhs_runs_wave3.csv`
- Implausibility threshold: `2.75`
- SA domain: `nroy`
- Baseline points used: `429`
- Reductions: `10%, 20%, 30%, 40%`

## 2) Baseline Uncertainty Decomposition
| Metric | OU var | EV var | MD var | CU var (median) | Total var (median) |
|---|---:|---:|---:|---:|---:|
| AvgPrice_mean | 0.0025 | 0.000364502 | 0.00359508 | 0.000273238 | 0.00673282 |
| BankBailedOut_share | 0.0004 | 6.80272e-05 | 0.00054572 | 2.12258e-05 | 0.00103497 |
| BankBailoutAmount_mean | 0.04 | 0.0128575 | 0.0326365 | 0.00492585 | 0.0904198 |
| BankResolutionHaircut_mean | 0.0001 | 6.21654e-05 | 0.000242542 | 3.13208e-05 | 0.000436028 |
| BankResolved_share | 0.0009 | 0.000527211 | 0.00151589 | 0.000317619 | 0.00326072 |
| Consumption_mean | 100 | 77.7088 | 105.651 | 41.2169 | 324.577 |
| CreditRejections_mean | 100 | 8.07533 | 242.542 | 4.81212 | 355.429 |
| DefaultsHH_rate | 4e-06 | 2.28741e-07 | 5.4572e-06 | 1.82586e-07 | 9.86852e-06 |
| HH_Deposit_mean | 1 | 0.112101 | 0.815911 | 0.0159612 | 1.94397 |
| InventoryGap_mean | 0.0625 | 0.0221312 | 0.0742785 | 0.0133427 | 0.172252 |
| Output_mean | 100 | 61.1071 | 105.651 | 39.8282 | 306.587 |
| PriceDispersion_mean | 0.0036 | 0.000278473 | 0.00873151 | 0.000166867 | 0.0127769 |
| Transfers_mean | 4 | 0.642719 | 3.26365 | 0.350258 | 8.25662 |

## 3) SA Results (newly implausible share)
| Component | Reduction | Points | Newly implausible | Share |
|---|---:|---:|---:|---:|
| CU | 10% | 429 | 0 | 0.0000 |
| CU | 20% | 429 | 3 | 0.0070 |
| CU | 30% | 429 | 15 | 0.0350 |
| CU | 40% | 429 | 24 | 0.0559 |
| EV | 10% | 429 | 0 | 0.0000 |
| EV | 20% | 429 | 21 | 0.0490 |
| EV | 30% | 429 | 39 | 0.0909 |
| EV | 40% | 429 | 48 | 0.1119 |
| MD | 10% | 429 | 12 | 0.0280 |
| MD | 20% | 429 | 42 | 0.0979 |
| MD | 30% | 429 | 63 | 0.1469 |
| MD | 40% | 429 | 87 | 0.2028 |
| OU | 10% | 429 | 3 | 0.0070 |
| OU | 20% | 429 | 42 | 0.0979 |
| OU | 30% | 429 | 63 | 0.1469 |
| OU | 40% | 429 | 78 | 0.1818 |

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
