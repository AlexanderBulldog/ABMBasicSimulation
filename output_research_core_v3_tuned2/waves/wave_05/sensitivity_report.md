# Uncertainty Sensitivity Report (2026-02-13)

## 1) Setup
- Dataset: `C:\Users\Alexander\Desktop\Education\Aspa\code\output_research_core_v3_tuned2\waves\wave_05\lhs_runs_wave5.csv`
- Implausibility threshold: `2.75`
- SA domain: `nroy`
- Baseline points used: `459`
- Reductions: `10%, 20%, 30%, 40%`

## 2) Baseline Uncertainty Decomposition
| Metric | OU var | EV var | MD var | CU var (median) | Total var (median) |
|---|---:|---:|---:|---:|---:|
| AvgPrice_mean | 0.0025 | 0.00035336 | 0.0138525 | 0.000299924 | 0.0170058 |
| BankBailedOut_share | 0.0004 | 0 | 0.00210276 | 1.24876e-05 | 0.00251525 |
| BankBailoutAmount_mean | 0.04 | 0 | 0.125754 | 0.00239869 | 0.168153 |
| BankResolutionHaircut_mean | 0.0001 | 9.55473e-05 | 0.000934559 | 5.49212e-05 | 0.00118503 |
| BankResolved_share | 0.0009 | 0.00141204 | 0.005841 | 0.000813238 | 0.00896627 |
| Consumption_mean | 100 | 36.7859 | 407.094 | 29.2455 | 573.125 |
| CreditRejections_mean | 25 | 17.4258 | 400 | 12.8499 | 455.276 |
| DefaultsHH_rate | 4e-06 | 4.63426e-07 | 2.10276e-05 | 4.43943e-07 | 2.5935e-05 |
| Employment_mean | 25 | 23.6426 | 101.774 | 13.8908 | 164.307 |
| HH_Deposit_mean | 1 | 2.01919 | 3.14386 | 2.00682 | 8.16987 |
| InventoryGap_mean | 0.0625 | 0.0125188 | 0.1225 | 0.0119211 | 0.20944 |
| Output_mean | 100 | 39.0547 | 407.094 | 25.7329 | 571.882 |
| PriceDispersion_mean | 0.0036 | 0.000477053 | 0.0144 | 0.000255792 | 0.0187328 |
| Transfers_mean | 4 | 0.349533 | 12.5754 | 0.232543 | 17.1575 |

## 3) SA Results (newly implausible share)
| Component | Reduction | Points | Newly implausible | Share |
|---|---:|---:|---:|---:|
| CU | 10% | 459 | 6 | 0.0131 |
| CU | 20% | 459 | 9 | 0.0196 |
| CU | 30% | 459 | 33 | 0.0719 |
| CU | 40% | 459 | 42 | 0.0915 |
| EV | 10% | 459 | 6 | 0.0131 |
| EV | 20% | 459 | 9 | 0.0196 |
| EV | 30% | 459 | 18 | 0.0392 |
| EV | 40% | 459 | 21 | 0.0458 |
| MD | 10% | 459 | 36 | 0.0784 |
| MD | 20% | 459 | 132 | 0.2876 |
| MD | 30% | 459 | 237 | 0.5163 |
| MD | 40% | 459 | 300 | 0.6536 |
| OU | 10% | 459 | 15 | 0.0327 |
| OU | 20% | 459 | 24 | 0.0523 |
| OU | 30% | 459 | 27 | 0.0588 |
| OU | 40% | 459 | 36 | 0.0784 |

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
