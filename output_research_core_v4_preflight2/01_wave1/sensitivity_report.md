# Uncertainty Sensitivity Report (2026-02-14)

## 1) Setup
- Dataset: `C:\Users\Alexander\Desktop\Education\Aspa\code\output_research_core_v4_preflight2\01_wave1\lhs_runs.csv`
- Implausibility threshold: `2.75`
- SA domain: `nroy`
- Baseline points used: `63`
- Reductions: `10%, 20%, 30%, 40%`

## 2) Baseline Uncertainty Decomposition
| Metric | OU var | EV var | MD var | CU var (median) | Total var (median) |
|---|---:|---:|---:|---:|---:|
| AvgPrice_mean | 0.0025 | 0.000380075 | 0.005929 | 0.000217402 | 0.00902648 |
| BankBailedOut_share | 0.0004 | 1.70068e-05 | 0.0009 | 8.80415e-06 | 0.00132581 |
| BankBailoutAmount_mean | 0.04 | 0.00221469 | 0.053824 | 0.00196674 | 0.0980054 |
| BankResolutionHaircut_mean | 0.0001 | 2.89801e-05 | 0.0004 | 1.87548e-05 | 0.000547735 |
| BankResolved_share | 0.0009 | 0.000272109 | 0.0025 | 0.000162438 | 0.00383455 |
| Bank_Equity_mean | 56.25 | 22.4781 | 59.9076 | 14.5333 | 153.169 |
| Consumption_mean | 100 | 46.0166 | 174.24 | 20.0585 | 340.315 |
| CreditRejections_mean | 100 | 8.37058 | 400 | 6.38334 | 514.754 |
| DefaultsHH_rate | 4e-06 | 1.9898e-07 | 9e-06 | 1.47078e-07 | 1.33461e-05 |
| Employment_mean | 25 | 27.9308 | 43.56 | 11.2402 | 107.731 |
| HH_Deposit_mean | 1 | 0.165997 | 1.3456 | 0.0265246 | 2.53812 |
| InventoryGap_mean | 0.0625 | 0.0164204 | 0.1225 | 0.00877264 | 0.210193 |
| Output_mean | 100 | 51.4922 | 174.24 | 24.0023 | 349.734 |
| PriceDispersion_mean | 0.0036 | 0.000205499 | 0.0144 | 0.00013477 | 0.0183403 |
| Transfers_mean | 4 | 0.34557 | 5.3824 | 0.15783 | 9.8858 |

## 3) SA Results (newly implausible share)
| Component | Reduction | Points | Newly implausible | Share |
|---|---:|---:|---:|---:|
| CU | 10% | 63 | 3 | 0.0476 |
| CU | 20% | 63 | 3 | 0.0476 |
| CU | 30% | 63 | 6 | 0.0952 |
| CU | 40% | 63 | 6 | 0.0952 |
| EV | 10% | 63 | 3 | 0.0476 |
| EV | 20% | 63 | 12 | 0.1905 |
| EV | 30% | 63 | 24 | 0.3810 |
| EV | 40% | 63 | 30 | 0.4762 |
| MD | 10% | 63 | 6 | 0.0952 |
| MD | 20% | 63 | 24 | 0.3810 |
| MD | 30% | 63 | 30 | 0.4762 |
| MD | 40% | 63 | 33 | 0.5238 |
| OU | 10% | 63 | 6 | 0.0952 |
| OU | 20% | 63 | 9 | 0.1429 |
| OU | 30% | 63 | 21 | 0.3333 |
| OU | 40% | 63 | 27 | 0.4286 |

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
