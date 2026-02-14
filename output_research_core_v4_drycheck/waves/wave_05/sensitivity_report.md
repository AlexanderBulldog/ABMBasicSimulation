# Uncertainty Sensitivity Report (2026-02-14)

## 1) Setup
- Dataset: `C:\Users\Alexander\Desktop\Education\Aspa\code\output_research_core_v4_drycheck\waves\wave_05\lhs_runs_wave5.csv`
- Implausibility threshold: `2.6`
- SA domain: `nroy`
- Baseline points used: `33`
- Reductions: `10%, 20%, 30%, 40%`

## 2) Baseline Uncertainty Decomposition
| Metric | OU var | EV var | MD var | CU var (median) | Total var (median) |
|---|---:|---:|---:|---:|---:|
| AvgPrice_mean | 0.0025 | 0.00047603 | 0.00355559 | 0.000355651 | 0.00688728 |
| BankBailedOut_share | 0.0004 | 0.000398148 | 0.000539726 | 0.000267763 | 0.00160564 |
| BankBailoutAmount_mean | 0.04 | 0.279959 | 0.032278 | 0.126613 | 0.47885 |
| BankResolutionHaircut_mean | 0.0001 | 0.000835269 | 0.000239878 | 0.000555811 | 0.00173096 |
| BankResolved_share | 0.0009 | 0.00177778 | 0.00149924 | 0.00141073 | 0.00558775 |
| Consumption_mean | 100 | 72.6187 | 104.491 | 46.9484 | 324.058 |
| DefaultsHH_rate | 4e-06 | 3.95926e-06 | 5.39726e-06 | 3.92221e-06 | 1.72787e-05 |
| Employment_mean | 25 | 46.1168 | 26.1227 | 31.7648 | 129.004 |
| HH_Deposit_mean | 1 | 0.0974215 | 0.80695 | 0.0291537 | 1.93353 |
| InventoryGap_mean | 0.0625 | 0.0915729 | 0.0734627 | 0.0620688 | 0.289604 |
| Output_mean | 100 | 58.3838 | 104.491 | 36.2857 | 299.16 |
| PriceDispersion_mean | 0.0036 | 0.000521722 | 0.00863561 | 0.000321637 | 0.013079 |
| Transfers_mean | 4 | 0.762432 | 3.2278 | 0.48533 | 8.47556 |

## 3) SA Results (newly implausible share)
| Component | Reduction | Points | Newly implausible | Share |
|---|---:|---:|---:|---:|
| CU | 10% | 33 | 0 | 0.0000 |
| CU | 20% | 33 | 3 | 0.0909 |
| CU | 30% | 33 | 3 | 0.0909 |
| CU | 40% | 33 | 3 | 0.0909 |
| EV | 10% | 33 | 0 | 0.0000 |
| EV | 20% | 33 | 3 | 0.0909 |
| EV | 30% | 33 | 3 | 0.0909 |
| EV | 40% | 33 | 3 | 0.0909 |
| MD | 10% | 33 | 3 | 0.0909 |
| MD | 20% | 33 | 3 | 0.0909 |
| MD | 30% | 33 | 3 | 0.0909 |
| MD | 40% | 33 | 6 | 0.1818 |
| OU | 10% | 33 | 3 | 0.0909 |
| OU | 20% | 33 | 3 | 0.0909 |
| OU | 30% | 33 | 3 | 0.0909 |
| OU | 40% | 33 | 3 | 0.0909 |

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
