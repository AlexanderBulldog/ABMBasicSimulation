# Uncertainty Sensitivity Report (2026-02-14)

## 1) Setup
- Dataset: `C:\Users\Alexander\Desktop\Education\Aspa\code\output_quality_campaign_night_20260214_021850\exp01_baseline_dense\waves\wave_04\lhs_runs_wave4.csv`
- Implausibility threshold: `2.75`
- SA domain: `nroy`
- Baseline points used: `258`
- Reductions: `10%, 20%, 30%, 40%`

## 2) Baseline Uncertainty Decomposition
| Metric | OU var | EV var | MD var | CU var (median) | Total var (median) |
|---|---:|---:|---:|---:|---:|
| AvgPrice_mean | 0.0025 | 0.000476856 | 0.0105036 | 0.000384119 | 0.0138646 |
| BankBailedOut_share | 0.0004 | 0 | 0.0015944 | 6.1234e-06 | 0.00200053 |
| BankBailoutAmount_mean | 0.04 | 0 | 0.0953525 | 0.00122297 | 0.136575 |
| BankResolutionHaircut_mean | 0.0001 | 4.15991e-05 | 0.000708624 | 3.34714e-05 | 0.000883695 |
| BankResolved_share | 0.0009 | 0.00142857 | 0.0044289 | 0.000713101 | 0.00747057 |
| Consumption_mean | 100 | 32.2298 | 308.677 | 16.8273 | 457.734 |
| CreditRejections_mean | 25 | 20.8513 | 708.624 | 14.1251 | 768.601 |
| DefaultsHH_rate | 4e-06 | 1.02568e-06 | 1.5944e-05 | 7.19682e-07 | 2.16894e-05 |
| Employment_mean | 25 | 21.0698 | 77.1692 | 10.2839 | 133.523 |
| HH_Deposit_mean | 1 | 1.25755 | 2.38381 | 0.601121 | 5.24248 |
| InventoryGap_mean | 0.0625 | 0.0190663 | 0.217016 | 0.0136768 | 0.312259 |
| Output_mean | 100 | 33.8837 | 308.677 | 15.5282 | 458.089 |
| PriceDispersion_mean | 0.0036 | 0.000549604 | 0.0255105 | 0.000242912 | 0.029903 |
| Transfers_mean | 4 | 0.283342 | 9.53525 | 0.137602 | 13.9562 |

## 3) SA Results (newly implausible share)
| Component | Reduction | Points | Newly implausible | Share |
|---|---:|---:|---:|---:|
| CU | 10% | 258 | 9 | 0.0349 |
| CU | 20% | 258 | 18 | 0.0698 |
| CU | 30% | 258 | 27 | 0.1047 |
| CU | 40% | 258 | 27 | 0.1047 |
| EV | 10% | 258 | 12 | 0.0465 |
| EV | 20% | 258 | 24 | 0.0930 |
| EV | 30% | 258 | 42 | 0.1628 |
| EV | 40% | 258 | 54 | 0.2093 |
| MD | 10% | 258 | 54 | 0.2093 |
| MD | 20% | 258 | 105 | 0.4070 |
| MD | 30% | 258 | 153 | 0.5930 |
| MD | 40% | 258 | 183 | 0.7093 |
| OU | 10% | 258 | 15 | 0.0581 |
| OU | 20% | 258 | 33 | 0.1279 |
| OU | 30% | 258 | 51 | 0.1977 |
| OU | 40% | 258 | 63 | 0.2442 |

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
