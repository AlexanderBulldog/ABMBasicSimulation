# Uncertainty Sensitivity Report (2026-02-14)

## 1) Setup
- Dataset: `C:\Users\Alexander\Desktop\Education\Aspa\code\output_quality_campaign_night_20260214_021850\exp01_baseline_dense\waves\wave_05\lhs_runs_wave5.csv`
- Implausibility threshold: `2.75`
- SA domain: `nroy`
- Baseline points used: `426`
- Reductions: `10%, 20%, 30%, 40%`

## 2) Baseline Uncertainty Decomposition
| Metric | OU var | EV var | MD var | CU var (median) | Total var (median) |
|---|---:|---:|---:|---:|---:|
| AvgPrice_mean | 0.0025 | 0.000540892 | 0.0105036 | 0.000326297 | 0.0138708 |
| BankBailoutAmount_mean | 0.04 | 0 | 0.0953525 | 0.00117999 | 0.136532 |
| BankResolutionHaircut_mean | 0.0001 | 6.3625e-05 | 0.000708624 | 6.14099e-05 | 0.000933659 |
| BankResolved_share | 0.0009 | 0.00130102 | 0.0044289 | 0.000700341 | 0.00733026 |
| Consumption_mean | 100 | 33.8434 | 308.677 | 21.3837 | 463.904 |
| CreditRejections_mean | 25 | 21.1648 | 708.624 | 13.8113 | 768.601 |
| DefaultsHH_rate | 4e-06 | 7.5e-07 | 1.5944e-05 | 6.83499e-07 | 2.13775e-05 |
| Employment_mean | 25 | 24.321 | 77.1692 | 13.6606 | 140.151 |
| HH_Deposit_mean | 1 | 1.16029 | 2.38381 | 0.631394 | 5.1755 |
| InventoryGap_mean | 0.0625 | 0.0166178 | 0.217016 | 0.00915287 | 0.305287 |
| Output_mean | 100 | 28.3446 | 308.677 | 18.9918 | 456.013 |
| PriceDispersion_mean | 0.0036 | 0.00045065 | 0.0255105 | 0.000272098 | 0.0298332 |
| Transfers_mean | 4 | 0.324454 | 9.53525 | 0.1888 | 14.0485 |

## 3) SA Results (newly implausible share)
| Component | Reduction | Points | Newly implausible | Share |
|---|---:|---:|---:|---:|
| CU | 10% | 426 | 36 | 0.0845 |
| CU | 20% | 426 | 60 | 0.1408 |
| CU | 30% | 426 | 78 | 0.1831 |
| CU | 40% | 426 | 96 | 0.2254 |
| EV | 10% | 426 | 48 | 0.1127 |
| EV | 20% | 426 | 87 | 0.2042 |
| EV | 30% | 426 | 114 | 0.2676 |
| EV | 40% | 426 | 150 | 0.3521 |
| MD | 10% | 426 | 123 | 0.2887 |
| MD | 20% | 426 | 192 | 0.4507 |
| MD | 30% | 426 | 261 | 0.6127 |
| MD | 40% | 426 | 312 | 0.7324 |
| OU | 10% | 426 | 48 | 0.1127 |
| OU | 20% | 426 | 90 | 0.2113 |
| OU | 30% | 426 | 117 | 0.2746 |
| OU | 40% | 426 | 150 | 0.3521 |

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
