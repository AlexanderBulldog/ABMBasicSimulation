# Uncertainty Sensitivity Report (2026-02-14)

## 1) Setup
- Dataset: `C:\Users\Alexander\Desktop\Education\Aspa\code\output_quality_campaign_night_20260214_021850\exp03_wave1_from_baseline_bounds\01_wave1\lhs_runs.csv`
- Implausibility threshold: `2.6`
- SA domain: `nroy`
- Baseline points used: `0`
- Reductions: `10%, 20%, 30%, 40%`

## 2) Baseline Uncertainty Decomposition
| Metric | OU var | EV var | MD var | CU var (median) | Total var (median) |
|---|---:|---:|---:|---:|---:|
| AvgPrice_mean | 0.0025 | 0.000281372 | 0.005929 | nan | nan |
| BankBailedOut_share | 0.0004 | 1.70068e-05 | 0.0009 | nan | nan |
| BankBailoutAmount_mean | 0.04 | 0.00587678 | 0.053824 | nan | nan |
| BankResolutionHaircut_mean | 0.0001 | 4.98608e-05 | 0.0004 | nan | nan |
| BankResolved_share | 0.0009 | 0.000459184 | 0.0025 | nan | nan |
| Consumption_mean | 100 | 31.7553 | 174.24 | nan | nan |
| CreditRejections_mean | 25 | 10.1835 | 400 | nan | nan |
| DefaultsHH_rate | 4e-06 | 2.44898e-07 | 9e-06 | nan | nan |
| Employment_mean | 25 | 15.2467 | 43.56 | nan | nan |
| HH_Deposit_mean | 1 | 0.203826 | 1.3456 | nan | nan |
| InventoryGap_mean | 0.0625 | 0.0165605 | 0.1225 | nan | nan |
| Output_mean | 100 | 37.1767 | 174.24 | nan | nan |
| PriceDispersion_mean | 0.0036 | 0.000214684 | 0.0144 | nan | nan |
| Transfers_mean | 4 | 0.226196 | 5.3824 | nan | nan |

## 3) SA Results (newly implausible share)
| Component | Reduction | Points | Newly implausible | Share |
|---|---:|---:|---:|---:|
| EV | 10% | 0 | 0 | nan |
| EV | 20% | 0 | 0 | nan |
| EV | 30% | 0 | 0 | nan |
| EV | 40% | 0 | 0 | nan |
| OU | 10% | 0 | 0 | nan |
| OU | 20% | 0 | 0 | nan |
| OU | 30% | 0 | 0 | nan |
| OU | 40% | 0 | 0 | nan |
| MD | 10% | 0 | 0 | nan |
| MD | 20% | 0 | 0 | nan |
| MD | 30% | 0 | 0 | nan |
| MD | 40% | 0 | 0 | nan |
| CU | 10% | 0 | 0 | nan |
| CU | 20% | 0 | 0 | nan |
| CU | 30% | 0 | 0 | nan |
| CU | 40% | 0 | 0 | nan |

## 4) Interpretation
- Dominant component (rank at max reduction): `n/a`
- Investment guidance:
  - `CU` dominant: increase design density / improve GP specification.
  - `EV` dominant: increase replicate seeds `K` or refine stochastic structure.
  - `OU` dominant: improve measurement quality / uncertainty model of observations.
  - `MD` dominant: revisit model structure or discrepancy assumptions.

## 5) Limitations
- EV is treated as metric-level scalar (not x-dependent).
- SA is local to selected baseline domain.
- MD remains expert-specified and is not inferred from data.
