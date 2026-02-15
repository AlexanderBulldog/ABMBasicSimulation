# Uncertainty Sensitivity Report (2026-02-15)

## 1) Setup
- Dataset: `C:\Users\Alexander\Desktop\Education\Aspa\code\output_evidence_campaign_v2_sanity_A1\01_wave1\lhs_runs.csv`
- Implausibility threshold: `2.6`
- SA domain: `nroy`
- Baseline points used: `72`
- Reductions: `10%, 20%, 30%, 40%`

## 2) Baseline Uncertainty Decomposition
| Metric | OU var | EV var | MD var | CU var (median) | Total var (median) |
|---|---:|---:|---:|---:|---:|
| AvgPrice_mean | 0.0025 | 0.00112705 | 0.005929 | 0.000490374 | 0.0100464 |
| BankBailedOut_share | 0.0004 | 0.000106481 | 0.0009 | 9.05157e-05 | 0.001497 |
| BankBailoutAmount_mean | 0.04 | 0.0382609 | 0.053824 | 0.0604316 | 0.192517 |
| BankResolutionHaircut_mean | 0.0001 | 0.000222648 | 0.0004 | 0.000110216 | 0.000832864 |
| BankResolved_share | 0.0009 | 0.00131944 | 0.0025 | 0.000788967 | 0.00550841 |
| Bank_Equity_mean | 56.25 | 43.2014 | 59.9076 | 10.9408 | 170.3 |
| Consumption_mean | 100 | 133.236 | 174.24 | 88.4541 | 495.93 |
| DefaultsHH_rate | 4e-06 | 2.39213e-06 | 9e-06 | 2.66328e-06 | 1.80554e-05 |
| Employment_mean | 25 | 82.5558 | 43.56 | 45.5837 | 196.699 |
| HH_Deposit_mean | 1 | 0.350931 | 1.3456 | 0.0303518 | 2.72688 |
| InventoryGap_mean | 0.0625 | 0.104336 | 0.1225 | 0.0569906 | 0.346327 |
| Output_mean | 100 | 156.381 | 174.24 | 92.7932 | 523.415 |
| PriceDispersion_mean | 0.0036 | 0.000291879 | 0.0144 | 0.000138094 | 0.01843 |
| Transfers_mean | 4 | 1.17147 | 5.3824 | 0.740045 | 11.2939 |

## 3) SA Results (newly implausible share)
| Component | Reduction | Points | Newly implausible | Share |
|---|---:|---:|---:|---:|
| CU | 10% | 72 | 0 | 0.0000 |
| CU | 20% | 72 | 0 | 0.0000 |
| CU | 30% | 72 | 0 | 0.0000 |
| CU | 40% | 72 | 3 | 0.0417 |
| EV | 10% | 72 | 0 | 0.0000 |
| EV | 20% | 72 | 0 | 0.0000 |
| EV | 30% | 72 | 12 | 0.1667 |
| EV | 40% | 72 | 21 | 0.2917 |
| MD | 10% | 72 | 3 | 0.0417 |
| MD | 20% | 72 | 6 | 0.0833 |
| MD | 30% | 72 | 6 | 0.0833 |
| MD | 40% | 72 | 9 | 0.1250 |
| OU | 10% | 72 | 0 | 0.0000 |
| OU | 20% | 72 | 3 | 0.0417 |
| OU | 30% | 72 | 3 | 0.0417 |
| OU | 40% | 72 | 6 | 0.0833 |

## 4) Interpretation
- Dominant component (rank at max reduction): `EV`
- Investment guidance:
  - `CU` dominant: increase design density / improve GP specification.
  - `EV` dominant: increase replicate seeds `K` or refine stochastic structure.
  - `OU` dominant: improve measurement quality / uncertainty model of observations.
  - `MD` dominant: revisit model structure or discrepancy assumptions.

## 5) Limitations
- EV is treated as metric-level scalar (not x-dependent).
- SA is local to selected baseline domain.
- MD remains expert-specified and is not inferred from data.
