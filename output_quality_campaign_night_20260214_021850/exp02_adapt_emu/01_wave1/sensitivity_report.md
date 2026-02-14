# Uncertainty Sensitivity Report (2026-02-14)

## 1) Setup
- Dataset: `C:\Users\Alexander\Desktop\Education\Aspa\code\output_quality_campaign_night_20260214_021850\exp02_adapt_emu\01_wave1\lhs_runs.csv`
- Implausibility threshold: `2.75`
- SA domain: `nroy`
- Baseline points used: `63`
- Reductions: `10%, 20%, 30%, 40%`

## 2) Baseline Uncertainty Decomposition
| Metric | OU var | EV var | MD var | CU var (median) | Total var (median) |
|---|---:|---:|---:|---:|---:|
| AvgPrice_mean | 0.0025 | 0.000884112 | 0.005929 | 0.054796 | 0.0641091 |
| BankBailedOut_share | 0.0004 | 0.000127407 | 0.0009 | 0.00180033 | 0.00322774 |
| BankBailoutAmount_mean | 0.04 | 0.0310981 | 0.053824 | 0.860727 | 0.98565 |
| BankResolutionHaircut_mean | 0.0001 | 0.0001719 | 0.0004 | 0.00268441 | 0.00335631 |
| BankResolved_share | 0.0009 | 0.00134815 | 0.0025 | 0.00888649 | 0.0136346 |
| Consumption_mean | 100 | 144.795 | 174.24 | 773.257 | 1192.29 |
| CreditRejections_mean | 25 | 23.8202 | 400 | 607.028 | 1055.85 |
| DefaultsHH_rate | 4e-06 | 2.10667e-06 | 9e-06 | 2.01503e-05 | 3.5257e-05 |
| Employment_mean | 25 | 74.7025 | 43.56 | 338.277 | 481.539 |
| FirmDowntimeShare_mean | 0.0016 | 0 | 0.0049 | 8.52416e-06 | 0.00650852 |
| HH_Deposit_mean | 1 | 1.98356 | 1.3456 | 12.8246 | 17.1538 |
| Output_mean | 100 | 140.65 | 174.24 | 1510.54 | 1925.43 |
| PriceDispersion_mean | 0.0036 | 0.000337068 | 0.0144 | 0.00130065 | 0.0196377 |
| Transfers_mean | 4 | 1.02288 | 5.3824 | 6.64777 | 17.0531 |

## 3) SA Results (newly implausible share)
| Component | Reduction | Points | Newly implausible | Share |
|---|---:|---:|---:|---:|
| CU | 10% | 63 | 0 | 0.0000 |
| CU | 20% | 63 | 0 | 0.0000 |
| CU | 30% | 63 | 0 | 0.0000 |
| CU | 40% | 63 | 0 | 0.0000 |
| EV | 10% | 63 | 0 | 0.0000 |
| EV | 20% | 63 | 0 | 0.0000 |
| EV | 30% | 63 | 0 | 0.0000 |
| EV | 40% | 63 | 0 | 0.0000 |
| MD | 10% | 63 | 0 | 0.0000 |
| MD | 20% | 63 | 0 | 0.0000 |
| MD | 30% | 63 | 3 | 0.0476 |
| MD | 40% | 63 | 3 | 0.0476 |
| OU | 10% | 63 | 0 | 0.0000 |
| OU | 20% | 63 | 0 | 0.0000 |
| OU | 30% | 63 | 0 | 0.0000 |
| OU | 40% | 63 | 0 | 0.0000 |

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
