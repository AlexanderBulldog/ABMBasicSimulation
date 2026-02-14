# Uncertainty Sensitivity Report (2026-02-14)

## 1) Setup
- Dataset: `C:\Users\Alexander\Desktop\Education\Aspa\code\output_quality_campaign_night_20260214_021850\exp03_adapt_emu\01_wave1\lhs_runs.csv`
- Implausibility threshold: `2.6`
- SA domain: `nroy`
- Baseline points used: `42`
- Reductions: `10%, 20%, 30%, 40%`

## 2) Baseline Uncertainty Decomposition
| Metric | OU var | EV var | MD var | CU var (median) | Total var (median) |
|---|---:|---:|---:|---:|---:|
| AvgPrice_mean | 0.0025 | 0.00076059 | 0.005929 | 0.000572082 | 0.00976167 |
| BankBailedOut_share | 0.0004 | 0.000200521 | 0.0009 | 0.000360243 | 0.00186076 |
| BankBailoutAmount_mean | 0.04 | 0.0353761 | 0.053824 | 0.0383196 | 0.16752 |
| BankResolutionHaircut_mean | 0.0001 | 0.000336021 | 0.0004 | 0.000387675 | 0.0012237 |
| BankResolved_share | 0.0009 | 0.00145833 | 0.0025 | 0.000989841 | 0.00584817 |
| Consumption_mean | 100 | 184.406 | 174.24 | 189.175 | 647.821 |
| CreditRejections_mean | 25 | 24.1532 | 400 | 18.5485 | 467.702 |
| DefaultsHH_rate | 4e-06 | 1.75443e-06 | 9e-06 | 8.72392e-07 | 1.56268e-05 |
| Employment_mean | 25 | 79.3552 | 43.56 | 104.809 | 252.724 |
| HH_Deposit_mean | 1 | 2.20596 | 1.3456 | 2.16218 | 6.71374 |
| Output_mean | 100 | 162.816 | 174.24 | 177.509 | 614.566 |
| PriceDispersion_mean | 0.0036 | 0.000420732 | 0.0144 | 0.000520261 | 0.018941 |
| Transfers_mean | 4 | 1.37759 | 5.3824 | 1.53867 | 12.2987 |

## 3) SA Results (newly implausible share)
| Component | Reduction | Points | Newly implausible | Share |
|---|---:|---:|---:|---:|
| CU | 10% | 42 | 0 | 0.0000 |
| CU | 20% | 42 | 9 | 0.2143 |
| CU | 30% | 42 | 9 | 0.2143 |
| CU | 40% | 42 | 9 | 0.2143 |
| EV | 10% | 42 | 0 | 0.0000 |
| EV | 20% | 42 | 0 | 0.0000 |
| EV | 30% | 42 | 0 | 0.0000 |
| EV | 40% | 42 | 3 | 0.0714 |
| MD | 10% | 42 | 0 | 0.0000 |
| MD | 20% | 42 | 0 | 0.0000 |
| MD | 30% | 42 | 0 | 0.0000 |
| MD | 40% | 42 | 0 | 0.0000 |
| OU | 10% | 42 | 0 | 0.0000 |
| OU | 20% | 42 | 0 | 0.0000 |
| OU | 30% | 42 | 0 | 0.0000 |
| OU | 40% | 42 | 0 | 0.0000 |

## 4) Interpretation
- Dominant component (rank at max reduction): `CU`
- Investment guidance:
  - `CU` dominant: increase design density / improve GP specification.
  - `EV` dominant: increase replicate seeds `K` or refine stochastic structure.
  - `OU` dominant: improve measurement quality / uncertainty model of observations.
  - `MD` dominant: revisit model structure or discrepancy assumptions.

## 5) Limitations
- EV is treated as metric-level scalar (not x-dependent).
- SA is local to selected baseline domain.
- MD remains expert-specified and is not inferred from data.
