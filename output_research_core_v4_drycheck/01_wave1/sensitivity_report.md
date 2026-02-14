# Uncertainty Sensitivity Report (2026-02-14)

## 1) Setup
- Dataset: `C:\Users\Alexander\Desktop\Education\Aspa\code\output_research_core_v4_drycheck\01_wave1\lhs_runs.csv`
- Implausibility threshold: `2.6`
- SA domain: `nroy`
- Baseline points used: `12`
- Reductions: `10%, 20%, 30%, 40%`

## 2) Baseline Uncertainty Decomposition
| Metric | OU var | EV var | MD var | CU var (median) | Total var (median) |
|---|---:|---:|---:|---:|---:|
| AvgPrice_mean | 0.0025 | 0.00124122 | 0.005929 | 0.000392888 | 0.0100631 |
| BankBailedOut_share | 0.0004 | 0.000351852 | 0.0009 | 0.000172928 | 0.00182478 |
| BankBailoutAmount_mean | 0.04 | 0.170937 | 0.053824 | 0.14246 | 0.407222 |
| BankResolutionHaircut_mean | 0.0001 | 0.000990666 | 0.0004 | 0.000475393 | 0.00196606 |
| BankResolved_share | 0.0009 | 0.00222222 | 0.0025 | 0.00115231 | 0.00677453 |
| Bank_Equity_mean | 56.25 | 59.7121 | 59.9076 | 36.8488 | 212.718 |
| Consumption_mean | 100 | 778.525 | 174.24 | 279.851 | 1332.62 |
| CreditRejections_mean | 100 | 67.6942 | 400 | 42.8262 | 610.52 |
| DefaultsHH_rate | 4e-06 | 5.52593e-06 | 9e-06 | 1.80216e-06 | 2.03281e-05 |
| Employment_mean | 25 | 297.548 | 43.56 | 108.603 | 474.711 |
| HH_Deposit_mean | 1 | 0.417657 | 1.3456 | 0.241451 | 3.00471 |
| Output_mean | 100 | 446.206 | 174.24 | 178.696 | 899.142 |
| PriceDispersion_mean | 0.0036 | 0.000430392 | 0.0144 | 0.000114914 | 0.0185453 |
| Transfers_mean | 4 | 4.97753 | 5.3824 | 1.95782 | 16.3178 |

## 3) SA Results (newly implausible share)
| Component | Reduction | Points | Newly implausible | Share |
|---|---:|---:|---:|---:|
| CU | 10% | 12 | 3 | 0.2500 |
| CU | 20% | 12 | 3 | 0.2500 |
| CU | 30% | 12 | 3 | 0.2500 |
| CU | 40% | 12 | 3 | 0.2500 |
| EV | 10% | 12 | 3 | 0.2500 |
| EV | 20% | 12 | 3 | 0.2500 |
| EV | 30% | 12 | 3 | 0.2500 |
| EV | 40% | 12 | 3 | 0.2500 |
| MD | 10% | 12 | 6 | 0.5000 |
| MD | 20% | 12 | 6 | 0.5000 |
| MD | 30% | 12 | 6 | 0.5000 |
| MD | 40% | 12 | 6 | 0.5000 |
| OU | 10% | 12 | 3 | 0.2500 |
| OU | 20% | 12 | 3 | 0.2500 |
| OU | 30% | 12 | 6 | 0.5000 |
| OU | 40% | 12 | 6 | 0.5000 |

## 4) Interpretation
- Dominant component (rank at max reduction): `MD, OU`
- Investment guidance:
  - `CU` dominant: increase design density / improve GP specification.
  - `EV` dominant: increase replicate seeds `K` or refine stochastic structure.
  - `OU` dominant: improve measurement quality / uncertainty model of observations.
  - `MD` dominant: revisit model structure or discrepancy assumptions.

## 5) Limitations
- EV is treated as metric-level scalar (not x-dependent).
- SA is local to selected baseline domain.
- MD remains expert-specified and is not inferred from data.
