# Uncertainty Sensitivity Report (2026-02-15)

## 1) Setup
- Dataset: `C:\Users\Alexander\Desktop\Education\Aspa\code\output_evidence_campaign_v3_20260215\stage_a\A2\quick_confirm\01_wave1\lhs_runs.csv`
- Implausibility threshold: `2.55`
- SA domain: `nroy`
- Baseline points used: `67`
- Reductions: `10%, 20%, 30%, 40%`

## 2) Baseline Uncertainty Decomposition
| Metric | OU var | EV var | MD var | CU var (median) | Total var (median) |
|---|---:|---:|---:|---:|---:|
| AvgPrice_mean | 0.0025 | 0.00084582 | 0.005929 | 0.000506151 | 0.00978097 |
| BankBailedOut_share | 0.0004 | 0.000162037 | 0.0009 | 6.14672e-05 | 0.0015235 |
| BankBailoutAmount_mean | 0.04 | 0.0390615 | 0.053824 | 0.0191547 | 0.15204 |
| BankResolutionHaircut_mean | 0.0001 | 0.000193697 | 0.0004 | 0.000111037 | 0.000804734 |
| BankResolved_share | 0.0009 | 0.00145833 | 0.0025 | 0.000998269 | 0.0058566 |
| Bank_Equity_mean | 56.25 | 21.0849 | 59.9076 | 18.1709 | 155.413 |
| Consumption_mean | 100 | 77.3516 | 174.24 | 72.2927 | 423.884 |
| CreditRejections_mean | 100 | 492.014 | 400 | 290.463 | 1282.48 |
| DefaultsHH_rate | 4e-06 | 1.75648e-06 | 9e-06 | 1.7988e-06 | 1.65553e-05 |
| Employment_mean | 25 | 38.3569 | 43.56 | 44.8409 | 151.758 |
| HH_Deposit_mean | 1 | 0.463385 | 1.3456 | 0.039712 | 2.8487 |
| Output_mean | 100 | 76.5914 | 174.24 | 93.0699 | 443.901 |
| PriceDispersion_mean | 0.0036 | 0.000557341 | 0.0144 | 0.000333367 | 0.0188907 |
| Transfers_mean | 4 | 0.598618 | 5.3824 | 0.52853 | 10.5095 |

## 3) SA Results (newly implausible share)
| Component | Reduction | Points | Newly implausible | Share |
|---|---:|---:|---:|---:|
| CU | 10% | 67 | 0 | 0.0000 |
| CU | 20% | 67 | 12 | 0.1791 |
| CU | 30% | 67 | 18 | 0.2687 |
| CU | 40% | 67 | 24 | 0.3582 |
| EV | 10% | 67 | 0 | 0.0000 |
| EV | 20% | 67 | 9 | 0.1343 |
| EV | 30% | 67 | 18 | 0.2687 |
| EV | 40% | 67 | 24 | 0.3582 |
| MD | 10% | 67 | 3 | 0.0448 |
| MD | 20% | 67 | 18 | 0.2687 |
| MD | 30% | 67 | 27 | 0.4030 |
| MD | 40% | 67 | 30 | 0.4478 |
| OU | 10% | 67 | 0 | 0.0000 |
| OU | 20% | 67 | 9 | 0.1343 |
| OU | 30% | 67 | 12 | 0.1791 |
| OU | 40% | 67 | 18 | 0.2687 |

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
