# Uncertainty Sensitivity Report (2026-02-14)

## 1) Setup
- Dataset: `C:\Users\Alexander\Desktop\Education\Aspa\code\output_evidence_campaign_20260214\stage_a\C2\01_wave1\lhs_runs.csv`
- Implausibility threshold: `2.75`
- SA domain: `nroy`
- Baseline points used: `109`
- Reductions: `10%, 20%, 30%, 40%`

## 2) Baseline Uncertainty Decomposition
| Metric | OU var | EV var | MD var | CU var (median) | Total var (median) |
|---|---:|---:|---:|---:|---:|
| AvgPrice_mean | 0.0025 | 0.000805301 | 0.005929 | 0.000507452 | 0.00974175 |
| BankBailedOut_share | 0.0004 | 0.000162037 | 0.0009 | 6.12792e-05 | 0.00152332 |
| BankBailoutAmount_mean | 0.04 | 0.0390615 | 0.053824 | 0.0191547 | 0.15204 |
| BankResolutionHaircut_mean | 0.0001 | 0.000193697 | 0.0004 | 0.000111037 | 0.000804734 |
| BankResolved_share | 0.0009 | 0.00145833 | 0.0025 | 0.000998269 | 0.0058566 |
| Bank_Equity_mean | 56.25 | 23.3513 | 59.9076 | 15.752 | 155.261 |
| Consumption_mean | 100 | 80.8011 | 174.24 | 92.1682 | 447.209 |
| DefaultsHH_rate | 4e-06 | 1.75648e-06 | 9e-06 | 1.7988e-06 | 1.65553e-05 |
| Employment_mean | 25 | 43.3458 | 43.56 | 59.9476 | 171.853 |
| HH_Deposit_mean | 1 | 0.452269 | 1.3456 | 0.0409274 | 2.8388 |
| Output_mean | 100 | 103.721 | 174.24 | 125.984 | 503.945 |
| PriceDispersion_mean | 0.0036 | 0.00060206 | 0.0144 | 0.000349286 | 0.0189513 |
| Transfers_mean | 4 | 0.584837 | 5.3824 | 0.702112 | 10.6693 |

## 3) SA Results (newly implausible share)
| Component | Reduction | Points | Newly implausible | Share |
|---|---:|---:|---:|---:|
| CU | 10% | 109 | 0 | 0.0000 |
| CU | 20% | 109 | 9 | 0.0826 |
| CU | 30% | 109 | 9 | 0.0826 |
| CU | 40% | 109 | 24 | 0.2202 |
| EV | 10% | 109 | 0 | 0.0000 |
| EV | 20% | 109 | 0 | 0.0000 |
| EV | 30% | 109 | 9 | 0.0826 |
| EV | 40% | 109 | 9 | 0.0826 |
| MD | 10% | 109 | 0 | 0.0000 |
| MD | 20% | 109 | 0 | 0.0000 |
| MD | 30% | 109 | 12 | 0.1101 |
| MD | 40% | 109 | 15 | 0.1376 |
| OU | 10% | 109 | 0 | 0.0000 |
| OU | 20% | 109 | 0 | 0.0000 |
| OU | 30% | 109 | 0 | 0.0000 |
| OU | 40% | 109 | 9 | 0.0826 |

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
