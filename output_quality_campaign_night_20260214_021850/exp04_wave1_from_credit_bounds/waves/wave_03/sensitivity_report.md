# Uncertainty Sensitivity Report (2026-02-14)

## 1) Setup
- Dataset: `C:\Users\Alexander\Desktop\Education\Aspa\code\output_quality_campaign_night_20260214_021850\exp04_wave1_from_credit_bounds\waves\wave_03\lhs_runs_wave3.csv`
- Implausibility threshold: `2.75`
- SA domain: `nroy`
- Baseline points used: `210`
- Reductions: `10%, 20%, 30%, 40%`

## 2) Baseline Uncertainty Decomposition
| Metric | OU var | EV var | MD var | CU var (median) | Total var (median) |
|---|---:|---:|---:|---:|---:|
| AvgPrice_mean | 0.0025 | 0.000469952 | 0.00461325 | 0.000321272 | 0.00790448 |
| BankResolutionHaircut_mean | 0.0001 | 2.10652e-05 | 0.000311233 | 2.04389e-05 | 0.000452737 |
| BankResolved_share | 0.0009 | 0.0012466 | 0.00194521 | 0.000708506 | 0.00480031 |
| Consumption_mean | 100 | 56.7583 | 135.573 | 31.0659 | 323.397 |
| CreditRejections_mean | 25 | 13.5633 | 311.233 | 8.78536 | 358.582 |
| DefaultsHH_rate | 4e-06 | 4.7517e-07 | 7.00274e-06 | 1.95938e-07 | 1.16739e-05 |
| HH_Deposit_mean | 1 | 2.23807 | 1.04699 | 2.43244 | 6.7175 |
| Output_mean | 100 | 44.9821 | 135.573 | 24.3333 | 304.889 |
| PriceDispersion_mean | 0.0036 | 0.000424776 | 0.0112044 | 0.000253077 | 0.0154822 |
| Transfers_mean | 4 | 0.459001 | 4.18795 | 0.277826 | 8.92478 |

## 3) SA Results (newly implausible share)
| Component | Reduction | Points | Newly implausible | Share |
|---|---:|---:|---:|---:|
| CU | 10% | 210 | 15 | 0.0714 |
| CU | 20% | 210 | 33 | 0.1571 |
| CU | 30% | 210 | 48 | 0.2286 |
| CU | 40% | 210 | 60 | 0.2857 |
| EV | 10% | 210 | 24 | 0.1143 |
| EV | 20% | 210 | 39 | 0.1857 |
| EV | 30% | 210 | 48 | 0.2286 |
| EV | 40% | 210 | 69 | 0.3286 |
| MD | 10% | 210 | 42 | 0.2000 |
| MD | 20% | 210 | 108 | 0.5143 |
| MD | 30% | 210 | 132 | 0.6286 |
| MD | 40% | 210 | 153 | 0.7286 |
| OU | 10% | 210 | 36 | 0.1714 |
| OU | 20% | 210 | 60 | 0.2857 |
| OU | 30% | 210 | 120 | 0.5714 |
| OU | 40% | 210 | 132 | 0.6286 |

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
