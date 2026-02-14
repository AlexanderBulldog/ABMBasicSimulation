# Uncertainty Sensitivity Report (2026-02-14)

## 1) Setup
- Dataset: `C:\Users\Alexander\Desktop\Education\Aspa\code\output_research_core_v6_full_20260214_182812\01_wave1\lhs_runs.csv`
- Implausibility threshold: `2.6`
- SA domain: `nroy`
- Baseline points used: `363`
- Reductions: `10%, 20%, 30%, 40%`

## 2) Baseline Uncertainty Decomposition
| Metric | OU var | EV var | MD var | CU var (median) | Total var (median) |
|---|---:|---:|---:|---:|---:|
| AvgPrice_mean | 0.0025 | 0.000542179 | 0.005929 | 0.000341659 | 0.00931284 |
| BankBailedOut_share | 0.0004 | 6.19835e-05 | 0.0009 | 5.31451e-05 | 0.00141513 |
| BankBailoutAmount_mean | 0.04 | 0.0181607 | 0.053824 | 0.0384239 | 0.150409 |
| BankResolutionHaircut_mean | 0.0001 | 0.000135465 | 0.0004 | 0.000111617 | 0.000747083 |
| BankResolved_share | 0.0009 | 0.000851928 | 0.0025 | 0.000600846 | 0.00485277 |
| Consumption_mean | 100 | 68.7699 | 174.24 | 214.394 | 557.404 |
| DefaultsHH_rate | 4e-06 | 1.82686e-06 | 9e-06 | 1.60148e-06 | 1.64283e-05 |
| HH_Deposit_mean | 1 | 0.851892 | 1.3456 | 0.0387344 | 3.23623 |
| Output_mean | 100 | 95.383 | 174.24 | 234.918 | 604.541 |
| PriceDispersion_mean | 0.0036 | 0.00035487 | 0.0144 | 0.000794598 | 0.0191495 |
| Transfers_mean | 4 | 0.494992 | 5.3824 | 1.76039 | 11.6378 |

## 3) SA Results (newly implausible share)
| Component | Reduction | Points | Newly implausible | Share |
|---|---:|---:|---:|---:|
| CU | 10% | 363 | 9 | 0.0248 |
| CU | 20% | 363 | 24 | 0.0661 |
| CU | 30% | 363 | 45 | 0.1240 |
| CU | 40% | 363 | 72 | 0.1983 |
| EV | 10% | 363 | 12 | 0.0331 |
| EV | 20% | 363 | 18 | 0.0496 |
| EV | 30% | 363 | 27 | 0.0744 |
| EV | 40% | 363 | 27 | 0.0744 |
| MD | 10% | 363 | 30 | 0.0826 |
| MD | 20% | 363 | 45 | 0.1240 |
| MD | 30% | 363 | 72 | 0.1983 |
| MD | 40% | 363 | 90 | 0.2479 |
| OU | 10% | 363 | 15 | 0.0413 |
| OU | 20% | 363 | 21 | 0.0579 |
| OU | 30% | 363 | 33 | 0.0909 |
| OU | 40% | 363 | 42 | 0.1157 |

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
