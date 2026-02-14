# Uncertainty Sensitivity Report (2026-02-14)

## 1) Setup
- Dataset: `C:\Users\Alexander\Desktop\Education\Aspa\code\output_research_core_v5_preflight1\01_wave1\lhs_runs.csv`
- Implausibility threshold: `2.75`
- SA domain: `nroy`
- Baseline points used: `192`
- Reductions: `10%, 20%, 30%, 40%`

## 2) Baseline Uncertainty Decomposition
| Metric | OU var | EV var | MD var | CU var (median) | Total var (median) |
|---|---:|---:|---:|---:|---:|
| AvgPrice_mean | 0.0025 | 0.000845597 | 0.005929 | 0.000711212 | 0.00998581 |
| BankBailedOut_share | 0.0004 | 0.000138889 | 0.0009 | 0.000108835 | 0.00154772 |
| BankBailoutAmount_mean | 0.04 | 0.0626977 | 0.053824 | 0.0498818 | 0.206404 |
| BankResolutionHaircut_mean | 0.0001 | 0.000274083 | 0.0004 | 0.000159275 | 0.000933358 |
| BankResolved_share | 0.0009 | 0.00126157 | 0.0025 | 0.00086195 | 0.00552352 |
| Bank_Equity_mean | 56.25 | 38.8229 | 59.9076 | 18.9043 | 173.885 |
| Consumption_mean | 100 | 163.626 | 174.24 | 300.952 | 738.819 |
| CreditRejections_mean | 100 | 689.707 | 400 | 361.087 | 1550.79 |
| DefaultsHH_rate | 4e-06 | 4.93519e-06 | 9e-06 | 3.80309e-06 | 2.17383e-05 |
| HH_Deposit_mean | 1 | 0.700037 | 1.3456 | 0.0350362 | 3.08067 |
| Output_mean | 100 | 185.02 | 174.24 | 358.63 | 817.89 |
| PriceDispersion_mean | 0.0036 | 0.000360504 | 0.0144 | 0.0004765 | 0.018837 |
| Transfers_mean | 4 | 1.27138 | 5.3824 | 2.40799 | 13.0618 |

## 3) SA Results (newly implausible share)
| Component | Reduction | Points | Newly implausible | Share |
|---|---:|---:|---:|---:|
| CU | 10% | 192 | 0 | 0.0000 |
| CU | 20% | 192 | 6 | 0.0312 |
| CU | 30% | 192 | 9 | 0.0469 |
| CU | 40% | 192 | 15 | 0.0781 |
| EV | 10% | 192 | 0 | 0.0000 |
| EV | 20% | 192 | 3 | 0.0156 |
| EV | 30% | 192 | 6 | 0.0312 |
| EV | 40% | 192 | 6 | 0.0312 |
| MD | 10% | 192 | 3 | 0.0156 |
| MD | 20% | 192 | 3 | 0.0156 |
| MD | 30% | 192 | 15 | 0.0781 |
| MD | 40% | 192 | 27 | 0.1406 |
| OU | 10% | 192 | 3 | 0.0156 |
| OU | 20% | 192 | 3 | 0.0156 |
| OU | 30% | 192 | 3 | 0.0156 |
| OU | 40% | 192 | 3 | 0.0156 |

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
