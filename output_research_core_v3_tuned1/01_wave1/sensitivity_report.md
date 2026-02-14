# Uncertainty Sensitivity Report (2026-02-13)

## 1) Setup
- Dataset: `C:\Users\Alexander\Desktop\Education\Aspa\code\output_research_core_v3_tuned1\01_wave1\lhs_runs.csv`
- Implausibility threshold: `2.75`
- SA domain: `nroy`
- Baseline points used: `51`
- Reductions: `10%, 20%, 30%, 40%`

## 2) Baseline Uncertainty Decomposition
| Metric | OU var | EV var | MD var | CU var (median) | Total var (median) |
|---|---:|---:|---:|---:|---:|
| AvgPrice_mean | 0.0025 | 0.000768886 | 0.005929 | 0.000464739 | 0.00966262 |
| BankBailedOut_share | 0.0004 | 0.000162037 | 0.0009 | 0.000125804 | 0.00158784 |
| BankBailoutAmount_mean | 0.04 | 0.0573976 | 0.053824 | 0.0306194 | 0.181841 |
| BankResolutionHaircut_mean | 0.0001 | 0.000321198 | 0.0004 | 0.000258925 | 0.00108012 |
| BankResolved_share | 0.0009 | 0.0018287 | 0.0025 | 0.0011282 | 0.0063569 |
| Consumption_mean | 100 | 218.675 | 174.24 | 211.912 | 704.827 |
| CreditRejections_mean | 25 | 31.9928 | 400 | 24.9716 | 481.964 |
| DefaultsHH_rate | 4e-06 | 2.76204e-06 | 9e-06 | 2.03362e-06 | 1.77957e-05 |
| Employment_mean | 25 | 129.602 | 43.56 | 111.992 | 310.154 |
| HH_Deposit_mean | 1 | 2.79342 | 1.3456 | 2.26374 | 7.40276 |
| Output_mean | 100 | 240.433 | 174.24 | 225.762 | 740.435 |
| PriceDispersion_mean | 0.0036 | 0.000563851 | 0.0144 | 0.000803188 | 0.019367 |
| Transfers_mean | 4 | 1.60983 | 5.3824 | 1.90864 | 12.9009 |

## 3) SA Results (newly implausible share)
| Component | Reduction | Points | Newly implausible | Share |
|---|---:|---:|---:|---:|
| CU | 10% | 51 | 0 | 0.0000 |
| CU | 20% | 51 | 0 | 0.0000 |
| CU | 30% | 51 | 3 | 0.0588 |
| CU | 40% | 51 | 3 | 0.0588 |
| EV | 10% | 51 | 0 | 0.0000 |
| EV | 20% | 51 | 0 | 0.0000 |
| EV | 30% | 51 | 3 | 0.0588 |
| EV | 40% | 51 | 3 | 0.0588 |
| MD | 10% | 51 | 6 | 0.1176 |
| MD | 20% | 51 | 6 | 0.1176 |
| MD | 30% | 51 | 6 | 0.1176 |
| MD | 40% | 51 | 9 | 0.1765 |
| OU | 10% | 51 | 0 | 0.0000 |
| OU | 20% | 51 | 3 | 0.0588 |
| OU | 30% | 51 | 6 | 0.1176 |
| OU | 40% | 51 | 6 | 0.1176 |

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
