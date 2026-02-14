# Uncertainty Sensitivity Report (2026-02-13)

## 1) Setup
- Dataset: `C:\Users\Alexander\Desktop\Education\Aspa\code\output_research_core_v3_tuned1\waves\wave_04\lhs_runs_wave4.csv`
- Implausibility threshold: `2.75`
- SA domain: `nroy`
- Baseline points used: `237`
- Reductions: `10%, 20%, 30%, 40%`

## 2) Baseline Uncertainty Decomposition
| Metric | OU var | EV var | MD var | CU var (median) | Total var (median) |
|---|---:|---:|---:|---:|---:|
| AvgPrice_mean | 0.0025 | 0.00049975 | 0.00722468 | 0.000312806 | 0.0105372 |
| BankBailedOut_share | 0.0004 | 0.000166667 | 0.00109668 | 9.75892e-05 | 0.00176094 |
| BankBailoutAmount_mean | 0.04 | 0.039554 | 0.0655863 | 0.0176028 | 0.162743 |
| BankResolutionHaircut_mean | 0.0001 | 0.000236169 | 0.000487413 | 0.000144125 | 0.000967707 |
| BankResolved_share | 0.0009 | 0.00194444 | 0.00304633 | 0.00111033 | 0.00700111 |
| Consumption_mean | 100 | 50.5009 | 212.317 | 43.9107 | 406.729 |
| CreditRejections_mean | 25 | 17.941 | 487.413 | 10.9108 | 541.265 |
| DefaultsHH_rate | 4e-06 | 5.0787e-07 | 1.09668e-05 | 4.04377e-07 | 1.5879e-05 |
| HH_Deposit_mean | 1 | 1.30012 | 1.63966 | 1.44955 | 5.38932 |
| Output_mean | 100 | 53.2903 | 212.317 | 46.2932 | 411.901 |
| PriceDispersion_mean | 0.0036 | 0.000384847 | 0.0175469 | 0.000405687 | 0.0219374 |
| Transfers_mean | 4 | 0.375151 | 6.55863 | 0.37175 | 11.3055 |

## 3) SA Results (newly implausible share)
| Component | Reduction | Points | Newly implausible | Share |
|---|---:|---:|---:|---:|
| CU | 10% | 237 | 6 | 0.0253 |
| CU | 20% | 237 | 12 | 0.0506 |
| CU | 30% | 237 | 15 | 0.0633 |
| CU | 40% | 237 | 18 | 0.0759 |
| EV | 10% | 237 | 3 | 0.0127 |
| EV | 20% | 237 | 15 | 0.0633 |
| EV | 30% | 237 | 15 | 0.0633 |
| EV | 40% | 237 | 18 | 0.0759 |
| MD | 10% | 237 | 18 | 0.0759 |
| MD | 20% | 237 | 30 | 0.1266 |
| MD | 30% | 237 | 39 | 0.1646 |
| MD | 40% | 237 | 60 | 0.2532 |
| OU | 10% | 237 | 12 | 0.0506 |
| OU | 20% | 237 | 15 | 0.0633 |
| OU | 30% | 237 | 21 | 0.0886 |
| OU | 40% | 237 | 30 | 0.1266 |

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
