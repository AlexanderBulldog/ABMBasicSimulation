# Uncertainty Sensitivity Report (2026-02-15)

## 1) Setup
- Dataset: `C:\Users\Alexander\Desktop\Education\Aspa\code\output_confirm_fix_check_20260215\waves\wave_03\lhs_runs_wave3.csv`
- Implausibility threshold: `2.6`
- SA domain: `nroy`
- Baseline points used: `357`
- Reductions: `10%, 20%, 30%, 40%`

## 2) Baseline Uncertainty Decomposition
| Metric | OU var | EV var | MD var | CU var (median) | Total var (median) |
|---|---:|---:|---:|---:|---:|
| AvgPrice_mean | 0.0025 | 0.000532136 | 0.00717409 | 0.000500756 | 0.010707 |
| BankBailedOut_share | 0.0004 | 4.82094e-05 | 0.001089 | 4.13864e-05 | 0.0015786 |
| BankBailoutAmount_mean | 0.04 | 0.0256834 | 0.065127 | 0.0162264 | 0.147037 |
| BankResolutionHaircut_mean | 0.0001 | 9.81608e-05 | 0.000484 | 6.02317e-05 | 0.000742392 |
| BankResolved_share | 0.0009 | 0.00116804 | 0.003025 | 0.000639872 | 0.00573292 |
| Consumption_mean | 100 | 53.1098 | 210.83 | 82.2114 | 446.152 |
| DefaultsHH_rate | 4e-06 | 1.36846e-06 | 1.089e-05 | 1.00944e-06 | 1.72679e-05 |
| Employment_mean | 25 | 30.0221 | 52.7076 | 51.1438 | 158.874 |
| HH_Deposit_mean | 1 | 0.200068 | 1.62818 | 0.0259057 | 2.85415 |
| Output_mean | 100 | 60.0823 | 210.83 | 78.5069 | 449.42 |
| PriceDispersion_mean | 0.0036 | 0.00040828 | 0.017424 | 0.000393913 | 0.0218262 |
| Transfers_mean | 4 | 0.314489 | 6.5127 | 0.657574 | 11.4848 |

## 3) SA Results (newly implausible share)
| Component | Reduction | Points | Newly implausible | Share |
|---|---:|---:|---:|---:|
| CU | 10% | 357 | 75 | 0.2101 |
| CU | 20% | 357 | 120 | 0.3361 |
| CU | 30% | 357 | 177 | 0.4958 |
| CU | 40% | 357 | 216 | 0.6050 |
| EV | 10% | 357 | 42 | 0.1176 |
| EV | 20% | 357 | 84 | 0.2353 |
| EV | 30% | 357 | 111 | 0.3109 |
| EV | 40% | 357 | 147 | 0.4118 |
| MD | 10% | 357 | 75 | 0.2101 |
| MD | 20% | 357 | 135 | 0.3782 |
| MD | 30% | 357 | 189 | 0.5294 |
| MD | 40% | 357 | 225 | 0.6303 |
| OU | 10% | 357 | 39 | 0.1092 |
| OU | 20% | 357 | 72 | 0.2017 |
| OU | 30% | 357 | 111 | 0.3109 |
| OU | 40% | 357 | 123 | 0.3445 |

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
