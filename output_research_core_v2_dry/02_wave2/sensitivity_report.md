# Uncertainty Sensitivity Report (2026-02-10)

## 1) Setup
- Dataset: `C:\Users\Alexander\Desktop\Education\Aspa\code\output_research_core_v2_dry\02_wave2\lhs_runs_wave2.csv`
- Implausibility threshold: `3.0`
- SA domain: `nroy`
- Baseline points used: `45`
- Reductions: `10%, 20%, 30%, 40%`

## 2) Baseline Uncertainty Decomposition
| Metric | OU var | EV var | MD var | CU var (median) | Total var (median) |
|---|---:|---:|---:|---:|---:|
| AvgPrice_mean | 0.0025 | 0.000102952 | 0.0605896 | 9.96982e-05 | 0.0632922 |
| BankBailedOut_share | 0.0004 | 0.00037037 | 0.000324 | 0.000169703 | 0.00126407 |
| BankBailoutAmount_mean | 0.04 | 0.0163867 | 0.867262 | 0.00799535 | 0.931644 |
| BankResolutionHaircut_mean | 0.0001 | 0.00034956 | 0.00125335 | 0.000165493 | 0.0018684 |
| BankResolved_share | 0.0009 | 0.000648148 | 0.0036 | 0.000450697 | 0.00559885 |
| Bank_Equity_mean | 56.25 | 18.0688 | 66.0501 | 0.567754 | 140.937 |
| Consumption_mean | 100 | 56.1763 | 1869.67 | 18.0961 | 2043.94 |
| DefaultsHH_rate | 4e-06 | 8.7037e-07 | 4.096e-05 | 4.78449e-07 | 4.63088e-05 |
| Employment_mean | 25 | 19.8229 | 1037.16 | 8.98738 | 1090.97 |
| HH_Deposit_mean | 1 | 0.176158 | 1.75671 | 0.0130377 | 2.9459 |
| Output_mean | 100 | 64.8502 | 989.531 | 22.5617 | 1176.94 |
| Transfers_mean | 4 | 0.364584 | 9.46527 | 0.120863 | 13.9507 |

## 3) SA Results (newly implausible share)
| Component | Reduction | Points | Newly implausible | Share |
|---|---:|---:|---:|---:|
| CU | 10% | 45 | 0 | 0.0000 |
| CU | 20% | 45 | 0 | 0.0000 |
| CU | 30% | 45 | 0 | 0.0000 |
| CU | 40% | 45 | 0 | 0.0000 |
| EV | 10% | 45 | 0 | 0.0000 |
| EV | 20% | 45 | 0 | 0.0000 |
| EV | 30% | 45 | 0 | 0.0000 |
| EV | 40% | 45 | 0 | 0.0000 |
| MD | 10% | 45 | 3 | 0.0667 |
| MD | 20% | 45 | 6 | 0.1333 |
| MD | 30% | 45 | 6 | 0.1333 |
| MD | 40% | 45 | 9 | 0.2000 |
| OU | 10% | 45 | 0 | 0.0000 |
| OU | 20% | 45 | 0 | 0.0000 |
| OU | 30% | 45 | 0 | 0.0000 |
| OU | 40% | 45 | 0 | 0.0000 |

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
