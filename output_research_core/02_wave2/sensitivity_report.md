# Uncertainty Sensitivity Report (2026-02-06)

## 1) Setup
- Dataset: `C:\Users\Alexander\Desktop\Education\Aspa\code\output_research_core\02_wave2\lhs_runs_wave2.csv`
- Implausibility threshold: `3.0`
- SA domain: `nroy`
- Baseline points used: `749`
- Reductions: `10%, 20%, 30%, 40%`

## 2) Baseline Uncertainty Decomposition
| Metric | OU var | EV var | MD var | CU var (median) | Total var (median) |
|---|---:|---:|---:|---:|---:|
| AvgPrice_mean | 0.0025 | 0.000235104 | 0.01 | 0.000161079 | 0.0128962 |
| BankBailedOut_share | 0.0004 | 3.08642e-05 | 0.0009 | 2.39652e-05 | 0.00135483 |
| BankBailoutAmount_mean | 0.04 | 0.00263458 | 0.09 | 0.00174011 | 0.134375 |
| BankResolutionHaircut_mean | 0.0001 | 5.65757e-05 | 0.0004 | 3.75075e-05 | 0.000594083 |
| BankResolved_share | 0.0009 | 0.00141358 | 0.0025 | 0.000811887 | 0.00562547 |
| Bank_Equity_mean | 56.25 | 5.97458 | 100 | 2.50309 | 164.728 |
| Consumption_mean | 100 | 401.233 | 225 | 249.027 | 975.259 |
| DefaultsHH_rate | 4e-06 | 3.26955e-07 | 9e-06 | 3.41548e-07 | 1.36685e-05 |
| Employment_mean | 25 | 258.941 | 56.25 | 152.406 | 492.597 |
| HH_Deposit_mean | 1 | 0.372992 | 2.25 | 0.0605803 | 3.68357 |
| Output_mean | 100 | 632.082 | 225 | 312.739 | 1269.82 |
| Transfers_mean | 4 | 3.24804 | 9 | 2.04934 | 18.2974 |

## 3) SA Results (newly implausible share)
| Component | Reduction | Points | Newly implausible | Share |
|---|---:|---:|---:|---:|
| CU | 10% | 749 | 3 | 0.0040 |
| CU | 20% | 749 | 3 | 0.0040 |
| CU | 30% | 749 | 5 | 0.0067 |
| CU | 40% | 749 | 5 | 0.0067 |
| EV | 10% | 749 | 3 | 0.0040 |
| EV | 20% | 749 | 5 | 0.0067 |
| EV | 30% | 749 | 5 | 0.0067 |
| EV | 40% | 749 | 7 | 0.0093 |
| MD | 10% | 749 | 28 | 0.0374 |
| MD | 20% | 749 | 49 | 0.0654 |
| MD | 30% | 749 | 84 | 0.1121 |
| MD | 40% | 749 | 120 | 0.1602 |
| OU | 10% | 749 | 8 | 0.0107 |
| OU | 20% | 749 | 18 | 0.0240 |
| OU | 30% | 749 | 25 | 0.0334 |
| OU | 40% | 749 | 28 | 0.0374 |

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
