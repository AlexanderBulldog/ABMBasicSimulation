# Uncertainty Sensitivity Report (2026-02-15)

## 1) Setup
- Dataset: `C:\Users\Alexander\Desktop\Education\Aspa\code\output_evidence_campaign_20260214\stage_a\C2\waves\wave_03\lhs_runs_wave3.csv`
- Implausibility threshold: `2.75`
- SA domain: `nroy`
- Baseline points used: `267`
- Reductions: `10%, 20%, 30%, 40%`

## 2) Baseline Uncertainty Decomposition
| Metric | OU var | EV var | MD var | CU var (median) | Total var (median) |
|---|---:|---:|---:|---:|---:|
| AvgPrice_mean | 0.0025 | 0.000447333 | 0.00409024 | 0.000376714 | 0.00741428 |
| BankBailedOut_share | 0.0004 | 9.9537e-05 | 0.000620883 | 6.97327e-05 | 0.00119015 |
| BankBailoutAmount_mean | 0.04 | 0.0594614 | 0.0371316 | 0.0465658 | 0.183159 |
| BankResolutionHaircut_mean | 0.0001 | 0.000158659 | 0.000275948 | 9.43431e-05 | 0.00062895 |
| BankResolved_share | 0.0009 | 0.00131944 | 0.00172467 | 0.000853739 | 0.00479786 |
| Consumption_mean | 100 | 45.7247 | 120.203 | 36.0006 | 301.928 |
| DefaultsHH_rate | 4e-06 | 3.03773e-06 | 6.20883e-06 | 1.37725e-06 | 1.46238e-05 |
| HH_Deposit_mean | 1 | 0.156279 | 0.928289 | 0.0139069 | 2.09847 |
| Output_mean | 100 | 55.1973 | 120.203 | 38.0994 | 313.5 |
| PriceDispersion_mean | 0.0036 | 0.000409339 | 0.00993412 | 0.000233385 | 0.0141768 |
| Transfers_mean | 4 | 0.397407 | 3.71316 | 0.241869 | 8.35243 |

## 3) SA Results (newly implausible share)
| Component | Reduction | Points | Newly implausible | Share |
|---|---:|---:|---:|---:|
| CU | 10% | 267 | 0 | 0.0000 |
| CU | 20% | 267 | 3 | 0.0112 |
| CU | 30% | 267 | 3 | 0.0112 |
| CU | 40% | 267 | 9 | 0.0337 |
| EV | 10% | 267 | 3 | 0.0112 |
| EV | 20% | 267 | 3 | 0.0112 |
| EV | 30% | 267 | 12 | 0.0449 |
| EV | 40% | 267 | 18 | 0.0674 |
| MD | 10% | 267 | 9 | 0.0337 |
| MD | 20% | 267 | 21 | 0.0787 |
| MD | 30% | 267 | 39 | 0.1461 |
| MD | 40% | 267 | 66 | 0.2472 |
| OU | 10% | 267 | 3 | 0.0112 |
| OU | 20% | 267 | 21 | 0.0787 |
| OU | 30% | 267 | 21 | 0.0787 |
| OU | 40% | 267 | 33 | 0.1236 |

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
