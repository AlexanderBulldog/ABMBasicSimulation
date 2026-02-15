# Uncertainty Sensitivity Report (2026-02-15)

## 1) Setup
- Dataset: `C:\Users\Alexander\Desktop\Education\Aspa\code\output_evidence_campaign_20260214\stage_a\C3\waves\wave_03\lhs_runs_wave3.csv`
- Implausibility threshold: `2.75`
- SA domain: `nroy`
- Baseline points used: `252`
- Reductions: `10%, 20%, 30%, 40%`

## 2) Baseline Uncertainty Decomposition
| Metric | OU var | EV var | MD var | CU var (median) | Total var (median) |
|---|---:|---:|---:|---:|---:|
| AvgPrice_mean | 0.0025 | 0.000447333 | 0.00359508 | 0.000377066 | 0.00691948 |
| BankBailedOut_share | 0.0004 | 9.9537e-05 | 0.00054572 | 6.97327e-05 | 0.00111499 |
| BankBailoutAmount_mean | 0.04 | 0.0594614 | 0.0326365 | 0.0465658 | 0.178664 |
| BankResolutionHaircut_mean | 0.0001 | 0.000158659 | 0.000242542 | 9.43431e-05 | 0.000595544 |
| BankResolved_share | 0.0009 | 0.00131944 | 0.00151589 | 0.000853739 | 0.00458907 |
| Consumption_mean | 100 | 45.7247 | 105.651 | 36.0006 | 287.377 |
| DefaultsHH_rate | 4e-06 | 3.03773e-06 | 5.4572e-06 | 1.37855e-06 | 1.38735e-05 |
| HH_Deposit_mean | 1 | 0.156279 | 0.815911 | 0.0139069 | 1.9861 |
| Output_mean | 100 | 55.1973 | 105.651 | 38.0994 | 298.948 |
| PriceDispersion_mean | 0.0036 | 0.000409339 | 0.00873151 | 0.000233385 | 0.0129742 |
| Transfers_mean | 4 | 0.397407 | 3.26365 | 0.242049 | 7.9031 |

## 3) SA Results (newly implausible share)
| Component | Reduction | Points | Newly implausible | Share |
|---|---:|---:|---:|---:|
| CU | 10% | 252 | 6 | 0.0238 |
| CU | 20% | 252 | 6 | 0.0238 |
| CU | 30% | 252 | 6 | 0.0238 |
| CU | 40% | 252 | 9 | 0.0357 |
| EV | 10% | 252 | 6 | 0.0238 |
| EV | 20% | 252 | 6 | 0.0238 |
| EV | 30% | 252 | 6 | 0.0238 |
| EV | 40% | 252 | 18 | 0.0714 |
| MD | 10% | 252 | 6 | 0.0238 |
| MD | 20% | 252 | 21 | 0.0833 |
| MD | 30% | 252 | 42 | 0.1667 |
| MD | 40% | 252 | 66 | 0.2619 |
| OU | 10% | 252 | 6 | 0.0238 |
| OU | 20% | 252 | 9 | 0.0357 |
| OU | 30% | 252 | 39 | 0.1548 |
| OU | 40% | 252 | 57 | 0.2262 |

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
