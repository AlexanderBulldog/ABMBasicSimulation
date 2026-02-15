# Uncertainty Sensitivity Report (2026-02-15)

## 1) Setup
- Dataset: `C:\Users\Alexander\Desktop\Education\Aspa\code\output_evidence_campaign_v3_20260215\stage_b\R2\waves\wave_03\lhs_runs_wave3.csv`
- Implausibility threshold: `2.55`
- SA domain: `nroy`
- Baseline points used: `468`
- Reductions: `10%, 20%, 30%, 40%`

## 2) Baseline Uncertainty Decomposition
| Metric | OU var | EV var | MD var | CU var (median) | Total var (median) |
|---|---:|---:|---:|---:|---:|
| AvgPrice_mean | 0.0025 | 0.000520728 | 0.00315091 | 0.000273752 | 0.00644539 |
| BankBailedOut_share | 0.0004 | 4.82094e-05 | 0.000478297 | 3.00276e-05 | 0.000956534 |
| BankBailoutAmount_mean | 0.04 | 0.0309458 | 0.0286043 | 0.0170626 | 0.116613 |
| BankResolutionHaircut_mean | 0.0001 | 8.02705e-05 | 0.000212576 | 5.71307e-05 | 0.000449978 |
| BankResolved_share | 0.0009 | 0.00112879 | 0.0013286 | 0.000750598 | 0.00410799 |
| Consumption_mean | 100 | 37.7105 | 92.5983 | 63.2055 | 293.514 |
| DefaultsHH_rate | 4e-06 | 1.81136e-06 | 4.78297e-06 | 1.11413e-06 | 1.17085e-05 |
| HH_Deposit_mean | 1 | 0.239036 | 0.715107 | 0.0241654 | 1.97831 |
| Output_mean | 100 | 30.491 | 92.5983 | 54.524 | 277.613 |
| PriceDispersion_mean | 0.0036 | 0.000302733 | 0.00765275 | 0.000410594 | 0.0119661 |
| Transfers_mean | 4 | 0.243059 | 2.86043 | 0.635205 | 7.73869 |

## 3) SA Results (newly implausible share)
| Component | Reduction | Points | Newly implausible | Share |
|---|---:|---:|---:|---:|
| CU | 10% | 468 | 18 | 0.0385 |
| CU | 20% | 468 | 33 | 0.0705 |
| CU | 30% | 468 | 42 | 0.0897 |
| CU | 40% | 468 | 63 | 0.1346 |
| EV | 10% | 468 | 12 | 0.0256 |
| EV | 20% | 468 | 21 | 0.0449 |
| EV | 30% | 468 | 33 | 0.0705 |
| EV | 40% | 468 | 36 | 0.0769 |
| MD | 10% | 468 | 27 | 0.0577 |
| MD | 20% | 468 | 48 | 0.1026 |
| MD | 30% | 468 | 84 | 0.1795 |
| MD | 40% | 468 | 117 | 0.2500 |
| OU | 10% | 468 | 33 | 0.0705 |
| OU | 20% | 468 | 54 | 0.1154 |
| OU | 30% | 468 | 87 | 0.1859 |
| OU | 40% | 468 | 126 | 0.2692 |

## 4) Interpretation
- Dominant component (rank at max reduction): `OU`
- Investment guidance:
  - `CU` dominant: increase design density / improve GP specification.
  - `EV` dominant: increase replicate seeds `K` or refine stochastic structure.
  - `OU` dominant: improve measurement quality / uncertainty model of observations.
  - `MD` dominant: revisit model structure or discrepancy assumptions.

## 5) Limitations
- EV is treated as metric-level scalar (not x-dependent).
- SA is local to selected baseline domain.
- MD remains expert-specified and is not inferred from data.
