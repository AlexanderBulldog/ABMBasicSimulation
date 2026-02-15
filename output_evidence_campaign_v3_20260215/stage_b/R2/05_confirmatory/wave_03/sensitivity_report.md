# Uncertainty Sensitivity Report (2026-02-15)

## 1) Setup
- Dataset: `C:\Users\Alexander\Desktop\Education\Aspa\code\output_evidence_campaign_v3_20260215\stage_b\R2\05_confirmatory\wave_03\lhs_runs_confirm_wave3.csv`
- Implausibility threshold: `2.55`
- SA domain: `nroy`
- Baseline points used: `798`
- Reductions: `10%, 20%, 30%, 40%`

## 2) Baseline Uncertainty Decomposition
| Metric | OU var | EV var | MD var | CU var (median) | Total var (median) |
|---|---:|---:|---:|---:|---:|
| AvgPrice_mean | 0.0025 | 0.000358452 | 0.00315091 | 0.000177274 | 0.00618664 |
| BankBailedOut_share | 0.0004 | 4.82094e-05 | 0.000478297 | 2.73711e-05 | 0.000953877 |
| BankBailoutAmount_mean | 0.04 | 0.0217273 | 0.0286043 | 0.0158874 | 0.106219 |
| BankResolutionHaircut_mean | 0.0001 | 6.21799e-05 | 0.000212576 | 4.32915e-05 | 0.000418048 |
| BankResolved_share | 0.0009 | 0.000835399 | 0.0013286 | 0.000549465 | 0.00361347 |
| Consumption_mean | 100 | 34.2267 | 92.5983 | 15.8191 | 242.644 |
| DefaultsHH_rate | 4e-06 | 1.52624e-06 | 4.78297e-06 | 1.1418e-06 | 1.1451e-05 |
| HH_Deposit_mean | 1 | 0.172505 | 0.715107 | 0.0197875 | 1.9074 |
| Output_mean | 100 | 26.7427 | 92.5983 | 16.3542 | 235.695 |
| PriceDispersion_mean | 0.0036 | 0.000329235 | 0.00765275 | 0.000173159 | 0.0117551 |
| Transfers_mean | 4 | 0.214799 | 2.86043 | 0.115988 | 7.19121 |

## 3) SA Results (newly implausible share)
| Component | Reduction | Points | Newly implausible | Share |
|---|---:|---:|---:|---:|
| CU | 10% | 798 | 6 | 0.0075 |
| CU | 20% | 798 | 18 | 0.0226 |
| CU | 30% | 798 | 24 | 0.0301 |
| CU | 40% | 798 | 36 | 0.0451 |
| EV | 10% | 798 | 12 | 0.0150 |
| EV | 20% | 798 | 33 | 0.0414 |
| EV | 30% | 798 | 63 | 0.0789 |
| EV | 40% | 798 | 93 | 0.1165 |
| MD | 10% | 798 | 54 | 0.0677 |
| MD | 20% | 798 | 117 | 0.1466 |
| MD | 30% | 798 | 165 | 0.2068 |
| MD | 40% | 798 | 228 | 0.2857 |
| OU | 10% | 798 | 63 | 0.0789 |
| OU | 20% | 798 | 126 | 0.1579 |
| OU | 30% | 798 | 180 | 0.2256 |
| OU | 40% | 798 | 240 | 0.3008 |

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
