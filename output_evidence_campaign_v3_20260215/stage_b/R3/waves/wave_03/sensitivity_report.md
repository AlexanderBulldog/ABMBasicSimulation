# Uncertainty Sensitivity Report (2026-02-15)

## 1) Setup
- Dataset: `C:\Users\Alexander\Desktop\Education\Aspa\code\output_evidence_campaign_v3_20260215\stage_b\R3\waves\wave_03\lhs_runs_wave3.csv`
- Implausibility threshold: `2.55`
- SA domain: `nroy`
- Baseline points used: `375`
- Reductions: `10%, 20%, 30%, 40%`

## 2) Baseline Uncertainty Decomposition
| Metric | OU var | EV var | MD var | CU var (median) | Total var (median) |
|---|---:|---:|---:|---:|---:|
| AvgPrice_mean | 0.0025 | 0.000411102 | 0.0105036 | 0.000424497 | 0.0138392 |
| BankBailedOut_share | 0.0004 | 6.19835e-05 | 0.0015944 | 5.08318e-05 | 0.00210722 |
| BankBailoutAmount_mean | 0.04 | 0.0193985 | 0.0953525 | 0.01452 | 0.169271 |
| BankResolutionHaircut_mean | 0.0001 | 7.57883e-05 | 0.000708624 | 6.94972e-05 | 0.00095391 |
| BankResolved_share | 0.0009 | 0.00126446 | 0.0044289 | 0.0006312 | 0.00722457 |
| Consumption_mean | 100 | 36.421 | 308.677 | 47.4654 | 492.563 |
| DefaultsHH_rate | 4e-06 | 2.22107e-06 | 1.5944e-05 | 9.39972e-07 | 2.31051e-05 |
| Employment_mean | 25 | 19.3673 | 77.1692 | 29.4794 | 151.016 |
| HH_Deposit_mean | 1 | 0.271091 | 2.38381 | 0.031054 | 3.68596 |
| Output_mean | 100 | 30.3111 | 308.677 | 42.5523 | 481.54 |
| PriceDispersion_mean | 0.0036 | 0.000448451 | 0.0255105 | 0.000431707 | 0.0299906 |
| Transfers_mean | 4 | 0.23586 | 9.53525 | 0.382674 | 14.1538 |

## 3) SA Results (newly implausible share)
| Component | Reduction | Points | Newly implausible | Share |
|---|---:|---:|---:|---:|
| CU | 10% | 375 | 48 | 0.1280 |
| CU | 20% | 375 | 72 | 0.1920 |
| CU | 30% | 375 | 108 | 0.2880 |
| CU | 40% | 375 | 165 | 0.4400 |
| EV | 10% | 375 | 36 | 0.0960 |
| EV | 20% | 375 | 57 | 0.1520 |
| EV | 30% | 375 | 69 | 0.1840 |
| EV | 40% | 375 | 90 | 0.2400 |
| MD | 10% | 375 | 90 | 0.2400 |
| MD | 20% | 375 | 198 | 0.5280 |
| MD | 30% | 375 | 270 | 0.7200 |
| MD | 40% | 375 | 300 | 0.8000 |
| OU | 10% | 375 | 39 | 0.1040 |
| OU | 20% | 375 | 66 | 0.1760 |
| OU | 30% | 375 | 87 | 0.2320 |
| OU | 40% | 375 | 132 | 0.3520 |

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
