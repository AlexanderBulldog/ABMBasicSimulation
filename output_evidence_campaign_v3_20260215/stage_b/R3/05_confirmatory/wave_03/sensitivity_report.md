# Uncertainty Sensitivity Report (2026-02-15)

## 1) Setup
- Dataset: `C:\Users\Alexander\Desktop\Education\Aspa\code\output_evidence_campaign_v3_20260215\stage_b\R3\05_confirmatory\wave_03\lhs_runs_confirm_wave3.csv`
- Implausibility threshold: `2.55`
- SA domain: `nroy`
- Baseline points used: `1047`
- Reductions: `10%, 20%, 30%, 40%`

## 2) Baseline Uncertainty Decomposition
| Metric | OU var | EV var | MD var | CU var (median) | Total var (median) |
|---|---:|---:|---:|---:|---:|
| AvgPrice_mean | 0.0025 | 0.000381839 | 0.0105036 | 0.0001706 | 0.013556 |
| BankBailedOut_share | 0.0004 | 6.19835e-05 | 0.0015944 | 3.12838e-05 | 0.00208767 |
| BankBailoutAmount_mean | 0.04 | 0.0272024 | 0.0953525 | 0.0216405 | 0.184195 |
| BankResolutionHaircut_mean | 0.0001 | 8.77864e-05 | 0.000708624 | 5.16796e-05 | 0.00094809 |
| BankResolved_share | 0.0009 | 0.00153719 | 0.0044289 | 0.000712231 | 0.00757832 |
| Consumption_mean | 100 | 30.2773 | 308.677 | 18.1386 | 457.093 |
| DefaultsHH_rate | 4e-06 | 2.20475e-06 | 1.5944e-05 | 1.119e-06 | 2.32678e-05 |
| HH_Deposit_mean | 1 | 0.199749 | 2.38381 | 0.0256253 | 3.60919 |
| Output_mean | 100 | 29.0852 | 308.677 | 13.691 | 451.453 |
| PriceDispersion_mean | 0.0036 | 0.000428192 | 0.0255105 | 0.000259424 | 0.0297981 |
| Transfers_mean | 4 | 0.215025 | 9.53525 | 0.100325 | 13.8506 |

## 3) SA Results (newly implausible share)
| Component | Reduction | Points | Newly implausible | Share |
|---|---:|---:|---:|---:|
| CU | 10% | 1047 | 0 | 0.0000 |
| CU | 20% | 1047 | 0 | 0.0000 |
| CU | 30% | 1047 | 0 | 0.0000 |
| CU | 40% | 1047 | 0 | 0.0000 |
| EV | 10% | 1047 | 0 | 0.0000 |
| EV | 20% | 1047 | 0 | 0.0000 |
| EV | 30% | 1047 | 0 | 0.0000 |
| EV | 40% | 1047 | 0 | 0.0000 |
| MD | 10% | 1047 | 0 | 0.0000 |
| MD | 20% | 1047 | 9 | 0.0086 |
| MD | 30% | 1047 | 36 | 0.0344 |
| MD | 40% | 1047 | 156 | 0.1490 |
| OU | 10% | 1047 | 0 | 0.0000 |
| OU | 20% | 1047 | 0 | 0.0000 |
| OU | 30% | 1047 | 0 | 0.0000 |
| OU | 40% | 1047 | 0 | 0.0000 |

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
