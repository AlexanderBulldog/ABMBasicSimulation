# Uncertainty Sensitivity Report (2026-02-15)

## 1) Setup
- Dataset: `C:\Users\Alexander\Desktop\Education\Aspa\code\output_evidence_campaign_v3_20260215\stage_a\A2\quick_confirm\05_confirmatory\wave_03\lhs_runs_confirm_wave3.csv`
- Implausibility threshold: `2.55`
- SA domain: `nroy`
- Baseline points used: `291`
- Reductions: `10%, 20%, 30%, 40%`

## 2) Baseline Uncertainty Decomposition
| Metric | OU var | EV var | MD var | CU var (median) | Total var (median) |
|---|---:|---:|---:|---:|---:|
| AvgPrice_mean | 0.0025 | 0.00045623 | 0.00691559 | 0.000285433 | 0.0101572 |
| BankBailedOut_share | 0.0004 | 9.25926e-05 | 0.00104976 | 6.57006e-05 | 0.00160805 |
| BankBailoutAmount_mean | 0.04 | 0.0595827 | 0.0627803 | 0.0348047 | 0.197168 |
| BankResolutionHaircut_mean | 0.0001 | 0.000182422 | 0.00046656 | 0.000141788 | 0.000890771 |
| BankResolved_share | 0.0009 | 0.00142593 | 0.002916 | 0.000742207 | 0.00598413 |
| Consumption_mean | 100 | 37.4169 | 203.234 | 30.6953 | 371.346 |
| DefaultsHH_rate | 4e-06 | 1.34606e-06 | 1.04976e-05 | 1.55813e-06 | 1.74018e-05 |
| HH_Deposit_mean | 1 | 0.0753901 | 1.56951 | 0.0128536 | 2.65775 |
| Output_mean | 100 | 34.8462 | 203.234 | 31.3184 | 369.398 |
| PriceDispersion_mean | 0.0036 | 0.000377383 | 0.0167962 | 0.000237858 | 0.0210114 |
| Transfers_mean | 4 | 0.310215 | 6.27803 | 0.203938 | 10.7922 |

## 3) SA Results (newly implausible share)
| Component | Reduction | Points | Newly implausible | Share |
|---|---:|---:|---:|---:|
| CU | 10% | 291 | 0 | 0.0000 |
| CU | 20% | 291 | 0 | 0.0000 |
| CU | 30% | 291 | 3 | 0.0103 |
| CU | 40% | 291 | 3 | 0.0103 |
| EV | 10% | 291 | 0 | 0.0000 |
| EV | 20% | 291 | 3 | 0.0103 |
| EV | 30% | 291 | 3 | 0.0103 |
| EV | 40% | 291 | 9 | 0.0309 |
| MD | 10% | 291 | 12 | 0.0412 |
| MD | 20% | 291 | 30 | 0.1031 |
| MD | 30% | 291 | 42 | 0.1443 |
| MD | 40% | 291 | 66 | 0.2268 |
| OU | 10% | 291 | 3 | 0.0103 |
| OU | 20% | 291 | 12 | 0.0412 |
| OU | 30% | 291 | 24 | 0.0825 |
| OU | 40% | 291 | 30 | 0.1031 |

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
