# Uncertainty Sensitivity Report (2026-02-15)

## 1) Setup
- Dataset: `C:\Users\Alexander\Desktop\Education\Aspa\code\output_evidence_campaign_v2_20260215\stage_b\R3\05_confirmatory\wave_03\lhs_runs_confirm_wave3.csv`
- Implausibility threshold: `2.6`
- SA domain: `nroy`
- Baseline points used: `1044`
- Reductions: `10%, 20%, 30%, 40%`

## 2) Baseline Uncertainty Decomposition
| Metric | OU var | EV var | MD var | CU var (median) | Total var (median) |
|---|---:|---:|---:|---:|---:|
| AvgPrice_mean | 0.0025 | 0.000311235 | 0.00868065 | 0.000198981 | 0.0116909 |
| BankBailedOut_share | 0.0004 | 4.82094e-05 | 0.00131769 | 2.94963e-05 | 0.0017954 |
| BankBailoutAmount_mean | 0.04 | 0.0298546 | 0.0788037 | 0.0211837 | 0.169842 |
| BankResolutionHaircut_mean | 0.0001 | 6.21709e-05 | 0.00058564 | 3.50229e-05 | 0.000782834 |
| BankResolved_share | 0.0009 | 0.00125207 | 0.00366025 | 0.00054815 | 0.00636047 |
| Consumption_mean | 100 | 33.1813 | 255.105 | 17.5717 | 405.858 |
| DefaultsHH_rate | 4e-06 | 1.46288e-06 | 1.31769e-05 | 9.16481e-07 | 1.95563e-05 |
| HH_Deposit_mean | 1 | 0.294581 | 1.97009 | 0.0310042 | 3.29568 |
| Output_mean | 100 | 34.7614 | 255.105 | 20.9466 | 410.813 |
| PriceDispersion_mean | 0.0036 | 0.000346874 | 0.021083 | 0.000241643 | 0.0252716 |
| Transfers_mean | 4 | 0.225561 | 7.88037 | 0.133417 | 12.2393 |

## 3) SA Results (newly implausible share)
| Component | Reduction | Points | Newly implausible | Share |
|---|---:|---:|---:|---:|
| CU | 10% | 1044 | 0 | 0.0000 |
| CU | 20% | 1044 | 0 | 0.0000 |
| CU | 30% | 1044 | 3 | 0.0029 |
| CU | 40% | 1044 | 3 | 0.0029 |
| EV | 10% | 1044 | 0 | 0.0000 |
| EV | 20% | 1044 | 3 | 0.0029 |
| EV | 30% | 1044 | 3 | 0.0029 |
| EV | 40% | 1044 | 3 | 0.0029 |
| MD | 10% | 1044 | 12 | 0.0115 |
| MD | 20% | 1044 | 45 | 0.0431 |
| MD | 30% | 1044 | 144 | 0.1379 |
| MD | 40% | 1044 | 246 | 0.2356 |
| OU | 10% | 1044 | 3 | 0.0029 |
| OU | 20% | 1044 | 3 | 0.0029 |
| OU | 30% | 1044 | 18 | 0.0172 |
| OU | 40% | 1044 | 33 | 0.0316 |

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
