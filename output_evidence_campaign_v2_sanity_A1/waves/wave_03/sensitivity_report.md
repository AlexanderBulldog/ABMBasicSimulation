# Uncertainty Sensitivity Report (2026-02-15)

## 1) Setup
- Dataset: `C:\Users\Alexander\Desktop\Education\Aspa\code\output_evidence_campaign_v2_sanity_A1\waves\wave_03\lhs_runs_wave3.csv`
- Implausibility threshold: `2.6`
- SA domain: `nroy`
- Baseline points used: `138`
- Reductions: `10%, 20%, 30%, 40%`

## 2) Baseline Uncertainty Decomposition
| Metric | OU var | EV var | MD var | CU var (median) | Total var (median) |
|---|---:|---:|---:|---:|---:|
| AvgPrice_mean | 0.0025 | 0.000521011 | 0.005929 | 0.000300368 | 0.00925038 |
| BankBailedOut_share | 0.0004 | 9.25926e-05 | 0.0009 | 4.53929e-05 | 0.00143799 |
| BankBailoutAmount_mean | 0.04 | 0.0391881 | 0.053824 | 0.0302229 | 0.163235 |
| BankResolutionHaircut_mean | 0.0001 | 0.000203483 | 0.0004 | 8.64252e-05 | 0.000789908 |
| BankResolved_share | 0.0009 | 0.00146759 | 0.0025 | 0.000809065 | 0.00567666 |
| Consumption_mean | 100 | 45.5205 | 174.24 | 24.3712 | 344.132 |
| DefaultsHH_rate | 4e-06 | 2.50093e-06 | 9e-06 | 1.79423e-06 | 1.72952e-05 |
| HH_Deposit_mean | 1 | 0.117453 | 1.3456 | 0.0164315 | 2.47948 |
| Output_mean | 100 | 51.856 | 174.24 | 24.5617 | 350.658 |
| PriceDispersion_mean | 0.0036 | 0.000283466 | 0.0144 | 0.000175597 | 0.0184591 |
| Transfers_mean | 4 | 0.410502 | 5.3824 | 0.194397 | 9.9873 |

## 3) SA Results (newly implausible share)
| Component | Reduction | Points | Newly implausible | Share |
|---|---:|---:|---:|---:|
| CU | 10% | 138 | 3 | 0.0217 |
| CU | 20% | 138 | 3 | 0.0217 |
| CU | 30% | 138 | 3 | 0.0217 |
| CU | 40% | 138 | 6 | 0.0435 |
| EV | 10% | 138 | 3 | 0.0217 |
| EV | 20% | 138 | 6 | 0.0435 |
| EV | 30% | 138 | 9 | 0.0652 |
| EV | 40% | 138 | 9 | 0.0652 |
| MD | 10% | 138 | 9 | 0.0652 |
| MD | 20% | 138 | 24 | 0.1739 |
| MD | 30% | 138 | 33 | 0.2391 |
| MD | 40% | 138 | 45 | 0.3261 |
| OU | 10% | 138 | 6 | 0.0435 |
| OU | 20% | 138 | 9 | 0.0652 |
| OU | 30% | 138 | 18 | 0.1304 |
| OU | 40% | 138 | 24 | 0.1739 |

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
