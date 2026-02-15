# Uncertainty Sensitivity Report (2026-02-15)

## 1) Setup
- Dataset: `C:\Users\Alexander\Desktop\Education\Aspa\code\output_evidence_campaign_v3_20260215\stage_a\A1\quick_confirm\05_confirmatory\wave_03\lhs_runs_confirm_wave3.csv`
- Implausibility threshold: `2.6`
- SA domain: `nroy`
- Baseline points used: `318`
- Reductions: `10%, 20%, 30%, 40%`

## 2) Baseline Uncertainty Decomposition
| Metric | OU var | EV var | MD var | CU var (median) | Total var (median) |
|---|---:|---:|---:|---:|---:|
| AvgPrice_mean | 0.0025 | 0.000562075 | 0.00806634 | 0.000432917 | 0.0115613 |
| BankBailedOut_share | 0.0004 | 9.25926e-05 | 0.00122444 | 5.9579e-05 | 0.00177661 |
| BankBailoutAmount_mean | 0.04 | 0.0713277 | 0.073227 | 0.0368003 | 0.221355 |
| BankResolutionHaircut_mean | 0.0001 | 0.00025868 | 0.000544196 | 0.000122686 | 0.00102556 |
| BankResolved_share | 0.0009 | 0.00169676 | 0.00340122 | 0.000791611 | 0.00678959 |
| Consumption_mean | 100 | 33.7398 | 237.052 | 30.0855 | 400.877 |
| DefaultsHH_rate | 4e-06 | 2.86343e-06 | 1.22444e-05 | 3.22935e-06 | 2.23372e-05 |
| HH_Deposit_mean | 1 | 0.219422 | 1.83067 | 0.0253919 | 3.07549 |
| Output_mean | 100 | 41.3482 | 237.052 | 25.023 | 403.423 |
| PriceDispersion_mean | 0.0036 | 0.00038698 | 0.019591 | 0.000248118 | 0.0238261 |
| Transfers_mean | 4 | 0.277452 | 7.3227 | 0.211883 | 11.812 |

## 3) SA Results (newly implausible share)
| Component | Reduction | Points | Newly implausible | Share |
|---|---:|---:|---:|---:|
| CU | 10% | 318 | 0 | 0.0000 |
| CU | 20% | 318 | 9 | 0.0283 |
| CU | 30% | 318 | 9 | 0.0283 |
| CU | 40% | 318 | 15 | 0.0472 |
| EV | 10% | 318 | 6 | 0.0189 |
| EV | 20% | 318 | 15 | 0.0472 |
| EV | 30% | 318 | 15 | 0.0472 |
| EV | 40% | 318 | 27 | 0.0849 |
| MD | 10% | 318 | 21 | 0.0660 |
| MD | 20% | 318 | 42 | 0.1321 |
| MD | 30% | 318 | 51 | 0.1604 |
| MD | 40% | 318 | 75 | 0.2358 |
| OU | 10% | 318 | 9 | 0.0283 |
| OU | 20% | 318 | 18 | 0.0566 |
| OU | 30% | 318 | 33 | 0.1038 |
| OU | 40% | 318 | 36 | 0.1132 |

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
