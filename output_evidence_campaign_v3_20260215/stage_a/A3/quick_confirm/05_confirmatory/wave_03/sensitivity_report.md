# Uncertainty Sensitivity Report (2026-02-15)

## 1) Setup
- Dataset: `C:\Users\Alexander\Desktop\Education\Aspa\code\output_evidence_campaign_v3_20260215\stage_a\A3\quick_confirm\05_confirmatory\wave_03\lhs_runs_confirm_wave3.csv`
- Implausibility threshold: `2.6`
- SA domain: `nroy`
- Baseline points used: `315`
- Reductions: `10%, 20%, 30%, 40%`

## 2) Baseline Uncertainty Decomposition
| Metric | OU var | EV var | MD var | CU var (median) | Total var (median) |
|---|---:|---:|---:|---:|---:|
| AvgPrice_mean | 0.0025 | 0.000461072 | 0.00868065 | 0.000195473 | 0.0118372 |
| BankBailedOut_share | 0.0004 | 9.25926e-05 | 0.00131769 | 7.22604e-05 | 0.00188254 |
| BankBailoutAmount_mean | 0.04 | 0.0577926 | 0.0788037 | 0.0312431 | 0.207839 |
| BankResolutionHaircut_mean | 0.0001 | 0.000205491 | 0.00058564 | 0.00015086 | 0.00104199 |
| BankResolved_share | 0.0009 | 0.00168981 | 0.00366025 | 0.000858074 | 0.00710814 |
| Consumption_mean | 100 | 27.5475 | 255.105 | 15.939 | 398.591 |
| DefaultsHH_rate | 4e-06 | 3.71319e-06 | 1.31769e-05 | 3.46428e-06 | 2.43544e-05 |
| HH_Deposit_mean | 1 | 0.129212 | 1.97009 | 0.0228668 | 3.12217 |
| Output_mean | 100 | 32.0588 | 255.105 | 17.691 | 404.855 |
| PriceDispersion_mean | 0.0036 | 0.000363808 | 0.021083 | 0.000181895 | 0.0252287 |
| Transfers_mean | 4 | 0.270043 | 7.88037 | 0.124654 | 12.2751 |

## 3) SA Results (newly implausible share)
| Component | Reduction | Points | Newly implausible | Share |
|---|---:|---:|---:|---:|
| CU | 10% | 315 | 0 | 0.0000 |
| CU | 20% | 315 | 0 | 0.0000 |
| CU | 30% | 315 | 3 | 0.0095 |
| CU | 40% | 315 | 6 | 0.0190 |
| EV | 10% | 315 | 0 | 0.0000 |
| EV | 20% | 315 | 6 | 0.0190 |
| EV | 30% | 315 | 12 | 0.0381 |
| EV | 40% | 315 | 15 | 0.0476 |
| MD | 10% | 315 | 21 | 0.0667 |
| MD | 20% | 315 | 45 | 0.1429 |
| MD | 30% | 315 | 69 | 0.2190 |
| MD | 40% | 315 | 81 | 0.2571 |
| OU | 10% | 315 | 6 | 0.0190 |
| OU | 20% | 315 | 18 | 0.0571 |
| OU | 30% | 315 | 27 | 0.0857 |
| OU | 40% | 315 | 36 | 0.1143 |

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
