# Uncertainty Sensitivity Report (2026-02-15)

## 1) Setup
- Dataset: `C:\Users\Alexander\Desktop\Education\Aspa\code\output_evidence_campaign_v3_20260215\stage_a\A4\quick_confirm\05_confirmatory\wave_03\lhs_runs_confirm_wave3.csv`
- Implausibility threshold: `2.55`
- SA domain: `nroy`
- Baseline points used: `225`
- Reductions: `10%, 20%, 30%, 40%`

## 2) Baseline Uncertainty Decomposition
| Metric | OU var | EV var | MD var | CU var (median) | Total var (median) |
|---|---:|---:|---:|---:|---:|
| AvgPrice_mean | 0.0025 | 0.000488822 | 0.00581101 | 0.000347069 | 0.0091469 |
| BankBailedOut_share | 0.0004 | 0.000162037 | 0.00088209 | 7.12139e-05 | 0.00151534 |
| BankBailoutAmount_mean | 0.04 | 0.0460119 | 0.0527529 | 0.041519 | 0.180284 |
| BankResolutionHaircut_mean | 0.0001 | 0.000183506 | 0.00039204 | 0.000113583 | 0.000789129 |
| BankResolved_share | 0.0009 | 0.00146065 | 0.00245025 | 0.000901258 | 0.00571216 |
| Consumption_mean | 100 | 31.9583 | 170.773 | 18.6612 | 321.392 |
| DefaultsHH_rate | 4e-06 | 2.09815e-06 | 8.8209e-06 | 1.23914e-06 | 1.61582e-05 |
| HH_Deposit_mean | 1 | 0.0689492 | 1.31882 | 0.0121253 | 2.3999 |
| Output_mean | 100 | 30.5012 | 170.773 | 19.5013 | 320.775 |
| PriceDispersion_mean | 0.0036 | 0.000431827 | 0.0141134 | 0.000215182 | 0.0183604 |
| Transfers_mean | 4 | 0.221023 | 5.27529 | 0.141115 | 9.63743 |

## 3) SA Results (newly implausible share)
| Component | Reduction | Points | Newly implausible | Share |
|---|---:|---:|---:|---:|
| CU | 10% | 225 | 0 | 0.0000 |
| CU | 20% | 225 | 0 | 0.0000 |
| CU | 30% | 225 | 0 | 0.0000 |
| CU | 40% | 225 | 6 | 0.0267 |
| EV | 10% | 225 | 0 | 0.0000 |
| EV | 20% | 225 | 0 | 0.0000 |
| EV | 30% | 225 | 6 | 0.0267 |
| EV | 40% | 225 | 9 | 0.0400 |
| MD | 10% | 225 | 9 | 0.0400 |
| MD | 20% | 225 | 36 | 0.1600 |
| MD | 30% | 225 | 54 | 0.2400 |
| MD | 40% | 225 | 69 | 0.3067 |
| OU | 10% | 225 | 3 | 0.0133 |
| OU | 20% | 225 | 21 | 0.0933 |
| OU | 30% | 225 | 33 | 0.1467 |
| OU | 40% | 225 | 48 | 0.2133 |

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
