# Uncertainty Sensitivity Report (2026-02-14)

## 1) Setup
- Dataset: `C:\Users\Alexander\Desktop\Education\Aspa\code\output_research_core_v4_drycheck\waves\wave_03\lhs_runs_wave3.csv`
- Implausibility threshold: `2.6`
- SA domain: `nroy`
- Baseline points used: `45`
- Reductions: `10%, 20%, 30%, 40%`

## 2) Baseline Uncertainty Decomposition
| Metric | OU var | EV var | MD var | CU var (median) | Total var (median) |
|---|---:|---:|---:|---:|---:|
| AvgPrice_mean | 0.0025 | 0.00101616 | 0.00459142 | 0.000518966 | 0.00862654 |
| BankBailedOut_share | 0.0004 | 0.000648148 | 0.00069696 | 0.000294131 | 0.00203924 |
| BankBailoutAmount_mean | 0.04 | 0.405817 | 0.0416813 | 0.198283 | 0.68578 |
| BankResolutionHaircut_mean | 0.0001 | 0.000790067 | 0.00030976 | 0.000600597 | 0.00180042 |
| BankResolved_share | 0.0009 | 0.00234259 | 0.001936 | 0.00152507 | 0.00670366 |
| DefaultsHH_rate | 4e-06 | 6.7037e-06 | 6.9696e-06 | 7.89182e-06 | 2.55651e-05 |
| Employment_mean | 25 | 119.911 | 33.7329 | 127.909 | 306.553 |
| HH_Deposit_mean | 1 | 0.058514 | 1.04203 | 0.0140715 | 2.11462 |
| InventoryGap_mean | 0.0625 | 0.404969 | 0.094864 | 0.218553 | 0.780886 |
| Output_mean | 100 | 212.802 | 134.931 | 155.479 | 603.212 |
| PriceDispersion_mean | 0.0036 | 0.00041177 | 0.0111514 | 0.000307476 | 0.0154706 |

## 3) SA Results (newly implausible share)
| Component | Reduction | Points | Newly implausible | Share |
|---|---:|---:|---:|---:|
| CU | 10% | 45 | 0 | 0.0000 |
| CU | 20% | 45 | 0 | 0.0000 |
| CU | 30% | 45 | 0 | 0.0000 |
| CU | 40% | 45 | 0 | 0.0000 |
| EV | 10% | 45 | 0 | 0.0000 |
| EV | 20% | 45 | 0 | 0.0000 |
| EV | 30% | 45 | 0 | 0.0000 |
| EV | 40% | 45 | 0 | 0.0000 |
| MD | 10% | 45 | 0 | 0.0000 |
| MD | 20% | 45 | 0 | 0.0000 |
| MD | 30% | 45 | 3 | 0.0667 |
| MD | 40% | 45 | 3 | 0.0667 |
| OU | 10% | 45 | 0 | 0.0000 |
| OU | 20% | 45 | 0 | 0.0000 |
| OU | 30% | 45 | 0 | 0.0000 |
| OU | 40% | 45 | 0 | 0.0000 |

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
