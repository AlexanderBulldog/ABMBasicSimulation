# Uncertainty Sensitivity Report (2026-02-14)

## 1) Setup
- Dataset: `C:\Users\Alexander\Desktop\Education\Aspa\code\output_research_core_v6_full_20260214_182812\waves\wave_03\lhs_runs_wave3.csv`
- Implausibility threshold: `2.6`
- SA domain: `nroy`
- Baseline points used: `609`
- Reductions: `10%, 20%, 30%, 40%`

## 2) Baseline Uncertainty Decomposition
| Metric | OU var | EV var | MD var | CU var (median) | Total var (median) |
|---|---:|---:|---:|---:|---:|
| AvgPrice_mean | 0.0025 | 0.000391548 | 0.00424749 | 0.000243052 | 0.00738209 |
| BankBailedOut_share | 0.0004 | 8.95317e-05 | 0.000644754 | 5.66477e-05 | 0.00119093 |
| BankBailoutAmount_mean | 0.04 | 0.0338119 | 0.0385591 | 0.0287015 | 0.141073 |
| BankResolutionHaircut_mean | 0.0001 | 9.72368e-05 | 0.000286557 | 8.47777e-05 | 0.000568572 |
| BankResolved_share | 0.0009 | 0.00112259 | 0.00179098 | 0.000687476 | 0.00450105 |
| Consumption_mean | 100 | 40.0142 | 124.824 | 26.5222 | 291.361 |
| DefaultsHH_rate | 4e-06 | 1.73581e-06 | 6.44754e-06 | 1.53624e-06 | 1.37196e-05 |
| HH_Deposit_mean | 1 | 0.259167 | 0.963978 | 0.0232071 | 2.24635 |
| Output_mean | 100 | 36.7218 | 124.824 | 32.8498 | 294.396 |
| PriceDispersion_mean | 0.0036 | 0.000384374 | 0.0103161 | 0.000470963 | 0.0147714 |
| Transfers_mean | 4 | 0.250943 | 3.85591 | 0.20095 | 8.30781 |

## 3) SA Results (newly implausible share)
| Component | Reduction | Points | Newly implausible | Share |
|---|---:|---:|---:|---:|
| CU | 10% | 609 | 15 | 0.0246 |
| CU | 20% | 609 | 27 | 0.0443 |
| CU | 30% | 609 | 36 | 0.0591 |
| CU | 40% | 609 | 42 | 0.0690 |
| EV | 10% | 609 | 18 | 0.0296 |
| EV | 20% | 609 | 33 | 0.0542 |
| EV | 30% | 609 | 39 | 0.0640 |
| EV | 40% | 609 | 51 | 0.0837 |
| MD | 10% | 609 | 48 | 0.0788 |
| MD | 20% | 609 | 84 | 0.1379 |
| MD | 30% | 609 | 129 | 0.2118 |
| MD | 40% | 609 | 162 | 0.2660 |
| OU | 10% | 609 | 36 | 0.0591 |
| OU | 20% | 609 | 57 | 0.0936 |
| OU | 30% | 609 | 105 | 0.1724 |
| OU | 40% | 609 | 135 | 0.2217 |

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
