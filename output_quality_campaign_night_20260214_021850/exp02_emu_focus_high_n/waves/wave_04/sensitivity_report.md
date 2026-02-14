# Uncertainty Sensitivity Report (2026-02-14)

## 1) Setup
- Dataset: `C:\Users\Alexander\Desktop\Education\Aspa\code\output_quality_campaign_night_20260214_021850\exp02_emu_focus_high_n\waves\wave_04\lhs_runs_wave4.csv`
- Implausibility threshold: `2.75`
- SA domain: `nroy`
- Baseline points used: `381`
- Reductions: `10%, 20%, 30%, 40%`

## 2) Baseline Uncertainty Decomposition
| Metric | OU var | EV var | MD var | CU var (median) | Total var (median) |
|---|---:|---:|---:|---:|---:|
| AvgPrice_mean | 0.0025 | 0.000549371 | 0.00682735 | 0.000395325 | 0.010272 |
| BankBailedOut_share | 0.0004 | 5.92593e-05 | 0.00103637 | 3.66064e-05 | 0.00153223 |
| BankBailoutAmount_mean | 0.04 | 0.00837802 | 0.0619793 | 0.00501785 | 0.115375 |
| BankResolutionHaircut_mean | 0.0001 | 0.000197583 | 0.000460607 | 0.00011405 | 0.00087224 |
| BankResolved_share | 0.0009 | 0.00152593 | 0.00287879 | 0.00103415 | 0.00633887 |
| Consumption_mean | 100 | 40.602 | 200.64 | 48.6402 | 389.883 |
| CreditRejections_mean | 25 | 23.0523 | 460.607 | 13.602 | 522.262 |
| DefaultsHH_rate | 4e-06 | 1.79215e-06 | 1.03637e-05 | 1.4277e-06 | 1.75835e-05 |
| HH_Deposit_mean | 1 | 1.90483 | 1.54948 | 1.55815 | 6.01246 |
| Output_mean | 100 | 33.6296 | 200.64 | 42.0072 | 376.277 |
| PriceDispersion_mean | 0.0036 | 0.000552215 | 0.0165819 | 0.000255792 | 0.0209899 |
| Transfers_mean | 4 | 0.347742 | 6.19793 | 0.379083 | 10.9248 |

## 3) SA Results (newly implausible share)
| Component | Reduction | Points | Newly implausible | Share |
|---|---:|---:|---:|---:|
| CU | 10% | 381 | 18 | 0.0472 |
| CU | 20% | 381 | 36 | 0.0945 |
| CU | 30% | 381 | 48 | 0.1260 |
| CU | 40% | 381 | 57 | 0.1496 |
| EV | 10% | 381 | 21 | 0.0551 |
| EV | 20% | 381 | 27 | 0.0709 |
| EV | 30% | 381 | 33 | 0.0866 |
| EV | 40% | 381 | 45 | 0.1181 |
| MD | 10% | 381 | 48 | 0.1260 |
| MD | 20% | 381 | 87 | 0.2283 |
| MD | 30% | 381 | 117 | 0.3071 |
| MD | 40% | 381 | 147 | 0.3858 |
| OU | 10% | 381 | 21 | 0.0551 |
| OU | 20% | 381 | 45 | 0.1181 |
| OU | 30% | 381 | 72 | 0.1890 |
| OU | 40% | 381 | 87 | 0.2283 |

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
