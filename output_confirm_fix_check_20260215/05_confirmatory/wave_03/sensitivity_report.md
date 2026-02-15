# Uncertainty Sensitivity Report (2026-02-15)

## 1) Setup
- Dataset: `C:\Users\Alexander\Desktop\Education\Aspa\code\output_confirm_fix_check_20260215\05_confirmatory\wave_03\lhs_runs_confirm_wave3.csv`
- Implausibility threshold: `2.6`
- SA domain: `nroy`
- Baseline points used: `495`
- Reductions: `10%, 20%, 30%, 40%`

## 2) Baseline Uncertainty Decomposition
| Metric | OU var | EV var | MD var | CU var (median) | Total var (median) |
|---|---:|---:|---:|---:|---:|
| AvgPrice_mean | 0.0025 | 0.000417267 | 0.00717409 | 0.000225569 | 0.0103169 |
| BankBailedOut_share | 0.0004 | 6.19835e-05 | 0.001089 | 3.25234e-05 | 0.00158351 |
| BankBailoutAmount_mean | 0.04 | 0.0275097 | 0.065127 | 0.0155945 | 0.148231 |
| BankResolutionHaircut_mean | 0.0001 | 9.24903e-05 | 0.000484 | 4.93257e-05 | 0.000725816 |
| BankResolved_share | 0.0009 | 0.0010792 | 0.003025 | 0.000742188 | 0.00574639 |
| Consumption_mean | 100 | 35.3955 | 210.83 | 19.9771 | 366.203 |
| DefaultsHH_rate | 4e-06 | 1.32383e-06 | 1.089e-05 | 8.06883e-07 | 1.70207e-05 |
| HH_Deposit_mean | 1 | 0.166299 | 1.62818 | 0.0334124 | 2.82789 |
| Output_mean | 100 | 37.8425 | 210.83 | 21.8749 | 370.548 |
| PriceDispersion_mean | 0.0036 | 0.00034177 | 0.017424 | 0.0003368 | 0.0217026 |
| Transfers_mean | 4 | 0.242756 | 6.5127 | 0.13005 | 10.8855 |

## 3) SA Results (newly implausible share)
| Component | Reduction | Points | Newly implausible | Share |
|---|---:|---:|---:|---:|
| CU | 10% | 495 | 0 | 0.0000 |
| CU | 20% | 495 | 0 | 0.0000 |
| CU | 30% | 495 | 6 | 0.0121 |
| CU | 40% | 495 | 6 | 0.0121 |
| EV | 10% | 495 | 0 | 0.0000 |
| EV | 20% | 495 | 6 | 0.0121 |
| EV | 30% | 495 | 9 | 0.0182 |
| EV | 40% | 495 | 27 | 0.0545 |
| MD | 10% | 495 | 36 | 0.0727 |
| MD | 20% | 495 | 84 | 0.1697 |
| MD | 30% | 495 | 120 | 0.2424 |
| MD | 40% | 495 | 156 | 0.3152 |
| OU | 10% | 495 | 6 | 0.0121 |
| OU | 20% | 495 | 30 | 0.0606 |
| OU | 30% | 495 | 51 | 0.1030 |
| OU | 40% | 495 | 78 | 0.1576 |

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
