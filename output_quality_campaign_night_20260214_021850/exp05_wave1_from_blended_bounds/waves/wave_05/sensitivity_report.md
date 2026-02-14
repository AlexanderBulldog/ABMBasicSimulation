# Uncertainty Sensitivity Report (2026-02-14)

## 1) Setup
- Dataset: `C:\Users\Alexander\Desktop\Education\Aspa\code\output_quality_campaign_night_20260214_021850\exp05_wave1_from_blended_bounds\waves\wave_05\lhs_runs_wave5.csv`
- Implausibility threshold: `2.75`
- SA domain: `nroy`
- Baseline points used: `417`
- Reductions: `10%, 20%, 30%, 40%`

## 2) Baseline Uncertainty Decomposition
| Metric | OU var | EV var | MD var | CU var (median) | Total var (median) |
|---|---:|---:|---:|---:|---:|
| AvgPrice_mean | 0.0025 | 0.00032322 | 0.00359508 | 0.000196938 | 0.00661524 |
| Consumption_mean | 100 | 11.008 | 105.651 | 6.08331 | 222.743 |
| DefaultsHH_rate | 4e-06 | 2.13778e-07 | 5.4572e-06 | 1.1395e-07 | 9.78492e-06 |
| Output_mean | 100 | 12.9836 | 105.651 | 7.85538 | 226.49 |
| Transfers_mean | 4 | 0.0844989 | 3.26365 | 0.0429 | 7.39104 |

## 3) SA Results (newly implausible share)
| Component | Reduction | Points | Newly implausible | Share |
|---|---:|---:|---:|---:|
| CU | 10% | 417 | 6 | 0.0144 |
| CU | 20% | 417 | 27 | 0.0647 |
| CU | 30% | 417 | 33 | 0.0791 |
| CU | 40% | 417 | 45 | 0.1079 |
| EV | 10% | 417 | 21 | 0.0504 |
| EV | 20% | 417 | 39 | 0.0935 |
| EV | 30% | 417 | 51 | 0.1223 |
| EV | 40% | 417 | 75 | 0.1799 |
| MD | 10% | 417 | 180 | 0.4317 |
| MD | 20% | 417 | 315 | 0.7554 |
| MD | 30% | 417 | 360 | 0.8633 |
| MD | 40% | 417 | 399 | 0.9568 |
| OU | 10% | 417 | 174 | 0.4173 |
| OU | 20% | 417 | 300 | 0.7194 |
| OU | 30% | 417 | 354 | 0.8489 |
| OU | 40% | 417 | 384 | 0.9209 |

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
