# Uncertainty Sensitivity Report (2026-02-14)

## 1) Setup
- Dataset: `C:\Users\Alexander\Desktop\Education\Aspa\code\output_quality_campaign_night_20260214_021850\exp05_wave1_from_blended_bounds\waves\wave_04\lhs_runs_wave4.csv`
- Implausibility threshold: `2.75`
- SA domain: `nroy`
- Baseline points used: `432`
- Reductions: `10%, 20%, 30%, 40%`

## 2) Baseline Uncertainty Decomposition
| Metric | OU var | EV var | MD var | CU var (median) | Total var (median) |
|---|---:|---:|---:|---:|---:|
| AvgPrice_mean | 0.0025 | 0.000332984 | 0.00424749 | 0.00015797 | 0.00723845 |
| Consumption_mean | 100 | 11.1472 | 124.824 | 5.07266 | 241.044 |
| DefaultsHH_rate | 4e-06 | 1.83704e-07 | 6.44754e-06 | 9.5077e-08 | 1.07263e-05 |
| Output_mean | 100 | 12.8916 | 124.824 | 5.78954 | 243.505 |
| Transfers_mean | 4 | 0.0855314 | 3.85591 | 0.0291489 | 7.97059 |

## 3) SA Results (newly implausible share)
| Component | Reduction | Points | Newly implausible | Share |
|---|---:|---:|---:|---:|
| CU | 10% | 432 | 3 | 0.0069 |
| CU | 20% | 432 | 9 | 0.0208 |
| CU | 30% | 432 | 12 | 0.0278 |
| CU | 40% | 432 | 21 | 0.0486 |
| EV | 10% | 432 | 9 | 0.0208 |
| EV | 20% | 432 | 21 | 0.0486 |
| EV | 30% | 432 | 27 | 0.0625 |
| EV | 40% | 432 | 39 | 0.0903 |
| MD | 10% | 432 | 129 | 0.2986 |
| MD | 20% | 432 | 258 | 0.5972 |
| MD | 30% | 432 | 315 | 0.7292 |
| MD | 40% | 432 | 399 | 0.9236 |
| OU | 10% | 432 | 114 | 0.2639 |
| OU | 20% | 432 | 201 | 0.4653 |
| OU | 30% | 432 | 285 | 0.6597 |
| OU | 40% | 432 | 336 | 0.7778 |

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
