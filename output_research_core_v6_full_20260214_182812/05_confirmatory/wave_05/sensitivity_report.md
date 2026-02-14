# Uncertainty Sensitivity Report (2026-02-14)

## 1) Setup
- Dataset: `C:\Users\Alexander\Desktop\Education\Aspa\code\output_research_core_v6_full_20260214_182812\05_confirmatory\wave_05\lhs_runs_confirm_wave5.csv`
- Implausibility threshold: `2.6`
- SA domain: `nroy`
- Baseline points used: `879`
- Reductions: `10%, 20%, 30%, 40%`

## 2) Baseline Uncertainty Decomposition
| Metric | OU var | EV var | MD var | CU var (median) | Total var (median) |
|---|---:|---:|---:|---:|---:|
| AvgPrice_mean | 0.0025 | 0.00035093 | 0.00156166 | 0.000244789 | 0.00465738 |
| BankBailedOut_share | 0.0004 | 4.82094e-05 | 0.000237054 | 2.38537e-05 | 0.000709117 |
| BankBailoutAmount_mean | 0.04 | 0.0139369 | 0.0141769 | 0.0077508 | 0.0758646 |
| BankResolutionHaircut_mean | 0.0001 | 5.34575e-05 | 0.000105357 | 3.12317e-05 | 0.000290047 |
| BankResolved_share | 0.0009 | 0.00107438 | 0.000658484 | 0.000621543 | 0.00325441 |
| Consumption_mean | 100 | 30.4637 | 45.8937 | 18.8893 | 195.247 |
| DefaultsHH_rate | 4e-06 | 1.45523e-06 | 2.37054e-06 | 7.47783e-07 | 8.57356e-06 |
| HH_Deposit_mean | 1 | 0.120747 | 0.354422 | 0.0167148 | 1.49188 |
| Output_mean | 100 | 29.3443 | 45.8937 | 15.7112 | 190.949 |
| Transfers_mean | 4 | 0.192554 | 1.41769 | 0.126631 | 5.73687 |

## 3) SA Results (newly implausible share)
| Component | Reduction | Points | Newly implausible | Share |
|---|---:|---:|---:|---:|
| CU | 10% | 879 | 24 | 0.0273 |
| CU | 20% | 879 | 45 | 0.0512 |
| CU | 30% | 879 | 57 | 0.0648 |
| CU | 40% | 879 | 75 | 0.0853 |
| EV | 10% | 879 | 39 | 0.0444 |
| EV | 20% | 879 | 63 | 0.0717 |
| EV | 30% | 879 | 93 | 0.1058 |
| EV | 40% | 879 | 117 | 0.1331 |
| MD | 10% | 879 | 51 | 0.0580 |
| MD | 20% | 879 | 93 | 0.1058 |
| MD | 30% | 879 | 138 | 0.1570 |
| MD | 40% | 879 | 177 | 0.2014 |
| OU | 10% | 879 | 105 | 0.1195 |
| OU | 20% | 879 | 186 | 0.2116 |
| OU | 30% | 879 | 285 | 0.3242 |
| OU | 40% | 879 | 441 | 0.5017 |

## 4) Interpretation
- Dominant component (rank at max reduction): `OU`
- Investment guidance:
  - `CU` dominant: increase design density / improve GP specification.
  - `EV` dominant: increase replicate seeds `K` or refine stochastic structure.
  - `OU` dominant: improve measurement quality / uncertainty model of observations.
  - `MD` dominant: revisit model structure or discrepancy assumptions.

## 5) Limitations
- EV is treated as metric-level scalar (not x-dependent).
- SA is local to selected baseline domain.
- MD remains expert-specified and is not inferred from data.
