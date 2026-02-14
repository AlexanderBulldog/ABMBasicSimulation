# Uncertainty Sensitivity Report (2026-02-14)

## 1) Setup
- Dataset: `C:\Users\Alexander\Desktop\Education\Aspa\code\output_research_core_v6_full_20260214_182812\waves\wave_05\lhs_runs_wave5.csv`
- Implausibility threshold: `2.6`
- SA domain: `nroy`
- Baseline points used: `513`
- Reductions: `10%, 20%, 30%, 40%`

## 2) Baseline Uncertainty Decomposition
| Metric | OU var | EV var | MD var | CU var (median) | Total var (median) |
|---|---:|---:|---:|---:|---:|
| AvgPrice_mean | 0.0025 | 0.000326782 | 0.00156166 | 0.000232576 | 0.00462102 |
| BankBailedOut_share | 0.0004 | 4.82094e-05 | 0.000237054 | 2.11979e-05 | 0.000706461 |
| BankBailoutAmount_mean | 0.04 | 0.0169996 | 0.0141769 | 0.00886789 | 0.0800443 |
| BankResolutionHaircut_mean | 0.0001 | 5.80816e-05 | 0.000105357 | 3.61216e-05 | 0.000299561 |
| BankResolved_share | 0.0009 | 0.000891185 | 0.000658484 | 0.000607259 | 0.00305693 |
| Consumption_mean | 100 | 26.0343 | 45.8937 | 23.755 | 195.683 |
| DefaultsHH_rate | 4e-06 | 1.16701e-06 | 2.37054e-06 | 7.74967e-07 | 8.31252e-06 |
| HH_Deposit_mean | 1 | 0.137901 | 0.354422 | 0.0173404 | 1.50966 |
| Output_mean | 100 | 26.7239 | 45.8937 | 21.3348 | 193.952 |
| Transfers_mean | 4 | 0.184465 | 1.41769 | 0.180132 | 5.78229 |

## 3) SA Results (newly implausible share)
| Component | Reduction | Points | Newly implausible | Share |
|---|---:|---:|---:|---:|
| CU | 10% | 513 | 21 | 0.0409 |
| CU | 20% | 513 | 30 | 0.0585 |
| CU | 30% | 513 | 36 | 0.0702 |
| CU | 40% | 513 | 51 | 0.0994 |
| EV | 10% | 513 | 24 | 0.0468 |
| EV | 20% | 513 | 30 | 0.0585 |
| EV | 30% | 513 | 39 | 0.0760 |
| EV | 40% | 513 | 54 | 0.1053 |
| MD | 10% | 513 | 27 | 0.0526 |
| MD | 20% | 513 | 51 | 0.0994 |
| MD | 30% | 513 | 78 | 0.1520 |
| MD | 40% | 513 | 105 | 0.2047 |
| OU | 10% | 513 | 51 | 0.0994 |
| OU | 20% | 513 | 111 | 0.2164 |
| OU | 30% | 513 | 177 | 0.3450 |
| OU | 40% | 513 | 252 | 0.4912 |

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
