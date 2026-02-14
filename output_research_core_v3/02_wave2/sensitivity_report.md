# Uncertainty Sensitivity Report (2026-02-13)

## 1) Setup
- Dataset: `C:\Users\Alexander\Desktop\Education\Aspa\code\output_research_core_v3\waves\wave_05\lhs_runs_wave5.csv`
- Implausibility threshold: `2.6`
- SA domain: `nroy`
- Baseline points used: `432`
- Reductions: `10%, 20%, 30%, 40%`

## 2) Baseline Uncertainty Decomposition
| Metric | OU var | EV var | MD var | CU var (median) | Total var (median) |
|---|---:|---:|---:|---:|---:|
| AvgPrice_mean | 0.0025 | 0.000298373 | 0.00736354 | 0.000224323 | 0.0103862 |
| BankBailedOut_share | 0.0004 | 2.31481e-05 | 5.94871e-05 | 1.55679e-05 | 0.000498203 |
| BankBailoutAmount_mean | 0.04 | 0.0103716 | 0.00594871 | 0.00325835 | 0.0595786 |
| BankResolutionHaircut_mean | 0.0001 | 3.85776e-05 | 4.00973e-05 | 1.84881e-05 | 0.000197163 |
| BankResolved_share | 0.0009 | 0.000444444 | 0.000371795 | 0.000270434 | 0.00198667 |
| Consumption_mean | 100 | 37.1877 | 266.867 | 19.3307 | 423.385 |
| CreditRejections_mean | 1 | 7.60408 | 1484.21 | 5.12997 | 1497.94 |
| DefaultsHH_rate | 4e-06 | 2.10648e-07 | 1.28935e-06 | 1.51037e-07 | 5.65103e-06 |
| Employment_mean | 25 | 23.5933 | 163.923 | 11.4231 | 223.939 |
| HH_Deposit_mean | 1 | 0.157867 | 0.334615 | 0.0206586 | 1.51314 |
| InventoryGap_mean | 0.0625 | 0.0118451 | 0.00109167 | 0.00695303 | 0.0823898 |
| Output_mean | 100 | 33.3945 | 557.861 | 18.307 | 709.563 |
| PriceDispersion_mean | 0.0036 | 0.000325932 | 0.000283125 | 0.000173398 | 0.00438245 |
| Transfers_mean | 4 | 0.317954 | 3.35253 | 0.155363 | 7.82584 |

## 3) SA Results (newly implausible share)
| Component | Reduction | Points | Newly implausible | Share |
|---|---:|---:|---:|---:|
| CU | 10% | 432 | 3 | 0.0069 |
| CU | 20% | 432 | 3 | 0.0069 |
| CU | 30% | 432 | 6 | 0.0139 |
| CU | 40% | 432 | 9 | 0.0208 |
| EV | 10% | 432 | 3 | 0.0069 |
| EV | 20% | 432 | 9 | 0.0208 |
| EV | 30% | 432 | 15 | 0.0347 |
| EV | 40% | 432 | 18 | 0.0417 |
| MD | 10% | 432 | 21 | 0.0486 |
| MD | 20% | 432 | 42 | 0.0972 |
| MD | 30% | 432 | 81 | 0.1875 |
| MD | 40% | 432 | 174 | 0.4028 |
| OU | 10% | 432 | 15 | 0.0347 |
| OU | 20% | 432 | 21 | 0.0486 |
| OU | 30% | 432 | 51 | 0.1181 |
| OU | 40% | 432 | 84 | 0.1944 |

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
