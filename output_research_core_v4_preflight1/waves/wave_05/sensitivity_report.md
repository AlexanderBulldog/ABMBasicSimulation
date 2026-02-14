# Uncertainty Sensitivity Report (2026-02-14)

## 1) Setup
- Dataset: `C:\Users\Alexander\Desktop\Education\Aspa\code\output_research_core_v4_preflight1\waves\wave_05\lhs_runs_wave5.csv`
- Implausibility threshold: `2.6`
- SA domain: `nroy`
- Baseline points used: `195`
- Reductions: `10%, 20%, 30%, 40%`

## 2) Baseline Uncertainty Decomposition
| Metric | OU var | EV var | MD var | CU var (median) | Total var (median) |
|---|---:|---:|---:|---:|---:|
| AvgPrice_mean | 0.0025 | 0.000409065 | 0.00350392 | 0.000293721 | 0.00670671 |
| BankBailedOut_share | 0.0004 | 0.000103704 | 0.000531883 | 4.74738e-05 | 0.00108306 |
| BankBailoutAmount_mean | 0.04 | 0.0154778 | 0.0318089 | 0.00806823 | 0.0953549 |
| BankResolutionHaircut_mean | 0.0001 | 0.000101236 | 0.000236392 | 5.88615e-05 | 0.00049649 |
| BankResolved_share | 0.0009 | 0.00073037 | 0.00147745 | 0.00038671 | 0.00349453 |
| Consumption_mean | 100 | 65.3548 | 102.972 | 32.3894 | 300.717 |
| CreditRejections_mean | 100 | 9.42958 | 236.392 | 5.88251 | 351.704 |
| DefaultsHH_rate | 4e-06 | 3.08593e-07 | 5.31883e-06 | 2.81334e-07 | 9.90875e-06 |
| Employment_mean | 25 | 41.0828 | 25.7431 | 18.4133 | 110.239 |
| HH_Deposit_mean | 1 | 0.179158 | 0.795224 | 0.0189194 | 1.9933 |
| InventoryGap_mean | 0.0625 | 0.0261166 | 0.0723951 | 0.0139507 | 0.174962 |
| Output_mean | 100 | 52.9091 | 102.972 | 28.0184 | 283.9 |
| Transfers_mean | 4 | 0.569803 | 3.18089 | 0.267392 | 8.01809 |

## 3) SA Results (newly implausible share)
| Component | Reduction | Points | Newly implausible | Share |
|---|---:|---:|---:|---:|
| CU | 10% | 195 | 6 | 0.0308 |
| CU | 20% | 195 | 15 | 0.0769 |
| CU | 30% | 195 | 24 | 0.1231 |
| CU | 40% | 195 | 30 | 0.1538 |
| EV | 10% | 195 | 15 | 0.0769 |
| EV | 20% | 195 | 36 | 0.1846 |
| EV | 30% | 195 | 54 | 0.2769 |
| EV | 40% | 195 | 72 | 0.3692 |
| MD | 10% | 195 | 9 | 0.0462 |
| MD | 20% | 195 | 24 | 0.1231 |
| MD | 30% | 195 | 39 | 0.2000 |
| MD | 40% | 195 | 54 | 0.2769 |
| OU | 10% | 195 | 9 | 0.0462 |
| OU | 20% | 195 | 24 | 0.1231 |
| OU | 30% | 195 | 36 | 0.1846 |
| OU | 40% | 195 | 54 | 0.2769 |

## 4) Interpretation
- Dominant component (rank at max reduction): `EV`
- Investment guidance:
  - `CU` dominant: increase design density / improve GP specification.
  - `EV` dominant: increase replicate seeds `K` or refine stochastic structure.
  - `OU` dominant: improve measurement quality / uncertainty model of observations.
  - `MD` dominant: revisit model structure or discrepancy assumptions.

## 5) Limitations
- EV is treated as metric-level scalar (not x-dependent).
- SA is local to selected baseline domain.
- MD remains expert-specified and is not inferred from data.
