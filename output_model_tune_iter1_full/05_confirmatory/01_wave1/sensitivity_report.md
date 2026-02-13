# Uncertainty Sensitivity Report (2026-02-12)

## 1) Setup
- Dataset: `C:\Users\smart\OneDrive\Рабочий стол\Аспирантура\code\ABMBasicSimulation\output_model_tune_iter1_full\05_confirmatory\01_wave1\lhs_runs.csv`
- Implausibility threshold: `3.0`
- SA domain: `nroy`
- Baseline points used: `1125`
- Reductions: `10%, 20%, 30%, 40%`

## 2) Baseline Uncertainty Decomposition
| Metric | OU var | EV var | MD var | CU var (median) | Total var (median) |
|---|---:|---:|---:|---:|---:|
| AvgPrice_mean | 0.0025 | 0.000320884 | 0.0521591 | 0.000216039 | 0.055196 |
| BankBailedOut_share | 0.0004 | 3.08642e-05 | 0.0004 | 2.93096e-05 | 0.000860174 |
| BankBailoutAmount_mean | 0.04 | 0.00482276 | 0.04 | 0.00567259 | 0.0904954 |
| BankResolutionHaircut_mean | 0.0001 | 3.08721e-05 | 0.000248676 | 3.90752e-05 | 0.000418623 |
| BankResolved_share | 0.0009 | 0.000401235 | 0.00231824 | 0.000237002 | 0.00385648 |
| Bank_Equity_mean | 56.25 | 33.2485 | 14.2292 | 17.8475 | 121.575 |
| Consumption_mean | 100 | 44.8394 | 1838.83 | 45.5482 | 2029.22 |
| CreditRejections_mean | 1 | 14.5126 | 9988.89 | 7.20602 | 10011.6 |
| DefaultsHH_rate | 4e-06 | 4.20782e-07 | 4.07442e-06 | 5.23601e-07 | 9.0188e-06 |
| HH_Deposit_mean | 1 | 0.793192 | 2.25 | 0.0306069 | 4.0738 |
| Output_mean | 100 | 45.5419 | 3885.11 | 56.9149 | 4087.57 |
| PriceDispersion_mean | 0.0036 | 0.000224992 | 0.00179572 | 0.000215145 | 0.00583586 |
| Transfers_mean | 4 | 0.347459 | 22.5499 | 0.359038 | 27.2564 |

## 3) SA Results (newly implausible share)
| Component | Reduction | Points | Newly implausible | Share |
|---|---:|---:|---:|---:|
| CU | 10% | 1125 | 0 | 0.0000 |
| CU | 20% | 1125 | 0 | 0.0000 |
| CU | 30% | 1125 | 0 | 0.0000 |
| CU | 40% | 1125 | 0 | 0.0000 |
| EV | 10% | 1125 | 0 | 0.0000 |
| EV | 20% | 1125 | 0 | 0.0000 |
| EV | 30% | 1125 | 0 | 0.0000 |
| EV | 40% | 1125 | 0 | 0.0000 |
| MD | 10% | 1125 | 0 | 0.0000 |
| MD | 20% | 1125 | 0 | 0.0000 |
| MD | 30% | 1125 | 3 | 0.0027 |
| MD | 40% | 1125 | 3 | 0.0027 |
| OU | 10% | 1125 | 0 | 0.0000 |
| OU | 20% | 1125 | 0 | 0.0000 |
| OU | 30% | 1125 | 3 | 0.0027 |
| OU | 40% | 1125 | 3 | 0.0027 |

## 4) Interpretation
- Dominant component (rank at max reduction): `MD, OU`
- Investment guidance:
  - `CU` dominant: increase design density / improve GP specification.
  - `EV` dominant: increase replicate seeds `K` or refine stochastic structure.
  - `OU` dominant: improve measurement quality / uncertainty model of observations.
  - `MD` dominant: revisit model structure or discrepancy assumptions.

## 5) Limitations
- EV is treated as metric-level scalar (not x-dependent).
- SA is local to selected baseline domain.
- MD remains expert-specified and is not inferred from data.
