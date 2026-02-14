# Uncertainty Sensitivity Report (2026-02-13)

## 1) Setup
- Dataset: `C:\Users\Alexander\Desktop\Education\Aspa\code\output_research_core_v3\waves\wave_03\lhs_runs_wave3.csv`
- Implausibility threshold: `2.6`
- SA domain: `nroy`
- Baseline points used: `174`
- Reductions: `10%, 20%, 30%, 40%`

## 2) Baseline Uncertainty Decomposition
| Metric | OU var | EV var | MD var | CU var (median) | Total var (median) |
|---|---:|---:|---:|---:|---:|
| AvgPrice_mean | 0.0025 | 0.000337812 | 0.00826953 | 0.000487086 | 0.0115944 |
| BankBailedOut_share | 0.0004 | 2.31481e-05 | 6.68063e-05 | 1.82289e-05 | 0.000508183 |
| BankBailoutAmount_mean | 0.04 | 0.00294239 | 0.00668063 | 0.00538175 | 0.0550048 |
| BankResolutionHaircut_mean | 0.0001 | 4.50303e-05 | 4.50308e-05 | 3.43948e-05 | 0.000224456 |
| BankResolved_share | 0.0009 | 0.000486111 | 0.000417539 | 0.000267959 | 0.00207161 |
| Bank_Equity_mean | 56.25 | 20.0878 | 3.26678 | 16.3675 | 95.9721 |
| Consumption_mean | 100 | 33.2326 | 299.702 | 41.1882 | 474.122 |
| CreditRejections_mean | 1 | 9.98933 | 1666.82 | 5.8552 | 1683.66 |
| DefaultsHH_rate | 4e-06 | 2.63194e-07 | 1.44799e-06 | 1.31678e-06 | 7.02796e-06 |
| Employment_mean | 25 | 17.8758 | 184.091 | 17.8211 | 244.788 |
| HH_Deposit_mean | 1 | 0.294072 | 0.375785 | 0.0240212 | 1.69388 |
| InventoryGap_mean | 0.0625 | 0.0141791 | 0.00122599 | 0.0191573 | 0.0970624 |
| Output_mean | 100 | 39.6497 | 626.499 | 34.8906 | 801.04 |
| PriceDispersion_mean | 0.0036 | 0.000261396 | 0.00031796 | 0.000203081 | 0.00438244 |
| Transfers_mean | 4 | 0.298311 | 3.76502 | 0.281298 | 8.34463 |

## 3) SA Results (newly implausible share)
| Component | Reduction | Points | Newly implausible | Share |
|---|---:|---:|---:|---:|
| CU | 10% | 174 | 0 | 0.0000 |
| CU | 20% | 174 | 0 | 0.0000 |
| CU | 30% | 174 | 3 | 0.0172 |
| CU | 40% | 174 | 6 | 0.0345 |
| EV | 10% | 174 | 0 | 0.0000 |
| EV | 20% | 174 | 3 | 0.0172 |
| EV | 30% | 174 | 3 | 0.0172 |
| EV | 40% | 174 | 3 | 0.0172 |
| MD | 10% | 174 | 15 | 0.0862 |
| MD | 20% | 174 | 36 | 0.2069 |
| MD | 30% | 174 | 54 | 0.3103 |
| MD | 40% | 174 | 57 | 0.3276 |
| OU | 10% | 174 | 3 | 0.0172 |
| OU | 20% | 174 | 9 | 0.0517 |
| OU | 30% | 174 | 21 | 0.1207 |
| OU | 40% | 174 | 27 | 0.1552 |

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
