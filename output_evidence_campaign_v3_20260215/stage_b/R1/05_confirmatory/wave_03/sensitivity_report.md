# Uncertainty Sensitivity Report (2026-02-15)

## 1) Setup
- Dataset: `C:\Users\Alexander\Desktop\Education\Aspa\code\output_evidence_campaign_v3_20260215\stage_b\R1\05_confirmatory\wave_03\lhs_runs_confirm_wave3.csv`
- Implausibility threshold: `2.55`
- SA domain: `nroy`
- Baseline points used: `798`
- Reductions: `10%, 20%, 30%, 40%`

## 2) Baseline Uncertainty Decomposition
| Metric | OU var | EV var | MD var | CU var (median) | Total var (median) |
|---|---:|---:|---:|---:|---:|
| AvgPrice_mean | 0.0025 | 0.000376457 | 0.00315091 | 0.000180129 | 0.0062075 |
| BankBailedOut_share | 0.0004 | 4.82094e-05 | 0.000478297 | 2.08498e-05 | 0.000947356 |
| BankBailoutAmount_mean | 0.04 | 0.0145408 | 0.0286043 | 0.00767244 | 0.0908175 |
| BankResolutionHaircut_mean | 0.0001 | 7.10355e-05 | 0.000212576 | 3.1571e-05 | 0.000415183 |
| BankResolved_share | 0.0009 | 0.000860193 | 0.0013286 | 0.000567932 | 0.00365673 |
| Consumption_mean | 100 | 30.7837 | 92.5983 | 17.8725 | 241.254 |
| DefaultsHH_rate | 4e-06 | 1.21818e-06 | 4.78297e-06 | 7.36557e-07 | 1.07377e-05 |
| HH_Deposit_mean | 1 | 0.172773 | 0.715107 | 0.0216721 | 1.90955 |
| Output_mean | 100 | 27.5863 | 92.5983 | 15.8204 | 236.005 |
| PriceDispersion_mean | 0.0036 | 0.00033391 | 0.00765275 | 0.000199661 | 0.0117863 |
| Transfers_mean | 4 | 0.201891 | 2.86043 | 0.0954504 | 7.15777 |

## 3) SA Results (newly implausible share)
| Component | Reduction | Points | Newly implausible | Share |
|---|---:|---:|---:|---:|
| CU | 10% | 798 | 6 | 0.0075 |
| CU | 20% | 798 | 21 | 0.0263 |
| CU | 30% | 798 | 39 | 0.0489 |
| CU | 40% | 798 | 48 | 0.0602 |
| EV | 10% | 798 | 15 | 0.0188 |
| EV | 20% | 798 | 42 | 0.0526 |
| EV | 30% | 798 | 57 | 0.0714 |
| EV | 40% | 798 | 72 | 0.0902 |
| MD | 10% | 798 | 54 | 0.0677 |
| MD | 20% | 798 | 114 | 0.1429 |
| MD | 30% | 798 | 189 | 0.2368 |
| MD | 40% | 798 | 246 | 0.3083 |
| OU | 10% | 798 | 57 | 0.0714 |
| OU | 20% | 798 | 120 | 0.1504 |
| OU | 30% | 798 | 195 | 0.2444 |
| OU | 40% | 798 | 264 | 0.3308 |

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
