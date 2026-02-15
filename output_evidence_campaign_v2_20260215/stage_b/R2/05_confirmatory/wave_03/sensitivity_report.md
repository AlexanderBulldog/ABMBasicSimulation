# Uncertainty Sensitivity Report (2026-02-15)

## 1) Setup
- Dataset: `C:\Users\Alexander\Desktop\Education\Aspa\code\output_evidence_campaign_v2_20260215\stage_b\R2\05_confirmatory\wave_03\lhs_runs_confirm_wave3.csv`
- Implausibility threshold: `2.6`
- SA domain: `nroy`
- Baseline points used: `825`
- Reductions: `10%, 20%, 30%, 40%`

## 2) Baseline Uncertainty Decomposition
| Metric | OU var | EV var | MD var | CU var (median) | Total var (median) |
|---|---:|---:|---:|---:|---:|
| AvgPrice_mean | 0.0025 | 0.000322798 | 0.00315091 | 0.000165423 | 0.00613913 |
| BankBailedOut_share | 0.0004 | 4.82094e-05 | 0.000478297 | 2.7593e-05 | 0.000954099 |
| BankBailoutAmount_mean | 0.04 | 0.0174171 | 0.0286043 | 0.00971196 | 0.0957333 |
| BankResolutionHaircut_mean | 0.0001 | 7.74886e-05 | 0.000212576 | 3.73355e-05 | 0.0004274 |
| BankResolved_share | 0.0009 | 0.000915978 | 0.0013286 | 0.000557364 | 0.00370194 |
| Consumption_mean | 100 | 32.5977 | 92.5983 | 20.4949 | 245.691 |
| DefaultsHH_rate | 4e-06 | 1.39449e-06 | 4.78297e-06 | 9.00564e-07 | 1.1078e-05 |
| HH_Deposit_mean | 1 | 0.173471 | 0.715107 | 0.0186608 | 1.90724 |
| Output_mean | 100 | 27.7808 | 92.5983 | 18.1616 | 238.541 |
| Transfers_mean | 4 | 0.198716 | 2.86043 | 0.132123 | 7.19127 |

## 3) SA Results (newly implausible share)
| Component | Reduction | Points | Newly implausible | Share |
|---|---:|---:|---:|---:|
| CU | 10% | 825 | 9 | 0.0109 |
| CU | 20% | 825 | 27 | 0.0327 |
| CU | 30% | 825 | 30 | 0.0364 |
| CU | 40% | 825 | 42 | 0.0509 |
| EV | 10% | 825 | 24 | 0.0291 |
| EV | 20% | 825 | 33 | 0.0400 |
| EV | 30% | 825 | 60 | 0.0727 |
| EV | 40% | 825 | 78 | 0.0945 |
| MD | 10% | 825 | 54 | 0.0655 |
| MD | 20% | 825 | 108 | 0.1309 |
| MD | 30% | 825 | 183 | 0.2218 |
| MD | 40% | 825 | 234 | 0.2836 |
| OU | 10% | 825 | 63 | 0.0764 |
| OU | 20% | 825 | 111 | 0.1345 |
| OU | 30% | 825 | 189 | 0.2291 |
| OU | 40% | 825 | 252 | 0.3055 |

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
