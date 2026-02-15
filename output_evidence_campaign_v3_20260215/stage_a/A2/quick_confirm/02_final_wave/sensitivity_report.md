# Uncertainty Sensitivity Report (2026-02-15)

## 1) Setup
- Dataset: `C:\Users\Alexander\Desktop\Education\Aspa\code\output_evidence_campaign_v3_20260215\stage_a\A2\quick_confirm\waves\wave_03\lhs_runs_wave3.csv`
- Implausibility threshold: `2.55`
- SA domain: `nroy`
- Baseline points used: `231`
- Reductions: `10%, 20%, 30%, 40%`

## 2) Baseline Uncertainty Decomposition
| Metric | OU var | EV var | MD var | CU var (median) | Total var (median) |
|---|---:|---:|---:|---:|---:|
| AvgPrice_mean | 0.0025 | 0.000379495 | 0.00691559 | 0.000266393 | 0.0100615 |
| BankBailedOut_share | 0.0004 | 0.000166667 | 0.00104976 | 0.000121237 | 0.00173766 |
| BankBailoutAmount_mean | 0.04 | 0.0994461 | 0.0627803 | 0.0644157 | 0.266642 |
| BankResolutionHaircut_mean | 0.0001 | 0.000262179 | 0.00046656 | 0.00015556 | 0.000984299 |
| BankResolved_share | 0.0009 | 0.00210648 | 0.002916 | 0.00105461 | 0.00697709 |
| Consumption_mean | 100 | 39.0347 | 203.234 | 22.3717 | 364.64 |
| DefaultsHH_rate | 4e-06 | 2.1669e-06 | 1.04976e-05 | 1.54602e-06 | 1.82105e-05 |
| HH_Deposit_mean | 1 | 0.0939052 | 1.56951 | 0.00941988 | 2.67283 |
| Output_mean | 100 | 46.2346 | 203.234 | 24.3094 | 373.778 |
| PriceDispersion_mean | 0.0036 | 0.000418302 | 0.0167962 | 0.00022759 | 0.0210421 |
| Transfers_mean | 4 | 0.330283 | 6.27803 | 0.210847 | 10.8192 |

## 3) SA Results (newly implausible share)
| Component | Reduction | Points | Newly implausible | Share |
|---|---:|---:|---:|---:|
| CU | 10% | 231 | 0 | 0.0000 |
| CU | 20% | 231 | 0 | 0.0000 |
| CU | 30% | 231 | 0 | 0.0000 |
| CU | 40% | 231 | 3 | 0.0130 |
| EV | 10% | 231 | 0 | 0.0000 |
| EV | 20% | 231 | 6 | 0.0260 |
| EV | 30% | 231 | 6 | 0.0260 |
| EV | 40% | 231 | 12 | 0.0519 |
| MD | 10% | 231 | 21 | 0.0909 |
| MD | 20% | 231 | 33 | 0.1429 |
| MD | 30% | 231 | 63 | 0.2727 |
| MD | 40% | 231 | 84 | 0.3636 |
| OU | 10% | 231 | 6 | 0.0260 |
| OU | 20% | 231 | 21 | 0.0909 |
| OU | 30% | 231 | 30 | 0.1299 |
| OU | 40% | 231 | 33 | 0.1429 |

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
