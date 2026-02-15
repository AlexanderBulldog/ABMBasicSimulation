# Uncertainty Sensitivity Report (2026-02-15)

## 1) Setup
- Dataset: `C:\Users\Alexander\Desktop\Education\Aspa\code\output_evidence_campaign_v3_20260215\stage_b\R1\waves\wave_03\lhs_runs_wave3.csv`
- Implausibility threshold: `2.55`
- SA domain: `nroy`
- Baseline points used: `624`
- Reductions: `10%, 20%, 30%, 40%`

## 2) Baseline Uncertainty Decomposition
| Metric | OU var | EV var | MD var | CU var (median) | Total var (median) |
|---|---:|---:|---:|---:|---:|
| AvgPrice_mean | 0.0025 | 0.000452979 | 0.00315091 | 0.000231387 | 0.00633528 |
| BankBailedOut_share | 0.0004 | 6.19835e-05 | 0.000478297 | 4.30654e-05 | 0.000983346 |
| BankBailoutAmount_mean | 0.04 | 0.0310111 | 0.0286043 | 0.0167182 | 0.116334 |
| BankResolutionHaircut_mean | 0.0001 | 0.000115148 | 0.000212576 | 9.87509e-05 | 0.000526476 |
| BankResolved_share | 0.0009 | 0.00126928 | 0.0013286 | 0.00074716 | 0.00424505 |
| Consumption_mean | 100 | 42.8266 | 92.5983 | 41.3436 | 276.769 |
| DefaultsHH_rate | 4e-06 | 1.95647e-06 | 4.78297e-06 | 1.44499e-06 | 1.21844e-05 |
| HH_Deposit_mean | 1 | 0.265337 | 0.715107 | 0.0236208 | 2.00406 |
| Output_mean | 100 | 35.8872 | 92.5983 | 45.709 | 274.194 |
| PriceDispersion_mean | 0.0036 | 0.000386755 | 0.00765275 | 0.000450171 | 0.0120897 |
| Transfers_mean | 4 | 0.25339 | 2.86043 | 0.280616 | 7.39443 |

## 3) SA Results (newly implausible share)
| Component | Reduction | Points | Newly implausible | Share |
|---|---:|---:|---:|---:|
| CU | 10% | 624 | 21 | 0.0337 |
| CU | 20% | 624 | 27 | 0.0433 |
| CU | 30% | 624 | 36 | 0.0577 |
| CU | 40% | 624 | 63 | 0.1010 |
| EV | 10% | 624 | 21 | 0.0337 |
| EV | 20% | 624 | 21 | 0.0337 |
| EV | 30% | 624 | 39 | 0.0625 |
| EV | 40% | 624 | 51 | 0.0817 |
| MD | 10% | 624 | 27 | 0.0433 |
| MD | 20% | 624 | 78 | 0.1250 |
| MD | 30% | 624 | 111 | 0.1779 |
| MD | 40% | 624 | 156 | 0.2500 |
| OU | 10% | 624 | 30 | 0.0481 |
| OU | 20% | 624 | 87 | 0.1394 |
| OU | 30% | 624 | 120 | 0.1923 |
| OU | 40% | 624 | 171 | 0.2740 |

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
