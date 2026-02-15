# Uncertainty Sensitivity Report (2026-02-15)

## 1) Setup
- Dataset: `C:\Users\Alexander\Desktop\Education\Aspa\code\output_evidence_campaign_v2_20260215\stage_a\A4\waves\wave_03\lhs_runs_wave3.csv`
- Implausibility threshold: `2.45`
- SA domain: `nroy`
- Baseline points used: `144`
- Reductions: `10%, 20%, 30%, 40%`

## 2) Baseline Uncertainty Decomposition
| Metric | OU var | EV var | MD var | CU var (median) | Total var (median) |
|---|---:|---:|---:|---:|---:|
| AvgPrice_mean | 0.0025 | 0.000643786 | 0.00717409 | 0.00048202 | 0.0107999 |
| BankBailedOut_share | 0.0004 | 0.000166667 | 0.001089 | 0.000141014 | 0.00179668 |
| BankBailoutAmount_mean | 0.04 | 0.0555147 | 0.065127 | 0.12 | 0.280642 |
| BankResolutionHaircut_mean | 0.0001 | 0.00018441 | 0.000484 | 0.000152955 | 0.000921365 |
| BankResolved_share | 0.0009 | 0.0012037 | 0.003025 | 0.00106459 | 0.00619329 |
| Consumption_mean | 100 | 63.6655 | 210.83 | 59.0272 | 433.523 |
| DefaultsHH_rate | 4e-06 | 2.17315e-06 | 1.089e-05 | 1.80394e-06 | 1.88671e-05 |
| Employment_mean | 25 | 37.9091 | 52.7076 | 29.4521 | 145.069 |
| HH_Deposit_mean | 1 | 0.423945 | 1.62818 | 0.0374106 | 3.08953 |
| Output_mean | 100 | 61.8092 | 210.83 | 63.8288 | 436.468 |
| PriceDispersion_mean | 0.0036 | 0.000395785 | 0.017424 | 0.000435961 | 0.0218557 |
| Transfers_mean | 4 | 0.508076 | 6.5127 | 0.417299 | 11.4381 |

## 3) SA Results (newly implausible share)
| Component | Reduction | Points | Newly implausible | Share |
|---|---:|---:|---:|---:|
| CU | 10% | 144 | 6 | 0.0417 |
| CU | 20% | 144 | 18 | 0.1250 |
| CU | 30% | 144 | 21 | 0.1458 |
| CU | 40% | 144 | 27 | 0.1875 |
| EV | 10% | 144 | 6 | 0.0417 |
| EV | 20% | 144 | 21 | 0.1458 |
| EV | 30% | 144 | 24 | 0.1667 |
| EV | 40% | 144 | 27 | 0.1875 |
| MD | 10% | 144 | 15 | 0.1042 |
| MD | 20% | 144 | 30 | 0.2083 |
| MD | 30% | 144 | 42 | 0.2917 |
| MD | 40% | 144 | 87 | 0.6042 |
| OU | 10% | 144 | 3 | 0.0208 |
| OU | 20% | 144 | 15 | 0.1042 |
| OU | 30% | 144 | 21 | 0.1458 |
| OU | 40% | 144 | 27 | 0.1875 |

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
