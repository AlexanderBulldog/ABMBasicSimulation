# Uncertainty Sensitivity Report (2026-02-15)

## 1) Setup
- Dataset: `C:\Users\Alexander\Desktop\Education\Aspa\code\output_evidence_campaign_v2_20260215\stage_a\A2\waves\wave_03\lhs_runs_wave3.csv`
- Implausibility threshold: `2.45`
- SA domain: `nroy`
- Baseline points used: `141`
- Reductions: `10%, 20%, 30%, 40%`

## 2) Baseline Uncertainty Decomposition
| Metric | OU var | EV var | MD var | CU var (median) | Total var (median) |
|---|---:|---:|---:|---:|---:|
| AvgPrice_mean | 0.0025 | 0.000643786 | 0.00691559 | 0.00048202 | 0.0105414 |
| BankBailedOut_share | 0.0004 | 0.000166667 | 0.00104976 | 0.000141014 | 0.00175744 |
| BankBailoutAmount_mean | 0.04 | 0.0555147 | 0.0627803 | 0.12 | 0.278295 |
| BankResolutionHaircut_mean | 0.0001 | 0.00018441 | 0.00046656 | 0.000152955 | 0.000903925 |
| BankResolved_share | 0.0009 | 0.0012037 | 0.002916 | 0.00106459 | 0.00608429 |
| Consumption_mean | 100 | 63.6655 | 203.234 | 59.0272 | 425.926 |
| DefaultsHH_rate | 4e-06 | 2.17315e-06 | 1.04976e-05 | 1.80394e-06 | 1.84747e-05 |
| Employment_mean | 25 | 37.9091 | 50.8084 | 29.4521 | 143.17 |
| HH_Deposit_mean | 1 | 0.423945 | 1.56951 | 0.0366651 | 3.03012 |
| Output_mean | 100 | 61.8092 | 203.234 | 63.8288 | 428.872 |
| PriceDispersion_mean | 0.0036 | 0.000395785 | 0.0167962 | 0.000435961 | 0.0212279 |
| Transfers_mean | 4 | 0.508076 | 6.27803 | 0.417299 | 11.2034 |

## 3) SA Results (newly implausible share)
| Component | Reduction | Points | Newly implausible | Share |
|---|---:|---:|---:|---:|
| CU | 10% | 141 | 15 | 0.1064 |
| CU | 20% | 141 | 18 | 0.1277 |
| CU | 30% | 141 | 18 | 0.1277 |
| CU | 40% | 141 | 48 | 0.3404 |
| EV | 10% | 141 | 12 | 0.0851 |
| EV | 20% | 141 | 18 | 0.1277 |
| EV | 30% | 141 | 24 | 0.1702 |
| EV | 40% | 141 | 30 | 0.2128 |
| MD | 10% | 141 | 15 | 0.1064 |
| MD | 20% | 141 | 30 | 0.2128 |
| MD | 30% | 141 | 48 | 0.3404 |
| MD | 40% | 141 | 87 | 0.6170 |
| OU | 10% | 141 | 9 | 0.0638 |
| OU | 20% | 141 | 18 | 0.1277 |
| OU | 30% | 141 | 24 | 0.1702 |
| OU | 40% | 141 | 27 | 0.1915 |

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
