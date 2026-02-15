# Uncertainty Sensitivity Report (2026-02-15)

## 1) Setup
- Dataset: `C:\Users\Alexander\Desktop\Education\Aspa\code\output_evidence_campaign_v3_20260215\stage_a\A1\01_wave1\lhs_runs.csv`
- Implausibility threshold: `2.6`
- SA domain: `nroy`
- Baseline points used: `150`
- Reductions: `10%, 20%, 30%, 40%`

## 2) Baseline Uncertainty Decomposition
| Metric | OU var | EV var | MD var | CU var (median) | Total var (median) |
|---|---:|---:|---:|---:|---:|
| AvgPrice_mean | 0.0025 | 0.000711256 | 0.005929 | 0.000409072 | 0.00954933 |
| BankBailedOut_share | 0.0004 | 0.000162037 | 0.0009 | 7.21325e-05 | 0.00153417 |
| BankBailoutAmount_mean | 0.04 | 0.0616685 | 0.053824 | 0.0259175 | 0.18141 |
| BankResolutionHaircut_mean | 0.0001 | 0.000295956 | 0.0004 | 0.000111244 | 0.0009072 |
| BankResolved_share | 0.0009 | 0.0014213 | 0.0025 | 0.000651149 | 0.00547245 |
| Consumption_mean | 100 | 99.6567 | 174.24 | 106.177 | 480.074 |
| DefaultsHH_rate | 4e-06 | 3.38519e-06 | 9e-06 | 1.35888e-06 | 1.77441e-05 |
| Employment_mean | 25 | 60.1518 | 43.56 | 76.4825 | 205.194 |
| HH_Deposit_mean | 1 | 1.05973 | 1.3456 | 0.0382343 | 3.44357 |
| Output_mean | 100 | 128.221 | 174.24 | 144.14 | 546.601 |
| PriceDispersion_mean | 0.0036 | 0.000376752 | 0.0144 | 0.000340483 | 0.0187172 |
| Transfers_mean | 4 | 0.885366 | 5.3824 | 0.859987 | 11.1278 |

## 3) SA Results (newly implausible share)
| Component | Reduction | Points | Newly implausible | Share |
|---|---:|---:|---:|---:|
| CU | 10% | 150 | 6 | 0.0400 |
| CU | 20% | 150 | 12 | 0.0800 |
| CU | 30% | 150 | 18 | 0.1200 |
| CU | 40% | 150 | 36 | 0.2400 |
| EV | 10% | 150 | 6 | 0.0400 |
| EV | 20% | 150 | 18 | 0.1200 |
| EV | 30% | 150 | 18 | 0.1200 |
| EV | 40% | 150 | 33 | 0.2200 |
| MD | 10% | 150 | 18 | 0.1200 |
| MD | 20% | 150 | 21 | 0.1400 |
| MD | 30% | 150 | 24 | 0.1600 |
| MD | 40% | 150 | 30 | 0.2000 |
| OU | 10% | 150 | 12 | 0.0800 |
| OU | 20% | 150 | 18 | 0.1200 |
| OU | 30% | 150 | 18 | 0.1200 |
| OU | 40% | 150 | 21 | 0.1400 |

## 4) Interpretation
- Dominant component (rank at max reduction): `CU`
- Investment guidance:
  - `CU` dominant: increase design density / improve GP specification.
  - `EV` dominant: increase replicate seeds `K` or refine stochastic structure.
  - `OU` dominant: improve measurement quality / uncertainty model of observations.
  - `MD` dominant: revisit model structure or discrepancy assumptions.

## 5) Limitations
- EV is treated as metric-level scalar (not x-dependent).
- SA is local to selected baseline domain.
- MD remains expert-specified and is not inferred from data.
