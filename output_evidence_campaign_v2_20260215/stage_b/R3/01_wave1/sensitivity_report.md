# Uncertainty Sensitivity Report (2026-02-15)

## 1) Setup
- Dataset: `C:\Users\Alexander\Desktop\Education\Aspa\code\output_evidence_campaign_v2_20260215\stage_b\R3\01_wave1\lhs_runs.csv`
- Implausibility threshold: `2.6`
- SA domain: `nroy`
- Baseline points used: `250`
- Reductions: `10%, 20%, 30%, 40%`

## 2) Baseline Uncertainty Decomposition
| Metric | OU var | EV var | MD var | CU var (median) | Total var (median) |
|---|---:|---:|---:|---:|---:|
| AvgPrice_mean | 0.0025 | 0.000492339 | 0.005929 | 0.000356218 | 0.00927756 |
| BankBailedOut_share | 0.0004 | 4.82094e-05 | 0.0009 | 2.75667e-05 | 0.00137578 |
| BankBailoutAmount_mean | 0.04 | 0.015998 | 0.053824 | 0.0100382 | 0.11986 |
| BankResolutionHaircut_mean | 0.0001 | 8.1883e-05 | 0.0004 | 8.00919e-05 | 0.000661975 |
| BankResolved_share | 0.0009 | 0.000750689 | 0.0025 | 0.000704574 | 0.00485526 |
| Consumption_mean | 100 | 54.0732 | 174.24 | 101.283 | 429.596 |
| DefaultsHH_rate | 4e-06 | 1.51391e-06 | 9e-06 | 1.1779e-06 | 1.56918e-05 |
| HH_Deposit_mean | 1 | 0.718612 | 1.3456 | 0.0374416 | 3.10165 |
| Output_mean | 100 | 66.4067 | 174.24 | 100.961 | 441.608 |
| PriceDispersion_mean | 0.0036 | 0.000334801 | 0.0144 | 0.000467475 | 0.0188023 |
| Transfers_mean | 4 | 0.381118 | 5.3824 | 0.817527 | 10.581 |

## 3) SA Results (newly implausible share)
| Component | Reduction | Points | Newly implausible | Share |
|---|---:|---:|---:|---:|
| CU | 10% | 250 | 6 | 0.0240 |
| CU | 20% | 250 | 15 | 0.0600 |
| CU | 30% | 250 | 24 | 0.0960 |
| CU | 40% | 250 | 27 | 0.1080 |
| EV | 10% | 250 | 3 | 0.0120 |
| EV | 20% | 250 | 6 | 0.0240 |
| EV | 30% | 250 | 12 | 0.0480 |
| EV | 40% | 250 | 18 | 0.0720 |
| MD | 10% | 250 | 9 | 0.0360 |
| MD | 20% | 250 | 30 | 0.1200 |
| MD | 30% | 250 | 42 | 0.1680 |
| MD | 40% | 250 | 54 | 0.2160 |
| OU | 10% | 250 | 9 | 0.0360 |
| OU | 20% | 250 | 12 | 0.0480 |
| OU | 30% | 250 | 24 | 0.0960 |
| OU | 40% | 250 | 30 | 0.1200 |

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
