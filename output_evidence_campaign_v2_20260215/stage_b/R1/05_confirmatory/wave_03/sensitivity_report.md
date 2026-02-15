# Uncertainty Sensitivity Report (2026-02-15)

## 1) Setup
- Dataset: `C:\Users\Alexander\Desktop\Education\Aspa\code\output_evidence_campaign_v2_20260215\stage_b\R1\05_confirmatory\wave_03\lhs_runs_confirm_wave3.csv`
- Implausibility threshold: `2.6`
- SA domain: `nroy`
- Baseline points used: `1023`
- Reductions: `10%, 20%, 30%, 40%`

## 2) Baseline Uncertainty Decomposition
| Metric | OU var | EV var | MD var | CU var (median) | Total var (median) |
|---|---:|---:|---:|---:|---:|
| AvgPrice_mean | 0.0025 | 0.000369346 | 0.00389002 | 0.000234289 | 0.00699365 |
| BankBailedOut_share | 0.0004 | 4.82094e-05 | 0.00059049 | 2.68184e-05 | 0.00106552 |
| BankBailoutAmount_mean | 0.04 | 0.016112 | 0.0353139 | 0.0147989 | 0.106225 |
| BankResolutionHaircut_mean | 0.0001 | 6.24854e-05 | 0.00026244 | 4.04312e-05 | 0.000465357 |
| BankResolved_share | 0.0009 | 0.000856061 | 0.00164025 | 0.000475568 | 0.00387188 |
| Consumption_mean | 100 | 35.4755 | 114.319 | 26.793 | 276.587 |
| DefaultsHH_rate | 4e-06 | 1.13912e-06 | 5.9049e-06 | 6.18826e-07 | 1.16628e-05 |
| HH_Deposit_mean | 1 | 0.157922 | 0.882848 | 0.0186183 | 2.05939 |
| Output_mean | 100 | 35.4774 | 114.319 | 34.2645 | 284.061 |
| PriceDispersion_mean | 0.0036 | 0.000307704 | 0.00944784 | 0.000307336 | 0.0136629 |
| Transfers_mean | 4 | 0.247813 | 3.53139 | 0.211769 | 7.99098 |

## 3) SA Results (newly implausible share)
| Component | Reduction | Points | Newly implausible | Share |
|---|---:|---:|---:|---:|
| CU | 10% | 1023 | 3 | 0.0029 |
| CU | 20% | 1023 | 15 | 0.0147 |
| CU | 30% | 1023 | 27 | 0.0264 |
| CU | 40% | 1023 | 30 | 0.0293 |
| EV | 10% | 1023 | 6 | 0.0059 |
| EV | 20% | 1023 | 18 | 0.0176 |
| EV | 30% | 1023 | 27 | 0.0264 |
| EV | 40% | 1023 | 36 | 0.0352 |
| MD | 10% | 1023 | 30 | 0.0293 |
| MD | 20% | 1023 | 75 | 0.0733 |
| MD | 30% | 1023 | 147 | 0.1437 |
| MD | 40% | 1023 | 222 | 0.2170 |
| OU | 10% | 1023 | 27 | 0.0264 |
| OU | 20% | 1023 | 60 | 0.0587 |
| OU | 30% | 1023 | 96 | 0.0938 |
| OU | 40% | 1023 | 186 | 0.1818 |

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
