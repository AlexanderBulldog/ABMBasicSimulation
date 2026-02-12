# Uncertainty Sensitivity Report (2026-02-10)

## 1) Setup
- Dataset: `C:\Users\Alexander\Desktop\Education\Aspa\code\output_research_core_v2_full\05_confirmatory\01_wave1\lhs_runs.csv`
- Implausibility threshold: `3.0`
- SA domain: `nroy`
- Baseline points used: `823`
- Reductions: `10%, 20%, 30%, 40%`

## 2) Baseline Uncertainty Decomposition
| Metric | OU var | EV var | MD var | CU var (median) | Total var (median) |
|---|---:|---:|---:|---:|---:|
| AvgPrice_mean | 0.0025 | 0.00025079 | 0.0523965 | 0.000152681 | 0.0552999 |
| BankBailedOut_share | 0.0004 | 4.11523e-05 | 0.000158573 | 2.26566e-05 | 0.000622382 |
| BankBailoutAmount_mean | 0.04 | 0.00323447 | 0.04 | 0.00173308 | 0.0849676 |
| BankResolutionHaircut_mean | 0.0001 | 5.34449e-05 | 0.000116201 | 3.02378e-05 | 0.000299884 |
| BankResolved_share | 0.0009 | 0.00095679 | 0.0025 | 0.000543664 | 0.00490045 |
| Bank_Equity_mean | 56.25 | 20.0728 | 56.8148 | 2.84308 | 135.981 |
| Consumption_mean | 100 | 335.372 | 1704.77 | 249.243 | 2389.38 |
| DefaultsHH_rate | 4e-06 | 3.38272e-07 | 3.16049e-06 | 2.63184e-07 | 7.76195e-06 |
| Employment_mean | 25 | 189.913 | 957.903 | 140.934 | 1313.75 |
| HH_Deposit_mean | 1 | 0.341401 | 1.83065 | 0.0552746 | 3.22733 |
| Output_mean | 100 | 406.356 | 1208.04 | 302.634 | 2017.03 |
| Transfers_mean | 4 | 2.74494 | 9.91651 | 2.05846 | 18.7199 |

## 3) SA Results (newly implausible share)
| Component | Reduction | Points | Newly implausible | Share |
|---|---:|---:|---:|---:|
| CU | 10% | 823 | 0 | 0.0000 |
| CU | 20% | 823 | 0 | 0.0000 |
| CU | 30% | 823 | 0 | 0.0000 |
| CU | 40% | 823 | 0 | 0.0000 |
| EV | 10% | 823 | 0 | 0.0000 |
| EV | 20% | 823 | 0 | 0.0000 |
| EV | 30% | 823 | 0 | 0.0000 |
| EV | 40% | 823 | 0 | 0.0000 |
| MD | 10% | 823 | 2 | 0.0024 |
| MD | 20% | 823 | 5 | 0.0061 |
| MD | 30% | 823 | 8 | 0.0097 |
| MD | 40% | 823 | 23 | 0.0279 |
| OU | 10% | 823 | 2 | 0.0024 |
| OU | 20% | 823 | 5 | 0.0061 |
| OU | 30% | 823 | 8 | 0.0097 |
| OU | 40% | 823 | 20 | 0.0243 |

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
