# Uncertainty Sensitivity Report (2026-02-13)

## 1) Setup
- Dataset: `C:\Users\Alexander\Desktop\Education\Aspa\code\output_research_core_v3_tuned2\waves\wave_04\lhs_runs_wave4.csv`
- Implausibility threshold: `2.75`
- SA domain: `nroy`
- Baseline points used: `342`
- Reductions: `10%, 20%, 30%, 40%`

## 2) Baseline Uncertainty Decomposition
| Metric | OU var | EV var | MD var | CU var (median) | Total var (median) |
|---|---:|---:|---:|---:|---:|
| AvgPrice_mean | 0.0025 | 0.00035703 | 0.0230992 | 0.000224251 | 0.0261805 |
| BankBailedOut_share | 0.0004 | 9.25926e-05 | 0.00350638 | 4.53217e-05 | 0.00404429 |
| BankBailoutAmount_mean | 0.04 | 0.0331053 | 0.209697 | 0.0149658 | 0.297768 |
| BankResolutionHaircut_mean | 0.0001 | 0.000211774 | 0.00155839 | 9.78629e-05 | 0.00196803 |
| BankResolved_share | 0.0009 | 0.00170833 | 0.00973994 | 0.000967304 | 0.0133156 |
| Consumption_mean | 100 | 45.8353 | 678.835 | 28.1961 | 852.866 |
| CreditRejections_mean | 25 | 20.2807 | 400 | 12.5126 | 457.793 |
| DefaultsHH_rate | 4e-06 | 5.71759e-07 | 3.50638e-05 | 3.88724e-07 | 4.00243e-05 |
| Employment_mean | 25 | 25.1881 | 169.709 | 14.5012 | 234.398 |
| HH_Deposit_mean | 1 | 1.79464 | 5.24243 | 2.44714 | 10.4842 |
| InventoryGap_mean | 0.0625 | 0.0147198 | 0.1225 | 0.00917393 | 0.208894 |
| Output_mean | 100 | 49.9141 | 678.835 | 30.9523 | 859.701 |
| PriceDispersion_mean | 0.0036 | 0.000425068 | 0.0144 | 0.000258233 | 0.0186833 |
| Transfers_mean | 4 | 0.364068 | 20.9697 | 0.227128 | 25.5609 |

## 3) SA Results (newly implausible share)
| Component | Reduction | Points | Newly implausible | Share |
|---|---:|---:|---:|---:|
| CU | 10% | 342 | 0 | 0.0000 |
| CU | 20% | 342 | 6 | 0.0175 |
| CU | 30% | 342 | 6 | 0.0175 |
| CU | 40% | 342 | 15 | 0.0439 |
| EV | 10% | 342 | 0 | 0.0000 |
| EV | 20% | 342 | 0 | 0.0000 |
| EV | 30% | 342 | 0 | 0.0000 |
| EV | 40% | 342 | 0 | 0.0000 |
| MD | 10% | 342 | 9 | 0.0263 |
| MD | 20% | 342 | 30 | 0.0877 |
| MD | 30% | 342 | 45 | 0.1316 |
| MD | 40% | 342 | 69 | 0.2018 |
| OU | 10% | 342 | 0 | 0.0000 |
| OU | 20% | 342 | 0 | 0.0000 |
| OU | 30% | 342 | 0 | 0.0000 |
| OU | 40% | 342 | 0 | 0.0000 |

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
