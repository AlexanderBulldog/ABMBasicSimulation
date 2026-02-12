# Uncertainty Sensitivity Report (2026-02-10)

## 1) Setup
- Dataset: `C:\Users\Alexander\Desktop\Education\Aspa\code\output_research_core_v2_dry\05_confirmatory\02_wave2\lhs_runs_wave2.csv`
- Implausibility threshold: `3.0`
- SA domain: `nroy`
- Baseline points used: `42`
- Reductions: `10%, 20%, 30%, 40%`

## 2) Baseline Uncertainty Decomposition
| Metric | OU var | EV var | MD var | CU var (median) | Total var (median) |
|---|---:|---:|---:|---:|---:|
| AvgPrice_mean | 0.0025 | 0.000141591 | 0.0286585 | 0.000123707 | 0.0314238 |
| BankBailedOut_share | 0.0004 | 0.000277778 | 0.000324 | 6.95956e-05 | 0.00107137 |
| BankBailoutAmount_mean | 0.04 | 0.00873036 | 0.0800805 | 0.00208492 | 0.130896 |
| BankResolutionHaircut_mean | 0.0001 | 0.000176425 | 0.0017573 | 5.06736e-05 | 0.0020844 |
| BankResolved_share | 0.0009 | 0.00111111 | 0.0036 | 0.00105143 | 0.00666254 |
| Bank_Equity_mean | 56.25 | 30.0562 | 62.7692 | 5.83906 | 154.914 |
| Consumption_mean | 100 | 122.947 | 1349.71 | 103.546 | 1676.2 |
| DefaultsHH_rate | 4e-06 | 7.09259e-07 | 2.07025e-05 | 4.38387e-07 | 2.58501e-05 |
| Employment_mean | 25 | 50.7143 | 1063.41 | 47.6862 | 1186.81 |
| HH_Deposit_mean | 1 | 0.359763 | 1.39298 | 0.154157 | 2.9069 |
| Output_mean | 100 | 165.101 | 502.538 | 114.69 | 882.329 |
| Transfers_mean | 4 | 0.826865 | 6.69668 | 0.702872 | 12.2264 |

## 3) SA Results (newly implausible share)
| Component | Reduction | Points | Newly implausible | Share |
|---|---:|---:|---:|---:|
| CU | 10% | 42 | 0 | 0.0000 |
| CU | 20% | 42 | 0 | 0.0000 |
| CU | 30% | 42 | 0 | 0.0000 |
| CU | 40% | 42 | 0 | 0.0000 |
| EV | 10% | 42 | 0 | 0.0000 |
| EV | 20% | 42 | 0 | 0.0000 |
| EV | 30% | 42 | 0 | 0.0000 |
| EV | 40% | 42 | 0 | 0.0000 |
| MD | 10% | 42 | 0 | 0.0000 |
| MD | 20% | 42 | 3 | 0.0714 |
| MD | 30% | 42 | 9 | 0.2143 |
| MD | 40% | 42 | 9 | 0.2143 |
| OU | 10% | 42 | 0 | 0.0000 |
| OU | 20% | 42 | 0 | 0.0000 |
| OU | 30% | 42 | 0 | 0.0000 |
| OU | 40% | 42 | 0 | 0.0000 |

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
