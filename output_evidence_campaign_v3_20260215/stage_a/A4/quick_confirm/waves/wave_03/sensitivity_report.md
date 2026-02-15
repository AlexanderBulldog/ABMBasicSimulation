# Uncertainty Sensitivity Report (2026-02-15)

## 1) Setup
- Dataset: `C:\Users\Alexander\Desktop\Education\Aspa\code\output_evidence_campaign_v3_20260215\stage_a\A4\quick_confirm\waves\wave_03\lhs_runs_wave3.csv`
- Implausibility threshold: `2.55`
- SA domain: `nroy`
- Baseline points used: `231`
- Reductions: `10%, 20%, 30%, 40%`

## 2) Baseline Uncertainty Decomposition
| Metric | OU var | EV var | MD var | CU var (median) | Total var (median) |
|---|---:|---:|---:|---:|---:|
| AvgPrice_mean | 0.0025 | 0.000496206 | 0.00581101 | 0.000333343 | 0.00914056 |
| BankBailedOut_share | 0.0004 | 0.000162037 | 0.00088209 | 7.22034e-05 | 0.00151633 |
| BankBailoutAmount_mean | 0.04 | 0.100135 | 0.0527529 | 0.0386228 | 0.231511 |
| BankResolutionHaircut_mean | 0.0001 | 0.000228481 | 0.00039204 | 8.60747e-05 | 0.000806596 |
| BankResolved_share | 0.0009 | 0.00122454 | 0.00245025 | 0.000659072 | 0.00523386 |
| Consumption_mean | 100 | 42.1841 | 170.773 | 34.9028 | 347.859 |
| DefaultsHH_rate | 4e-06 | 3.14537e-06 | 8.8209e-06 | 1.44475e-06 | 1.7411e-05 |
| HH_Deposit_mean | 1 | 0.0926542 | 1.31882 | 0.0117058 | 2.42318 |
| Output_mean | 100 | 51.6649 | 170.773 | 36.9191 | 359.357 |
| PriceDispersion_mean | 0.0036 | 0.000472408 | 0.0141134 | 0.000297101 | 0.0184829 |
| Transfers_mean | 4 | 0.343807 | 5.27529 | 0.295269 | 9.91437 |

## 3) SA Results (newly implausible share)
| Component | Reduction | Points | Newly implausible | Share |
|---|---:|---:|---:|---:|
| CU | 10% | 231 | 0 | 0.0000 |
| CU | 20% | 231 | 12 | 0.0519 |
| CU | 30% | 231 | 12 | 0.0519 |
| CU | 40% | 231 | 15 | 0.0649 |
| EV | 10% | 231 | 3 | 0.0130 |
| EV | 20% | 231 | 15 | 0.0649 |
| EV | 30% | 231 | 15 | 0.0649 |
| EV | 40% | 231 | 24 | 0.1039 |
| MD | 10% | 231 | 18 | 0.0779 |
| MD | 20% | 231 | 42 | 0.1818 |
| MD | 30% | 231 | 66 | 0.2857 |
| MD | 40% | 231 | 87 | 0.3766 |
| OU | 10% | 231 | 12 | 0.0519 |
| OU | 20% | 231 | 24 | 0.1039 |
| OU | 30% | 231 | 36 | 0.1558 |
| OU | 40% | 231 | 51 | 0.2208 |

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
