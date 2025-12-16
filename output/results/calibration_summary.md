# Calibration summary (wave1 vs wave2)
## Emulator quality (wave1)
- Trained metrics: 22, skipped: 4
- GPR CV R2 mean: 0.617, median: 0.741

## History matching (report targets)
- Wave1 NROY: 27.3% | I_max median=3.786, p95=5.171, max=7.496
- Wave2 NROY: 38.8% | I_max median=3.470, p95=4.680, max=5.166

## Interval shrink (refined_intervals.csv)
- Wave1 shrink mean=2.27%, max=6.44%
- Wave2 shrink mean=15.66%, max=31.93%

## Model-pathology shares (raw LHS datasets)
| Metric | Wave1 | Wave2 |
|---|---:|---:|
| Employment_mean==0 | 27.4% | 22.7% |
| Output_mean==0 | 21.2% | 16.1% |
| Consumption_mean==0 | 21.2% | 16.1% |
| HH_Deposit_mean==0 | 43.2% | 41.1% |
| BankFailed_share>0 | 0.0% | 0.0% |
| BankResolved_share>0 | 90.7% | 92.3% |
| BankBailedOut_share>0 | 63.1% | 68.1% |

## Core distribution snapshots (wave2)
- `Employment_mean`: 0%=0, 5%=0, 50%=22, 95%=85.03, 100%=97.97
- `Output_mean`: 0%=0, 5%=0, 50%=33.78, 95%=110.4, 100%=160.5
- `Consumption_mean`: 0%=0, 5%=0, 50%=26.79, 95%=117.6, 100%=155.1
- `AvgPrice_mean`: 0%=0.533, 5%=0.5901, 50%=0.9165, 95%=1.402, 100%=1.653
- `DefaultsHH_rate`: 0%=0, 5%=0.0003636, 50%=0.003909, 95%=0.00657, 100%=0.008773
- `BankResolutionHaircut_mean`: 0%=0, 5%=0, 50%=0.02311, 95%=0.05449, 100%=0.08536
- `BankBailedOut_share`: 0%=0, 5%=0, 50%=0.009091, 95%=0.1, 100%=0.1818

