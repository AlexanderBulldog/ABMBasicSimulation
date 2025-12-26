# Calibration summary (wave1 vs wave2)
## Emulator quality (wave1)
- Trained metrics: 22, skipped: 4
- GPR CV R2 mean: 0.641, median: 0.761

## History matching (report targets)
- Wave1 NROY: 29.5% | I_max median=3.991, p95=5.669, max=7.189
- Wave2 NROY: 32.5% | I_max median=3.790, p95=4.828, max=5.242

## Interval shrink (refined_intervals.csv)
- Wave1 shrink mean=2.58%, max=10.77%
- Wave2 shrink mean=15.12%, max=30.73%

## Model-pathology shares (raw LHS datasets)
| Metric | Wave1 | Wave2 |
|---|---:|---:|
| Employment_mean==0 | 25.0% | 21.8% |
| Output_mean==0 | 17.5% | 15.2% |
| Consumption_mean==0 | 17.5% | 15.2% |
| HH_Deposit_mean==0 | 43.5% | 39.0% |
| BankFailed_share>0 | 0.0% | 0.0% |
| BankResolved_share>0 | 89.2% | 89.5% |
| BankBailedOut_share>0 | 64.3% | 65.3% |

## Core distribution snapshots (wave2)
- `Employment_mean`: 0%=0, 5%=0, 50%=22, 95%=75.01, 100%=96
- `Output_mean`: 0%=0, 5%=0, 50%=36.82, 95%=106.6, 100%=151.1
- `Consumption_mean`: 0%=0, 5%=0, 50%=26.73, 95%=102.7, 100%=147.7
- `AvgPrice_mean`: 0%=0.4597, 5%=0.5488, 50%=0.8179, 95%=1.435, 100%=1.592
- `DefaultsHH_rate`: 0%=0, 5%=0.0005333, 50%=0.005133, 95%=0.008203, 100%=0.0104
- `BankResolutionHaircut_mean`: 0%=0, 5%=0, 50%=0.03066, 95%=0.07677, 100%=0.1036
- `BankBailedOut_share`: 0%=0, 5%=0, 50%=0.01333, 95%=0.14, 100%=0.24

