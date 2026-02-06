# Calibration summary (wave1 vs wave2)
## Emulator quality (wave1)
- Trained metrics: 22, skipped: 4
- GPR CV R2 mean: 0.607, median: 0.671

## History matching (report targets)
- Wave1 NROY: 53.3% | I_max median=2.902, p95=4.336, max=5.069
- Wave2 NROY: 48.5% | I_max median=3.053, p95=4.147, max=5.435

## Interval shrink (refined_intervals.csv)
- Wave1 shrink mean=1.54%, max=3.62%
- Wave2 shrink mean=13.63%, max=17.39%

## Model-pathology shares (raw LHS datasets)
| Metric | Wave1 | Wave2 |
|---|---:|---:|
| Employment_mean==0 | 28.3% | 24.1% |
| Output_mean==0 | 20.9% | 16.4% |
| Consumption_mean==0 | 20.9% | 16.4% |
| HH_Deposit_mean==0 | 40.4% | 39.2% |
| BankFailed_share>0 | 0.0% | 0.0% |
| BankResolved_share>0 | 90.5% | 93.5% |
| BankBailedOut_share>0 | 71.3% | 72.5% |

## Data quality (bad_run-filtered)
- Wave1 bad_run share: 28.3% (538/750 kept)
- Wave2 bad_run share: 24.1% (569/750 kept)

### Pathologies among good runs only
| Metric | Wave1(good) | Wave2(good) |
|---|---:|---:|
| Employment_mean==0 | 0.0% | 0.0% |
| Output_mean==0 | 0.0% | 0.0% |
| Consumption_mean==0 | 0.0% | 0.0% |
| HH_Deposit_mean==0 | 17.8% | 20.4% |
| BankFailed_share>0 | 0.0% | 0.0% |
| BankResolved_share>0 | 86.8% | 91.4% |
| BankBailedOut_share>0 | 60.0% | 63.8% |

## Core distribution snapshots (wave2)
- `Employment_mean`: 0%=0, 5%=0, 50%=21, 95%=76.53, 100%=93
- `Output_mean`: 0%=0, 5%=0, 50%=30.74, 95%=107, 100%=153.2
- `Consumption_mean`: 0%=0, 5%=0, 50%=25.06, 95%=104.1, 100%=135
- `AvgPrice_mean`: 0%=0.5575, 5%=0.6041, 50%=0.882, 95%=1.251, 100%=1.453
- `DefaultsHH_rate`: 0%=0, 5%=0.0008, 50%=0.005333, 95%=0.007933, 100%=0.009667
- `BankResolutionHaircut_mean`: 0%=0, 5%=0, 50%=0.02909, 95%=0.07894, 100%=0.1079
- `BankBailedOut_share`: 0%=0, 5%=0, 50%=0.01333, 95%=0.1333, 100%=0.2333

