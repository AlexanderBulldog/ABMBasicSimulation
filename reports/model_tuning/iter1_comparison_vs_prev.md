# Iter1 Comparison vs Iter0 (FULL)

Сравнение построено скриптом `scripts/compare_model_iters.py`.

## Compared runs
- Baseline: `output_model_tune_baseline_full` (full)
- Iter1: `output_model_tune_iter1_full` (full)

## Result Table
| Metric | Baseline | Iter1_full | Delta (Iter1 - Baseline) |
|---|---:|---:|---:|
| NROY_pct | 54.9063 | 98.7458 | +43.8395 |
| Imax_median | 2.8735 | 1.8382 | -1.0353 |
| bad_run_pct_w2 | 24.4167 | 0.3333 | -24.0834 |
| MD share@max_reduction | 0.2229 | 0.0771 | -0.1458 |
| OU share@max_reduction | 0.1265 | 0.0135 | -0.1130 |
| Price fail-driver share | 0.8313 | 0.0000 | -0.8313 |
| confirmatory_nroy_delta_pp | 4.7711 | -1.8343 | -6.6054 |
| confirmatory_top2_stable | True | True | = |
| legacy_all_gates_pass | PASS | FAIL | - |

## Structural contour (B)
- PASS: `PriceDispersion median in [0.03,0.30]`.
- PASS: `|InventoryGap| p95 <= 1.5`.
- FAIL: `CreditRejections p90 <= 10` (`p90=105`).
- PASS: `FirmDowntimeShare max <= 0.20`.

## Legacy contour (A)
- Единственный критический провал: `NROY wave2 in [25,60]%` (получено `98.75%`).

## Retain / Revert decision
- **Revert: нет.**
- **Retain: да, условно.** Ядро модели дало сильное структурное улучшение (MD/OU/price-driver/bad_run/confirmatory), но требует отдельного итерационного ужесточения калибровочного контура (sigma/targets), чтобы вернуть дискриминацию NROY в рабочий диапазон.
