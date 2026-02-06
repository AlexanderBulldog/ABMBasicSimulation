# 3) Результаты и сравнение (baseline vs seminar)

## 3.1 Границы параметров (было vs стало)

baseline bounds (восстановлены по `output/results/refined_intervals.csv` `initial_low/high`):

| param | baseline_low | baseline_high |
|---|---:|---:|
| alpha_mean | 0.55 | 0.95 |
| alpha_std | 0.01 | 0.15 |
| wage | 0.8 | 1.5 |
| productivity | 0.8 | 2.2 |
| loan_rate | 0.01 | 0.08 |
| deposit_rate | 0.0 | 0.03 |
| bank_credit_multiplier | 4.0 | 12.0 |
| hh_debt_cap_multiplier | 2.0 | 5.0 |
| firm_debt_cap_multiplier | 1.5 | 5.0 |
| adaptation_rate | 0.2 | 1.0 |
| price_elasticity | 0.8 | 3.0 |
| skill_wage_weight | 0.0 | 0.5 |
| demand_smoothing | 0.05 | 0.4 |

seminar bounds (текущие `scripts/run_operator.py:21`):

| param | seminar_low | seminar_high |
|---|---:|---:|
| alpha_mean | 0.57 | 0.92 |
| alpha_std | 0.03 | 0.13 |
| wage | 0.9 | 1.45 |
| productivity | 0.95 | 1.95 |
| loan_rate | 0.012 | 0.073 |
| deposit_rate | 0.001 | 0.028 |
| bank_credit_multiplier | 4.2 | 11.5 |
| hh_debt_cap_multiplier | 2.1 | 4.8 |
| firm_debt_cap_multiplier | 1.6 | 4.8 |
| adaptation_rate | 0.25 | 0.95 |
| price_elasticity | 0.95 | 2.9 |
| skill_wage_weight | 0.03 | 0.49 |
| demand_smoothing | 0.06 | 0.35 |

## 3.2 Качество данных LHS (коллапсы и фильтрация)

### baseline (без фильтра)

- wave1 (`output/datasets/lhs_runs.csv`): `Employment_mean==0` 25.0%, `Output_mean==0` 17.5%, `Consumption_mean==0` 17.5%
- wave2 (`output/datasets/lhs_runs_wave2.csv`): `Employment_mean==0` 21.8%, `Output_mean==0` 15.2%, `Consumption_mean==0` 15.2%
- `bad_run` в baseline артефактах не использовался как строгий фильтр (0%).

### seminar (с фильтром `bad_run`)

Сводка: `output_seminar/results/calibration_summary.md`

- wave1: `bad_run` 28.3% (538/750 строк оставлено для эмулятора/HM)
- wave2: `bad_run` 24.1% (569/750 строк оставлено)
- среди **good runs**: `Employment_mean==0` / `Output_mean==0` / `Consumption_mean==0` = 0.0% (и в wave1, и в wave2)
- но `HH_Deposit_mean==0` среди good runs остаётся заметным:
  - wave1: 17.84%
  - wave2: 20.39%

Интерпретация: пайплайн теперь явно отделяет “вырожденные режимы” (коллапс) от корректных прогонов и не даёт им загрязнять обучающий датасет эмулятора.

## 3.3 Качество эмулятора (wave1)

Эмулятор: `scripts/train_emulator.py` (GPR, контроль — RF).

| run | trained_metrics | skipped_metrics | GPR CV R² mean | GPR CV R² median |
|---|---:|---:|---:|---:|
| baseline | 22 | 4 | 0.641 | 0.761 |
| seminar | 22 | 4 | 0.607 | 0.671 |

Комментарий: в seminar‑версии R² чуть ниже (ожидаемо при более строгой фильтрации/других bounds), но history matching стал существенно “мягче” по `I_max` (см. ниже), а report set стал стабильным.

## 3.4 History matching (NROY) — ключевые числа

| run | wave | rows_used | NROY share | I_max median | I_max p95 | I_max max |
|---|---|---:|---:|---:|---:|---:|
| baseline | wave1 | 600 | 0.295 | 3.991 | 5.669 | 7.189 |
| baseline | wave2 | 600 | 0.325 | 3.790 | 4.828 | 5.242 |
| seminar | wave1 | 538 | 0.533 | 2.902 | 4.336 | 5.069 |
| seminar | wave2 | 569 | 0.485 | 3.053 | 4.147 | 5.435 |

Важно: в seminar `rows_used` меньше, т.к. `bad_run=True` вычищается ДО обучения/History Matching.

## 3.5 Сужение интервалов (refined_intervals)

| run | wave | shrink mean | shrink max | топ‑параметры по shrink |
|---|---|---:|---:|---|
| baseline | wave1 | 2.58% | 10.77% | productivity, alpha_std, adaptation_rate, bank_credit_multiplier, hh_debt_cap_multiplier |
| baseline | wave2 | 15.12% | 30.73% | productivity, alpha_std, demand_smoothing, wage, adaptation_rate |
| seminar | wave1 | 1.54% | 3.62% | alpha_std, wage, firm_debt_cap_multiplier, demand_smoothing, price_elasticity |
| seminar | wave2 | 13.63% | 17.39% | alpha_std, wage, hh_debt_cap_multiplier, skill_wage_weight, firm_debt_cap_multiplier |

## 3.6 Репрезентативные траектории (report set): проверка стабильности

### baseline: есть коллапс в “репрезентативном” наборе

Файл: `output/results/report_set/representative_summary.csv`

| rep_id | Employment_min | Output_min | Consumption_min | BalanceOK_min | BankResolved_max | BankBailedOut_max | BankFailed_max |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 63 | 64.69 | 85.72 | 1 | 0.02 | 0 | 0 |
| 1 | 49.78 | 48.78 | 47.01 | 1 | 0 | 0 | 0 |
| 2 | 63 | 90.73 | 91.86 | 1 | 0 | 0 | 0 |
| 3 | 79 | 66.73 | 80.69 | 1 | 0 | 0 | 0 |
| 4 | 0 | 0 | 0 | 1 | 0.05 | 0.04667 | 0 |
| 5 | 68 | 101.2 | 89.16 | 1 | 0 | 0 | 0 |

Коллапс: `rep_id=4` (занятость/выпуск/потребление → 0 на длинном горизонте).

### seminar: отчётный набор валидирован на длинном горизонте (коллапсов нет)

Файл: `output_seminar/results/report_set_wave2/representative_summary.csv`

| rep_id | Employment_min | Output_min | Consumption_min | BalanceOK_min | BankResolved_max | BankBailedOut_max | BankFailed_max |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 63 | 78.29 | 74.34 | 1 | 0.01667 | 0 | 0 |
| 1 | 69 | 81.75 | 79.26 | 1 | 0.01333 | 0 | 0 |
| 2 | 56 | 92.19 | 72.65 | 1 | 0 | 0 | 0 |
| 3 | 71.46 | 81.88 | 89.36 | 1 | 0.01 | 0 | 0 |
| 4 | 61 | 95.47 | 84.51 | 1 | 0.003333 | 0 | 0 |
| 5 | 80 | 89.28 | 110.9 | 1 | 0.02 | 0 | 0 |

## 3.7 ML‑политика фирм (семинарная ветка)

Датасет: `output_seminar/ml_datasets/firm_policy.csv`

- строк: 27 000 (6 rep_id × 3 seed × 10 фирм × ~150 шагов хвоста)
- целевая переменная: `realized_markup` (клипуется в обучении до 0.5)
- распределение `realized_markup`:
  - min/median: 0.05
  - p95/max: 0.15
  - mean: ~0.0779, std: ~0.0330

Модель: `output_seminar/models/firm_policy.pkl` (GradientBoostingRegressor).

