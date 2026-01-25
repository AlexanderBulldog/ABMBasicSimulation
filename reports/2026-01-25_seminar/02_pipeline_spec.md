# 2) Спецификация пайплайна (данные → эмулятор → history matching → отчётный набор)

## 2.1 Артефакты и каталоги

- baseline: `output/`
- seminar: `output_seminar/`

Типовая структура:

- `datasets/` — LHS‑прогоны ABM (табличные данные)
- `results/` — эмулятор, history matching, refined intervals, отчётные наборы
- `ml_datasets/` — датасеты для ML‑политик
- `models/` — обученные ML‑модели

## 2.2 Шаги пайплайна

### Шаг A — LHS дизайн и прогоны ABM

Скрипт: `scripts/run_lhs.py`  
Оператор прогонов: `scripts/run_operator.py` (`run_batch`, `run_model`)

Что происходит:

1) строится LHS выборка `theta` в пределах `PARAM_BOUNDS`  
2) каждая точка прогоняется на нескольких `seed`  
3) вычисляются метрики выхода  
4) выставляется флаг `bad_run` (коллапсы/NaN/небаланс и т.д.)

Выход:

- `output_seminar/datasets/lhs_runs.csv`
- `output_seminar/datasets/lhs_runs_wave2.csv` (wave2)

### Шаг B — обучение эмулятора

Скрипт: `scripts/train_emulator.py`

Используемые модели:

- **GPR (GaussianProcessRegressor)** — основной эмулятор, умеет `predict(..., return_std=True)`
- **RandomForestRegressor** — baseline/контроль качества

Что важно:

- перед обучением фильтруются строки `bad_run=True` и строки с `error`
- для финансовых метрик может применяться преобразование `log1p`/`signed_log1p` (см. `choose_transform`)
- качество считается по KFold CV (R² mean/std)

Выход:

- `.../results/emulator_scores.csv`

### Шаг C — history matching (NROY)

Скрипт: `scripts/train_emulator.py` (функции history matching)

Таргеты/допуски задаются в:

- `scripts/targets_report.json`

Для каждой метрики с таргетом строится improbability:

- `I = |mu - target| / sqrt(sigma_obs^2 + sigma_model^2 + sigma_emul^2)`

Затем берётся `I_max` по метрикам и определяется:

- `nroy = (I_max < improb_threshold)`

Выход:

- `.../results/history_matching.csv`
- `.../results/refined_intervals.csv` (интервалы параметров по NROY)

### Шаг D — wave2 (итерация)

Скрипт: `scripts/run_lhs.py` с `--bounds-csv .../refined_intervals.csv`  
По умолчанию берутся границы `p05–p95` внутри NROY (и клипуются базовыми bounds).

### Шаг E — отчётный набор (representative report set)

Скрипт: `scripts/make_report_set.py`

Особенность версии для семинара:

- выбранные точки **валидируются полной симуляцией на всех seed** (чтобы не было “short‑run ок / long‑run collapse”).

Выход:

- `output_seminar/results/report_set_wave2/representative_points.csv`
- `output_seminar/results/report_set_wave2/representative_summary.csv`
- `output_seminar/results/report_set_wave2/rep_run_XX_seedY.csv` (временные ряды)

### Шаг F — ML‑политика фирм (markup policy)

- датасет: `scripts/build_firm_policy_dataset.py`
- обучение: `scripts/train_firm_policy.py` (GradientBoostingRegressor)

Выход:

- `output_seminar/ml_datasets/firm_policy.csv`
- `output_seminar/models/firm_policy.pkl`

## 2.3 Настройки конкретных прогонов

### baseline (по артефактам)

Из размеров датасета `output/datasets/lhs_runs.csv`:

- `n=200` (т.к. 200 × 3 seeds = 600 строк)
- `seeds=0,1,2`

Остальные настройки соответствуют дефолтам `scripts/run_lhs.py` и `scripts/run_operator.py`:

- `steps=150`, `window=30`, `balance_ok_threshold=0.5`

### seminar (зафиксировано при запуске)

- wave1: `n=250`, `seeds=0,1,2`, `steps=150`, `window=30`, `balance_ok_threshold=0.6`
- history matching: `min_r2_for_history_matching=0.25`
- report set validation: `steps=300`, `window=50`, `balance_ok_threshold=0.98`

